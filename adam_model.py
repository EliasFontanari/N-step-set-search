import adam
from adam.casadi import KinDynComputations
from casadi import MX, vertcat, norm_2, Function
import torch
import torch.nn as nn
import l4casadi as l4c
import numpy as np
from urdf_parser_py.urdf import URDF
from neural_network import NeuralNetwork
import casadi as cs
from numpy.random import uniform
from copy import deepcopy

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU(), ub=None):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size),
            activation,
        )
        self.ub = ub if ub is not None else 1.0

    def forward(self, x):
        out = self.linear_stack(x) * self.ub
        return out


class AdamModel:
    def __init__(self, params, n_dofs=False, not_locked=None):
        self.params = params
        # Robot dynamics with Adam (IIT)
        robot = URDF.from_xml_file(params.robot_urdf)
        try:
            n_dofs = n_dofs if n_dofs else len(robot.joints)
            if n_dofs > len(robot.joints) or n_dofs < 1:
                raise ValueError
        except ValueError:
            print(f'\nInvalid number of degrees of freedom! Must be > 1 and <= {len(robot.joints)}\n')
            exit()

        robot_joints = []
        #jj=0
        #while jj < n_dofs:
        for jointt in robot.joints:
            if jointt.type != 'fixed':
                robot_joints.append(jointt)
                    # jj += 1
                    # if jj == n_dofs:
                    #     break
        joint_names = [joint.name for joint in robot_joints]
        if not_locked == None:
            kin_dyn = KinDynComputations(params.robot_urdf, joint_names[:n_dofs], robot.get_root())  
        else:
            kin_dyn = KinDynComputations(params.robot_urdf, [joint_names[not_locked]], robot.get_root())     
        kin_dyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
        self.gravity = kin_dyn.gravity_term_fun()
        self.mass = kin_dyn.mass_matrix_fun()                           # Mass matrix
        self.bias = kin_dyn.bias_force_fun()                            # Nonlinear effects  
        # print(kin_dyn.rbdalgos.model.links.keys())
        self.fk = kin_dyn.forward_kinematics_fun(params.frame_name)     # Forward kinematics
        self.jac = kin_dyn.jacobian_fun(params.frame_name)
        self.jac_dot = kin_dyn.jacobian_dot_fun(params.frame_name)

        # nq = len(joint_names)
        nq=n_dofs



        self.x = MX.sym("x", nq * 2)
        self.x_dot = MX.sym("x_dot", nq * 2)
        self.u = MX.sym("u", nq)
        self.p = MX.sym("p", 1)     # Safety margin for the NN model
        # Double integrator
        self.f_disc = vertcat(
            self.x[:nq] + params.dt * self.x[nq:] + 0.5 * params.dt**2 * self.u,
            self.x[nq:] + params.dt * self.u
        ) 
        self.f_fun = Function('f', [self.x, self.u], [self.f_disc])

        self.nx = nq*2
        self.nu = nq
        self.ny = self.nx + self.nu
        self.nq = nq
        self.nv = nq
        self.np = 3

        # Real dynamics
        H_b = np.eye(4)
        self.tau = self.mass(H_b, self.x[:nq])[6:, 6:] @ self.u + \
                   self.bias(H_b, self.x[:nq], np.zeros(6), self.x[nq:])[6:]
        self.tau_fun = Function('tau', [self.x, self.u], [self.tau])

        self.f_forward = vertcat(
            self.x[nq:],
            cs.inv(self.mass(H_b, self.x[:nq])[6:, 6:]) 
            @ (self.u - self.bias(H_b, self.x[:nq], np.zeros(6), self.x[nq:])[6:])
        )
        self.f_fun_forw = Function('f', [self.x, self.u], [self.f_forward])

        # EE position (global frame)
        T_ee = self.fk(np.eye(4), self.x[:nq])
        self.t_loc = np.array([0.035, 0., 0.])
        self.t_glob = T_ee[:3, 3] + T_ee[:3, :3] @ self.t_loc
        self.ee_fun = Function('ee_fun', [self.x], [self.t_glob])

        # Joint limits
        joint_lower = np.array([joint.limit.lower for joint in robot_joints])
        joint_upper = np.array([joint.limit.upper for joint in robot_joints])
        joint_velocity = np.array([joint.limit.velocity for joint in robot_joints]) 

        if self.params.urdf_name=='z1':
            joint_effort = np.array([2., 23., 10., 4.])
        else:
            joint_effort = np.array([joint.limit.effort for joint in robot_joints])
            # if self.params.urdf_name=='fr3':
            #     joint_effort = np.array([1.])

        
        if not_locked == None:
            joint_lower=joint_lower[:n_dofs]
            joint_upper=joint_upper[:n_dofs]
            joint_velocity=joint_velocity[:n_dofs]
            joint_effort=joint_effort[:n_dofs]  
        else:
            joint_lower=joint_lower[not_locked]
            joint_upper=joint_upper[not_locked]
            joint_velocity=joint_velocity[not_locked]
            joint_effort=joint_effort[not_locked]


        self.tau_min = - joint_effort
        self.tau_max = joint_effort
        self.x_min = np.hstack([joint_lower, - joint_velocity])
        self.x_max = np.hstack([joint_upper, joint_velocity])

        # EE target
        self.ee_ref = self.jointToEE(np.zeros(self.nx))
        self.ee_ref = np.array([0.6, 0.28, 0.078])

        # NN model (viability constraint)
        self.l4c_model = None
        self.nn_model = None
        self.nn_func = None

        # Cartesian constraints
        self.obs_string = '_obs' if params.obs_flag else ''

        self.robot = robot
        self.joint_names = joint_names

    
    def define_problem(self, conf):
        N = conf.N
        ee_des = conf.ee_des

        opti = cs.Opti()
        param_x_init = opti.parameter(self.nx) # initial state
        param_state_cost_flag = opti.parameter(1) # boolean flag to activate/deactivate state cost
        cost = 0

        # create all the decision variables
        X, U = [], []
        X += [opti.variable(self.nx)]
        for k in range(N): 
            X += [opti.variable(self.nx)]
            opti.subject_to( opti.bounded(self.x_min, X[-1], self.x_max) )
        for k in range(N): 
            U += [opti.variable(self.nq)]

        opti.subject_to(X[0] == param_x_init)
        dist_b = []
        for k in range(N+1):     
            # print("Compute cost function")
            ee_pos = self.ee_fun(X[k])
            cost += param_state_cost_flag * ((ee_pos - ee_des).T @ conf.Q @ (ee_pos - ee_des))

            if(k<N):
                cost += U[k].T @ conf.R @ U[k]

                # print("Add dynamics constraints")
                opti.subject_to(X[k+1] == self.f_fun(X[k], U[k]))

                # print("Add torque constraints")
                opti.subject_to( opti.bounded(self.tau_min, self.tau_fun(X[k], U[k]), self.tau_max))
        
            if conf.obstacles is not None and conf.obs_flag:
                # Collision avoidance with obstacles
                for obs in conf.obstacles:
                    t_glob = self.ee_fun(X[k])
                    if obs['name'] == 'floor':
                        lb = obs['bounds'][0]
                        ub = obs['bounds'][1]
                        opti.subject_to( opti.bounded(lb, t_glob[2], ub))
                    elif obs['name'] == 'ball':
                        lb = obs['bounds'][0]
                        ub = obs['bounds'][1]
                        dist_b.append((t_glob - obs['position']).T @ (t_glob - obs['position']))
                        opti.subject_to( opti.bounded(lb, dist_b[k], ub))

        #opti.subject_to( self.nn_func(X[N]) >= param_nn_lb)

        opti.minimize(cost)

        self.opti = opti
        self.param_x_init = param_x_init
        self.param_state_cost_flag = param_state_cost_flag
        #self.param_nn_lb = param_nn_lb
        self.X = X
        self.U = U
        self.dist_b = dist_b
        self.cost = cost
        return opti


    def instantiate_problem(self, conf):
        opti = self.opti
        opts = {
            "ipopt.print_level": 0,
            "ipopt.tol": conf.solver_tol,
            "ipopt.constr_viol_tol": conf.solver_tol,
            "ipopt.compl_inf_tol": conf.solver_tol,
            "print_time": 0,                # print information about execution time
            "detect_simple_bounds": True,
            "ipopt.max_iter": conf.solver_max_iter
        }
        opti.solver("ipopt", opts)

        #opti.set_value(self.param_nn_lb, 0.0)
        opti.set_value(self.param_state_cost_flag, 1.0)
        
        return opti


    def sample_feasible_state(self, conf):
        initial_state_found = False
        while(initial_state_found==False):
            x0 = uniform(self.x_min, self.x_max)
            nn_0 = self.nn_func(x0).toarray()[0,0]
            t_glob = self.ee_fun(x0)
            for obs in conf.obstacles:
                if obs['name'] == 'floor':
                    lb = obs['bounds'][0]
                    floor_dist = t_glob[2] - lb
                elif obs['name'] == 'ball':
                    lb = obs['bounds'][0]
                    ball_dist = (t_glob - obs['position']).T @ (t_glob - obs['position']) - lb
            if(nn_0>0.0 and floor_dist>0.0 and ball_dist>0.0):
                initial_state_found = True
        return x0, nn_0
    

    def jointToEE(self, x):
        return np.array(self.ee_fun(x))
    
    def setNNmodel(self):
        nls = {
            'relu': torch.nn.ReLU(),
            'elu': torch.nn.ELU(),
            'tanh': torch.nn.Tanh(),
            'gelu': torch.nn.GELU(approximate='tanh'),
            'silu': torch.nn.SiLU()
        }
        act = self.params.act
        act_fun = nls[act]

        if act in ['tanh']: #, 'sine']:
            ub = max(self.x_max[self.nq:]) * np.sqrt(self.nq)
        else:
            ub = 1

        model = NeuralNetwork(self.nx, 256, 1, act_fun, ub)
        # print(model)
        # print(f'{self.params.NN_DIR}{self.nq}dof_{act}{self.obs_string}.pt')
        nn_data = torch.load(f'{self.params.NN_DIR}{self.nq}dof_{act}{self.obs_string}.pt',
                             map_location=torch.device('cpu'))
        model.load_state_dict(nn_data['model'])

        x_cp = deepcopy(self.x)
        x_cp[self.nq] += self.params.eps
        vel_norm = norm_2(x_cp[self.nq:])
        pos = (x_cp[:self.nq] - nn_data['mean']) / nn_data['std']
        vel_dir = x_cp[self.nq:] / vel_norm
        state = vertcat(pos, vel_dir)

        self.l4c_model = l4c.L4CasADi(model,
                                      device='cpu',
                                      name=f'{self.params.urdf_name}_model',
                                      build_dir=f'{self.params.GEN_DIR}nn_{self.params.urdf_name}')
        self.nn_model = self.l4c_model(state) * (100 - self.params.alpha) / 100 - vel_norm
        self.nn_func = Function('nn_func', [self.x], [self.nn_model])
        self.nn_out = Function('out', [self.x], [self.l4c_model(state)])



