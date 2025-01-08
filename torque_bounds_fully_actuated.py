import numpy as np
import adam
from adam.casadi import KinDynComputations
from casadi import MX, vertcat, norm_2, Function
from urdf_parser_py.urdf import URDF
import casadi as cs
import parser
import copy
import random
import pickle
import datetime
import os
import tqdm
import matplotlib.pyplot as plt


class AdamModel:
    def __init__(self, params, n_dofs=6):
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
        kin_dyn = KinDynComputations(params.robot_urdf, joint_names[:n_dofs], robot.get_root())       
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

        
    
        joint_lower=joint_lower[:n_dofs]
        joint_upper=joint_upper[:n_dofs]
        joint_velocity=joint_velocity[:n_dofs]
        joint_effort=joint_effort[:n_dofs]  

        self.tau_min = - joint_effort
        self.tau_max = joint_effort
        self.x_min = np.hstack([joint_lower, - joint_velocity])
        self.x_max = np.hstack([joint_upper, joint_velocity])

        # EE target
        #self.ee_ref = self.jointToEE(np.zeros(self.nx))
        self.ee_ref = np.array([0.6, 0.28, 0.078])

        # NN model (viability constraint)
        self.l4c_model = None
        self.nn_model = None
        self.nn_func = None

        self.robot = robot
        self.joint_names = joint_names
    



class MaxVelOCP:
    """ Define OCP problem and solver (IpOpt) """
    def __init__(self, model, n_steps,not_locked):
        self.params = model.params
        self.model = model
        self.nq = model.nq

        N = self.params.N
        Q_vel = 1e2*0
        opti = cs.Opti()
        x_init = opti.parameter(model.nx)
        vel_dir = opti.parameter(model.nq)
        

        # Define decision variables
        X, U = [], []
        X += [opti.variable(model.nx)]
        opti.subject_to(opti.bounded(model.x_min, X[-1], model.x_max))
        cost = -cs.dot(vel_dir,X[0][self.nq:])

        for k in range(n_steps):
            X += [opti.variable(model.nx)]
            opti.subject_to(opti.bounded(model.x_min, X[-1], model.x_max))
            cost += Q_vel * (X[-1][not_locked+robot.nq]**2)
            for l in range(model.nq):
                if l != not_locked:
                    #opti.subject_to(X[-1][model.nq+l] == 0)
                    opti.subject_to(X[-1][l] == X[-2][l])
            U += [opti.variable(model.nu)]

        opti.subject_to(X[0][:self.nq] == x_init[:self.nq])
        opti.subject_to(((cs.MX.eye(self.nq)-(vel_dir@vel_dir.T))@X[0][self.nq:])==cs.MX.zeros(self.nq,1))
        for k in range(n_steps+1):

            if k < n_steps:
                # Dynamics constraint
                opti.subject_to(X[k + 1] == model.f_fun(X[k], U[k]))
                # Torque constraints
                opti.subject_to(opti.bounded(model.tau_min, model.tau_fun(X[k], U[k]), model.tau_max))
        
        opti.subject_to(X[-1]==X[-2])

        self.opti = opti
        self.X = X
        self.U = U
        self.x_init = x_init
        self.vel_dir = vel_dir
        self.cost = cost
        self.additionalSetting()
        opti.minimize(cost)

    def additionalSetting(self):
        pass

    def instantiateProblem(self):
        opti = self.opti
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-6,
            'ipopt.constr_viol_tol': 1e-6,
            'ipopt.compl_inf_tol': 1e-6,
            #'ipopt.hessian_approximation': 'limited-memory',
            # 'detect_simple_bounds': 'yes',
            'ipopt.max_iter': 100,
            #'ipopt.linear_solver': 'ma57',
            'ipopt.sb': 'no'
        }

        opti.solver('ipopt', opts)
        return opti

if __name__ == "__main__":

    params = parser.Parameters('fr3')
    not_locked_joint =0
    robot = AdamModel(params)
    horizon_length = 45
    samples = 80
    samples_i = int(samples/10)
    samples_f = samples_i
    results_angle = []
    results_vel = []

    print(robot.x_min)
    #print(f'inertia {robot.mass(np.eye(4), np.zeros(7))[6:, 6:]}')
    print(f'ee_pos {robot.ee_fun(np.zeros(12))}')

    # torque bound on the studied joint
    divider = 1
    robot.tau_max[not_locked_joint]=robot.tau_max[not_locked_joint]/divider
    robot.tau_min[not_locked_joint]=robot.tau_min[not_locked_joint]/divider
    for i in range(len(robot.tau_max)):
        if i != not_locked_joint:
            robot.tau_max[i] = robot.tau_max[i] *100
            robot.tau_min[i] = robot.tau_min[i] *100
            robot.x_min[i] = -100
            robot.x_min[i+robot.nq] = robot.x_min[i+robot.nq] *100
            robot.x_max[i] = 100
            robot.x_max[i+robot.nq] = robot.x_max[i+robot.nq] *100



    x0_s_i = np.linspace(robot.x_min[not_locked_joint],robot.x_min[not_locked_joint]+0.1,samples_i)
    x0_s = np.linspace(robot.x_min[not_locked_joint]+0.1,robot.x_max[not_locked_joint]-0.1,samples)
    x0_s_f = np.linspace(robot.x_max[not_locked_joint]-0.1,robot.x_max[not_locked_joint],samples_f)
    x0_s=np.hstack((x0_s_i,x0_s,x0_s_f))

    x0_full = (robot.x_max + robot.x_min)/2 
    #x0_full[:robot.nq] = np.zeros(robot.nq)
    
    # constrain particular states if needed
    x0_full[1] = 0.8
    progress_bar = tqdm.tqdm(total=x0_s.shape[0], desc='Sampling started')
    for i in range(x0_s.shape[0]):
        # x0 = (robot.x_max-robot.x_min)*np.random.random_sample((robot.nx,)) + robot.x_min*np.ones((robot.nx,))
        # x0[:robot.nq] = x0_s[i]
        x0_init = copy.copy(x0_full)
        x0_init[not_locked_joint] = x0_s[i]
        x0_init[not_locked_joint+robot.nq] = ((robot.x_max[not_locked_joint+robot.nq]-robot.x_min[not_locked_joint+robot.nq])*np.random.random_sample((1,)) + robot.x_min[not_locked_joint+robot.nq])[0]
        vel_direction = x0_init[robot.nq:]/np.linalg.norm(x0_init[robot.nq:])

        ocp_form= MaxVelOCP(robot,horizon_length,not_locked_joint)
        ocp = ocp_form.instantiateProblem()
        ocp.set_value(ocp_form.x_init, x0_init)
        ocp.set_value(ocp_form.vel_dir, vel_direction)
        try:
            sol = ocp.solve()
            results_angle.append(x0_init[not_locked_joint])
            results_vel.append(sol.value(ocp_form.X[0][not_locked_joint+robot.nq]))
            print(sol.value(ocp_form.X[0][not_locked_joint+robot.nq]))
        except:
            print('Failed')

        ocp_form= MaxVelOCP(robot,horizon_length,not_locked_joint)
        ocp = ocp_form.instantiateProblem()
        ocp.set_value(ocp_form.x_init, x0_init)
        ocp.set_value(ocp_form.vel_dir, -vel_direction)
        try:
            sol = ocp.solve()
            results_angle.append(x0_init[not_locked_joint])
            results_vel.append(sol.value(ocp_form.X[0][not_locked_joint+robot.nq]))
            print(sol.value(ocp_form.X[0][not_locked_joint+robot.nq]))
        except:
            print('Failed')
        progress_bar.update(1)

    progress_bar.close()

    plt.figure(f'joint{not_locked_joint}, u_max {robot.tau_max}')
    plt.title(f'joint{not_locked_joint}, u_max {robot.tau_max}')
    plt.scatter(results_angle, results_vel,color='blue',marker='o')
    plt.show()
