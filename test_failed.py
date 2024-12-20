import numpy as np
import casadi as cs
import adam_model
import parser
import copy
import random
import pickle
import datetime
import os
import tqdm

class NaiveOCP:
    """ Define OCP problem and solver (IpOpt) """
    def __init__(self, model, n_steps,obstacles=None):
        self.params = model.params
        self.model = model
        self.nq = model.nq
        self.obstacles = obstacles

        N = self.params.N
        opti = cs.Opti()
        x_init = opti.parameter(model.nx)
        cost = 0

        # Define decision variables
        X, U = [], []
        X += [opti.variable(model.nx)]
        for k in range(n_steps):
            X += [opti.variable(model.nx)]
            opti.subject_to(opti.bounded(model.x_min, X[-1], model.x_max))
            U += [opti.variable(model.nu)]

        opti.subject_to(X[0] == x_init)


        Q = 1e2 * np.eye(3)
        R = 5e-3 * np.eye(self.model.nu)
        ee_ref = model.ee_ref
        for k in range(n_steps+1):

            ee_pos = model.ee_fun(X[k])
            cost += (ee_pos - ee_ref).T @ Q @ (ee_pos - ee_ref)

            if k < n_steps:
                cost += U[k].T @ R @ U[k]
                # Dynamics constraint
                opti.subject_to(X[k + 1] == model.f_fun(X[k], U[k]))
                # Torque constraints
                opti.subject_to(opti.bounded(model.tau_min, model.tau_fun(X[k], U[k]), model.tau_max))
            
            # if obstacles != None:
            #     for i in obstacles:
            #         opti.subject_to(opti.bounded(i['lb'],ee_pos[i['axis']],i['ub']))

            if obstacles != None:
                if len(obstacles['walls']):
                    for i in obstacles['walls']:
                        opti.subject_to(opti.bounded(i['lb'],ee_pos[i['axis']],i['ub']))
                if len(obstacles['objects']):
                    for i in obstacles['objects']:
                        opti.subject_to(cs.dot((ee_pos-i['position']),(ee_pos-i['position']))>i['radius']**2)

        self.opti = opti
        self.X = X
        self.U = U
        self.x_init = x_init
        self.cost = cost
        #self.dist_b = dist_b
        self.additionalSetting()
        opti.minimize(cost)



    def additionalSetting(self):
        pass

    def checkCollision(self, x):
        if self.obstacles is not None and self.params.obs_flag:
            t_glob = self.model.jointToEE(x)
            for obs in self.obstacles:
                if obs['name'] == 'floor':
                    if t_glob[2] + self.params.tol_obs < obs['bounds'][0]:
                        return False
                elif obs['name'] == 'ball':
                    dist_b = np.sum((t_glob.flatten() - obs['position']) ** 2)
                    if dist_b + self.params.tol_obs < obs['bounds'][0]:
                        return False
        return True

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
            'ipopt.max_iter': 1000,
            #'ipopt.linear_solver': 'ma57',
            'ipopt.sb': 'no'
        }

        opti.solver('ipopt', opts)
        return opti

class AccBoundsOCP(NaiveOCP):
    def __init__(self, model, n_steps, ddq_max, obstacles=None):
        self.ddq_max = ddq_max
        super().__init__(model, n_steps, obstacles)

    def additionalSetting(self):
        nq = self.model.nq

        # dq_min = - self.X[-1][nq:] ** 2 / ddq_max + self.X[-1][:nq]
        # dq_max = self.X[-1][nq:] ** 2 / ddq_max + self.X[-1][:nq]
        dq_min = -cs.sqrt(2*ddq_max[:nq]*(self.X[-1][:nq]-self.model.x_min[:nq])+regularization_sqrt)
        dq_max = cs.sqrt(2*ddq_max[:nq]*(self.model.x_max[:nq]-self.X[-1][:nq])+regularization_sqrt)
        
        self.opti.set_initial(self.X[-1],x0)
        # self.opti.subject_to(cs.fmax(dq_min,cs.MX.ones(nq,1)*regularization_sqrt) <= self.X[-1][nq:])
        # self.opti.subject_to(cs.fmax(dq_max,cs.MX.ones(nq,1)*regularization_sqrt) >= self.X[-1][nq:])
        self.opti.subject_to(dq_min <= self.X[-1][nq:])
        self.opti.subject_to(dq_max >= self.X[-1][nq:])
        if self.obstacles != None:
            if len(self.obstacles['walls'])>0:
                for i in self.obstacles['walls']:
                    distance= i['pos'] - robot.ee_fun(self.X[-1])[i['axis']]
                    dx_max = cs.sqrt(2*ddx_max[i['axis']]*cs.fabs(distance))   #  [i['axis']]
                    self.opti.subject_to(((robot.jac(np.eye(4),self.X[-1][:nq])[:3,6:]@self.X[-1][nq:])[i['axis']])*cs.sign(distance) <= dx_max)
                    #dx_max = cs.sqrt(2*ddx_max[i['axis']]*cs.fabs(robot.ee_fun(self.X[-1])[i['axis']]- i['pos']))   #  [i['axis']]
                    #self.opti.subject_to(self.opti.bounded(-dx_max,    (robot.jac(np.eye(4),self.X[-1][:nq])[:3,6:]@self.X[-1][nq:]) [i['axis']],    dx_max))
            if len(self.obstacles['objects'])>0:
                for i in self.obstacles['objects']:
                    dist_vec_end = (i['position'])-robot.ee_fun(self.X[-1])
                    dx_max = cs.sqrt(2*ddx_max*cs.fabs(dist_vec_end)) 
                    self.opti.subject_to(cs.dot((robot.jac(np.eye(4),self.X[-1][:nq])[:3,6:]@self.X[-1][nq:]),dist_vec_end/cs.norm_2(dist_vec_end))<=cs.norm_2(dx_max))
                    #self.opti.subject_to(self.opti.bounded(-dx_max,robot.jac(np.eye(4),self.X[-1][:robot.nq])[:3,6:]@self.X[-1][robot.nq:], dx_max))


        self.opti.cost = 0

def check_cartesian_constraint(state, obstacles):
    if obstacles != None:
        check = True
        if len(obstacles['walls'])>0:
            for i in obstacles['walls']:
                check = check and (i['lb'] <= robot.ee_fun(state)[i['axis']] <= i['ub'])
                distance = i['pos'] -robot.ee_fun(state)[i['axis']]
                dx_max = np.sqrt(2*ddx_max[i['axis']]*np.abs(distance))
                check = check and (((robot.jac(np.eye(4),state[:robot.nq])[:3,6:]@state[robot.nq:])[i['axis']])*np.sign(distance) <= dx_max)
                    
        if len(obstacles['objects'])>0:
            for i in obstacles['objects']:
                dist_vec = np.array(i['position'])-robot.ee_fun(state)
                check = check and (np.linalg.norm(dist_vec)>i['radius'])
                dx_max = np.sqrt(2*np.multiply(ddx_max,cs.fabs(dist_vec)))  
                check = check and cs.dot((robot.jac(np.eye(4),state[:robot.nq])[:3,6:]@state[robot.nq:]),dist_vec/cs.norm_2(dist_vec))<= np.linalg.norm(dx_max)
                #check = check and np.array(- dx_max <= robot.jac(np.eye(4),state[:robot.nq])[:3,6:]@state[robot.nq:]).all() \
                    #and np.array(robot.jac(np.eye(4),state[:robot.nq])[:3,6:]@state[robot.nq:]<= dx_max).all()

        return check
    else:
        return True
    
def load(filename):
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def perturbate_init(state,perturbation):
    state[:robot.nq] = state[:robot.nq] + perturbation*np.sign(np.random.rand(robot.nq))
    return state
 
if __name__ == "__main__":
    now = datetime.datetime.now()
    saving_date = str(datetime.datetime.now())
    np.random.seed(now.microsecond*now.second+now.minute) 

    params = parser.Parameters('z1')
    robot = adam_model.AdamModel(params,n_dofs=4)

    ddq_max = np.array([0.3,3,5,7])/3
    ddx_max = np.array([0.1, 0.1, 0.1])/0.1

    regularization_sqrt = 1e-6

    ee_radius = 0.075
    walls = [
    {'axis':2, 'lb':0+ee_radius, 'ub':1e6, 'pos':0+ee_radius},
    # {'axis':2, 'lb':-1e6, 'ub':0.5, 'pos':0.5},
    # {'axis':0, 'lb':-1e6, 'ub':0.5, 'pos':0.5},
    # {'axis':0, 'lb':-0.5, 'ub':1e5, 'pos':-0.5}
    ]
    #walls = None

    # objects modeled as spheres
    objects = [
        {'position':[0.6, 0., 0.12], 'radius':0.12+ee_radius}
        # {'position':[0.,0.3,0.15],'radius':0.05},
        # {'position':[0.3,-0.35,0.25], 'radius':0.1}
    ]

    obstacles = {'walls':walls,'objects':objects}
    #obstacles = None

    n_samples=10000
    max_n_steps = 300
    x0_successes = []
    x0_failed = []

    print(f'joint bounds = {robot.x_min} , {robot.x_max} ')

    data_folder = os.path.join(os.getcwd(),'N-steps results') 
    x0_s = load(os.path.join(data_folder,'x0_failed2024-12-19 11:48:49.644658.pkl'))

    progress_bar = tqdm.tqdm(total=len(x0_s), desc='Progress')
    for k in range(len(x0_s)):
        for j in range(1):
            x0=perturbate_init(x0_s[k],1e-4*j)
            print(f'{k} : {x0_s[k]}')
            horizon = 50
            while horizon <= max_n_steps:
                ocp_form= AccBoundsOCP(robot,horizon,ddq_max,obstacles)
                ocp = ocp_form.instantiateProblem()
                ocp.set_value(ocp_form.x_init, x0)
                try:
                    sol = ocp.solve()
                    x0_successes.append([x0,horizon])
                    print(f"Returned in {horizon} steps")
                    break
                except:
                    print(robot.ee_fun(x0))
                    #ocp.debug.g_describe(0)
                    #ocp.debug.show_infeasibilities()
                    print(f"Failed in {horizon} steps")
                    if horizon >= max_n_steps:
                        x0_failed.append(x0)
                        print(x0)
                if horizon >= 100: horizon +=50
                else: horizon += 25
        progress_bar.update(1)
    progress_bar.close()

    print(len(x0_successes))
    folder = os.getcwd()+'/N-steps-failed-tested'
    if not os.path.exists(folder):
        os.makedirs(folder)

    print(f'Failures: {len(x0_failed)}/{len(x0_s)}')
    with open(folder + '/x0_solved' + saving_date + '.pkl', 'wb') as file:
        pickle.dump(x0_successes, file)

    with open(folder + '/x0_failed' + saving_date + '.pkl', 'wb') as file:
        pickle.dump(x0_failed, file)