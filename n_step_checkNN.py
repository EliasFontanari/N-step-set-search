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

class NN_N_step_OCP:
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

        for k in range(n_steps+1):

            ee_pos = model.ee_fun(X[k])

            if k < n_steps:
                # Dynamics constraint
                opti.subject_to(X[k + 1] == model.f_fun(X[k], U[k]))
                # Torque constraints
                opti.subject_to(opti.bounded(model.tau_min, model.tau_fun(X[k], U[k]), model.tau_max))

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
        self.additionalSetting()
        opti.minimize(cost)



    def additionalSetting(self):
        self.opti.subject_to(robot.nn_func(self.X[-1])>=0)
        self.opti.cost = 0

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
    
def check_cartesian_constraint(state, obstacles):
    if obstacles != None:
        check = True
        if len(obstacles['walls'])>0:
            for i in obstacles['walls']:
                check = check and (i['lb'] <= robot.ee_fun(state)[i['axis']] <= i['ub'])
        if len(obstacles['objects'])>0:
            for i in obstacles['objects']:
                dist_vec = np.array(i['position'])-robot.ee_fun(state)
                check = check and (np.linalg.norm(dist_vec)>i['radius'])

        return check
    else:
        return True


def sample_state(obstacles=None):
    while True:
        x_sampled = np.array((robot.x_max-robot.x_min)*np.random.random_sample((robot.nx,)) + robot.x_min*np.ones((robot.nx,)))
        vel_direction = x_sampled[robot.nq:]/np.linalg.norm(x_sampled[robot.nq:])
        vel_max_norm= robot.nn_out(x_sampled)*(100-robot.params.alpha)/100
        vel = vel_direction*vel_max_norm
        x_sampled[robot.nq:] = np.squeeze(vel)
        # x_sampled[robot.nq:] = np.where(vel<robot.x_min[robot.nq:],robot.x_min[robot.nq:],vel)
        # x_sampled[robot.nq:] = np.where(vel>robot.x_max[robot.nq:],robot.x_max[robot.nq:],vel)
        # for i in range(robot.nq):
        #     # min/max velocity bound
        #     sign = random.choice([-1, 1])
        #     if sign > 0:
        #         x_sampled[robot.nq+i] = np.sqrt(2*ddq_max[i]*(robot.x_max[i]-x_sampled[i]))
        #         if x_sampled[robot.nq+i] > robot.x_max[robot.nq+i]:
        #             x_sampled[robot.nq+i] = robot.x_max[robot.nq+i]
        #     else:
        #         x_sampled[robot.nq+i] = -np.sqrt(2*ddq_max[i]*(x_sampled[i]-robot.x_min[i]))
        #         if x_sampled[robot.nq+i] < robot.x_min[robot.nq+i]:
        #             x_sampled[robot.nq+i] = robot.x_min[robot.nq+i]
        if obstacles != None:
            if check_cartesian_constraint(x_sampled,obstacles) and (robot.x_min <= x_sampled).all() and (x_sampled<= robot.x_max).all():
                return x_sampled
        else:
            return x_sampled

# def sample_state(obstacles=None):
#     # bound_l = robot.x_min/1.1
#     # bound_h = robot.x_max/1.1
#     while True:
#         x_sampled = np.array((robot.x_max-robot.x_min)*np.random.random_sample((robot.nx,)) + robot.x_min*np.ones((robot.nx,)))
        
#         if obstacles != None:
#             if (robot.x_min[:robot.nq] <= -x_sampled[robot.nq:] ** 2 / ddq_max + x_sampled[:robot.nq]).all() and \
#                 (x_sampled[robot.nq:] ** 2 / ddq_max + x_sampled[:robot.nq] <= robot.x_max[:robot.nq]).all(): 
#                 if check_cartesian_constraint(x_sampled,obstacles):
#                     return x_sampled
#         else:
#             return x_sampled

# 0.03 quantile ddq 25, 31, 37
# 0.03 quantile ddx_max 0.4, 0.65, 0.03
 
if __name__ == "__main__":
    now = datetime.datetime.now()
    np.random.seed(now.microsecond*now.second+now.minute) 

    params = parser.Parameters('z1')
    robot = adam_model.AdamModel(params,n_dofs=4)
    robot.setNNmodel()

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

    n_samples=10000
    max_n_steps = 40
    x0_successes = []
    x0_failed = []

    print(f'joint bounds = {robot.x_min} , {robot.x_max} ')

    
    progress_bar = tqdm.tqdm(total=n_samples, desc='Progress')
    for k in range(n_samples):
        x0=sample_state(obstacles)
        
        horizon = 1
        while horizon <= max_n_steps:
            ocp_form= NN_N_step_OCP(robot,horizon,obstacles)
            ocp = ocp_form.instantiateProblem()
            ocp.set_value(ocp_form.x_init, x0)
            try:
                sol = ocp.solve()
                print(f"Returned in {horizon} steps")
                print(sol.value(ocp_form.X[-1]))
                x0_successes.append([copy.copy(x0),horizon])
                break
            except:
                print(f"Failed in {horizon} steps")
                if horizon >= max_n_steps:
                    x0_failed.append(copy.copy(x0))
                    print(x0)
                    print(robot.ee_fun(x0))
            if horizon >= 10: horizon +=10
            else: horizon +=3
            print(f'number of failures: {len(x0_failed)}')
        progress_bar.update(1)
    progress_bar.close()
    print(len(x0_successes))
    folder = os.getcwd()+'/N-NN-steps-results'
    if not os.path.exists(folder):
        os.makedirs(folder)

    saving_date = str(datetime.datetime.now())
    print(f'Failures: {len(x0_failed)}/{n_samples}')
    with open(folder + '/x0_solved' + saving_date + '.pkl', 'wb') as file:
        pickle.dump(x0_successes, file)
    with open(folder + '/x0_failed' + saving_date + '.pkl', 'wb') as file:
        pickle.dump(x0_failed, file)