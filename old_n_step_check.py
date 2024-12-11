import numpy as np
import casadi as cs
import adam_model
import parser
import copy
import random



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
        dist_b = []
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
            # Floor
            #opti.subject_to(opti.bounded(0,ee_pos[2],1e6)) 
            # Ceil
            # opti.subject_to(opti.bounded(-1e-6,ee_pos[2],0.6))
            # Wall
            # #opti.subject_to(opti.bounded(-1e-6,ee_pos[0],0.5)) 

        opti.minimize(cost)
        self.opti = opti
        self.X = X
        self.U = U
        self.x_init = x_init
        self.cost = cost
        self.dist_b = dist_b
        self.additionalSetting()

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
            'ipopt.max_iter': self.params.nlp_max_iter,
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
        #ddq_max = np.ones(self.model.nv) * 10.
        
        dqq_min = -self.X[-1][nq:] ** 2 / ddq_max + self.X[-1][:nq]
        dqq_max = self.X[-1][nq:] ** 2 / ddq_max + self.X[-1][:nq]
        self.opti.subject_to(dqq_min >= self.model.x_min[:nq])        
        self.opti.subject_to(dqq_max <= self.model.x_max[:nq])

        # dq_min = -cs.sqrt(2*ddq_max[:nq]*(self.X[-1][:nq]-self.model.x_min[:nq])+0*1e-4)
        # dq_max = cs.sqrt(2*ddq_max[:nq]*(self.model.x_max[:nq]-self.X[-1][:nq])+0*1e-4)
        # #dq_min = 2*ddq_max[:nq]*(self.X[-1][:nq]-self.model.x_min[:nq])
        
        # self.opti.set_initial(self.X[-1],x0)
        # #self.opti.subject_to(cs.if_else(self.X[-1][nq:] < np.zeros(robot.nq) , dq_min <= self.X[-1][nq:]**2, cs.fabs(self.X[-1][nq:]) <  np.ones(robot.nq)*1e6))
        # self.opti.subject_to(dq_max >= self.X[-1][nq:])
        # self.opti.subject_to(dq_min <= self.X[-1][nq:])



        # cartesian velocity constraint
        # floor
        # dx_max1 = cs.sqrt(2*ddx_max*self.model.ee_fun(self.X[-1])[2]) 
        # self.opti.subject_to(self.opti.bounded(-dx_max1,         robot.jac(np.eye(4),self.X[-1][:nq])[:3,6:]@self.X[-1][nq:],            dx_max1))
        # # # ceil
        # dx_max2 = cs.sqrt(2*ddx_max*cs.fabs(self.model.ee_fun(self.X[-1])[2]-0.6)) 
        # self.opti.subject_to(self.opti.bounded(-dx_max2,         robot.jac(np.eye(4),self.X[-1][:nq])[:3,6:]@self.X[-1][nq:],            dx_max2))
        # wall
        # dx_max3 = cs.sqrt(2*ddx_max*cs.fabs(self.model.ee_fun(self.X[-1])[2]-0.5)) 
        # self.opti.subject_to(self.opti.bounded(-dx_max3,         robot.jac(np.eye(4),self.X[-1][:nq])[:3,6:]@self.X[-1][nq:],            dx_max3))
        self.opti.cost = 0

def check_cartesian_constraint(state, obstacles):
    check = True
    for i in obstacles:
        check = check and (i['lb'] <= robot.ee_fun(state)[i['axis']] <= i['ub'])
    for i in obstacles:
        dx_max = np.sqrt(2*ddx_max[i['axis']]*np.abs(robot.ee_fun(state)[i['axis']]- i['pos']))  
        check = check and (-dx_max <= (robot.jac(np.eye(4),state[:robot.nq])[:3,6:]@state[robot.nq:])[i['axis']] <= dx_max)
    return check


def sample_state():
    # bound_l = robot.x_min/1.1
    # bound_h = robot.x_max/1.1
    # while True:
    #     check = True
    #     x_sampled = np.array((robot.x_max-robot.x_min)*np.random.random_sample((robot.nx,)) + robot.x_min*np.ones((robot.nx,)))
    #     #x_sampled = np.array((bound_h-bound_l)*np.random.random_sample((robot.nx,)) + bound_l*np.ones((robot.nx,)))
    #     for i in range(robot.nq):        
    #         check = check and x_sampled[robot.nq+i]/ddq_max[i] + x_sampled[i] <= robot.x_max[i]
    #     for i in range(robot.nq):        
    #         check = check and x_sampled[robot.nq+i]/ddq_max[i] - x_sampled[i] <= -robot.x_min[i]
    #     if check:
    #         return x_sampled
        

    while True:
        x_sampled = np.array((robot.x_max-robot.x_min)*np.random.random_sample((robot.nx,)) + robot.x_min*np.ones((robot.nx,)))
        #x_sampled = np.array((bound_h-bound_l)*np.random.random_sample((robot.nx,)) + bound_l*np.ones((robot.nx,)))
        
        x_sampled[robot.nq:] = np.zeros(robot.nq)
        for i in range(robot.nq):
            # min/max velocity bound
            sign = random.choice([-1, 1])
            if sign > 0:
                x_sampled[robot.nq+i] = np.sqrt(2*ddq_max[i]*(robot.x_max[i]-x_sampled[i]))
                if x_sampled[robot.nq+i] > robot.x_max[robot.nq+i]:
                    x_sampled[robot.nq+i] = robot.x_max[robot.nq+i]
            else:
                x_sampled[robot.nq+i] = -np.sqrt(2*ddq_max[i]*(x_sampled[i]-robot.x_min[i]))
                if x_sampled[robot.nq+i] < robot.x_min[robot.nq+i]:
                    x_sampled[robot.nq+i] = robot.x_min[robot.nq+i]
        # if 1*robot.ee_fun(x_sampled)[2]>=0 and 0*robot.ee_fun(x_sampled)[2]<=0.6 and 0*robot.ee_fun(x_sampled)[0]<=0.5:
        #     return x_sampled
        if check_cartesian_constraint(x_sampled,obstacles):
            return x_sampled

# 0.03 quantile ddq 25, 31, 37
# 0.03 quantile ddx_max 0.4, 0.65, 0.03
 
if __name__ == "__main__":

    params = parser.Parameters('z1')
    robot = adam_model.AdamModel(params,n_dofs=3)

    ddq_max = np.array([25,31,37])
    #ddq_max = np.ones(robot.nv) * 10.
    ddx_max = np.array([0.4,0.65,0.03])

    obstacles = [
    {'axis':2, 'lb':0, 'ub':1e6, 'pos':0 },
    {'axis':2, 'lb':-1e6, 'ub':0.7, 'pos':0.7},
    {'axis':0, 'lb':-1e6, 'ub':0.6, 'pos':0.6 }
    ]
    # obstacles = None

    n_samples=100
    max_n_steps = 20
    x0_successes = []
    x0_failures = []

    print(f'joint bounds = {robot.x_min} , {robot.x_max} ')

    for k in range(n_samples):
        x0=sample_state()
        for horizon in range(1,max_n_steps+1):
            ocp_form= AccBoundsOCP(robot,horizon,ddq_max)
            ocp = ocp_form.instantiateProblem()
            ocp.set_value(ocp_form.x_init, x0)
            try:
                sol = ocp.solve()
                print(f"Returned in {horizon} steps")
                x0_successes.append([copy.copy(x0),horizon])
                break
            except:
                print(f"Failed in {horizon} steps")
                if horizon == max_n_steps:
                    x0_failures.append(copy.copy(x0))
                    print(x0)
    print(f'Failures: {len(x0_failures)}/{n_samples}')

