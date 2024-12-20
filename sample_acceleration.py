import numpy as np
import casadi as cs
import adam_model
import parser
import copy
import random
import tqdm
import pickle 
import datetime

class MinAccProblem:
    """ Define OCP problem and solver (IpOpt) """
    def __init__(self, model, joint):   # joint 0<= joint <= nq : joint on which maximize acceleration, while keeping zero accleration to the other
        self.params = model.params
        self.model = model
        self.nq = model.nq
        self.joint = joint

        opti = cs.Opti()
        x_init = opti.parameter(model.nx)

        # Define decision variables
        X, U = [], []
        X += [opti.variable(model.nx)]  # x_0
        X += [opti.variable(model.nx)]  # x_next
        opti.subject_to(opti.bounded(model.x_min, X[-1], model.x_max))
        U += [opti.variable(model.nu)]  # acceleration to maximize 

        opti.subject_to(X[0] == x_init)
        
        self.cost = U[-1][joint]
        # Dynamics constraint
        opti.subject_to(X[1] == model.f_fun(X[0], U[0]))
        # Torque constraint
        opti.subject_to(opti.bounded(model.tau_min, model.tau_fun(X[0], U[0]), model.tau_max))
        for i in range(self.model.nq):
            if i!= joint:
                opti.subject_to(U[-1][i]==0)
        self.X = X
        self.U = U
        self.x_init = x_init
        self.additionalSetting()
        opti.minimize(self.cost)
        self.opti = opti
        

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
            'ipopt.max_iter': self.params.nlp_max_iter,
            #'ipopt.linear_solver': 'ma57',
            'ipopt.sb': 'yes'
        }

        opti.solver('ipopt', opts)
        return opti

class MaxAccProblem(MinAccProblem):
    """ Define OCP problem and solver (IpOpt) """
    def __init__(self, model, joint):   # joint 0<= joint <= nq : joint on which maximize acceleration, while keeping zero accleration to the other
        self.params = model.params
        self.model = model
        self.nq = model.nq
        self.joint = joint

        opti = cs.Opti()
        x_init = opti.parameter(model.nx)

        # Define decision variables
        X, U = [], []
        X += [opti.variable(model.nx)]  # x_0
        X += [opti.variable(model.nx)]  # x_next
        opti.subject_to(opti.bounded(model.x_min, X[-1], model.x_max))
        U += [opti.variable(model.nu)]  # acceleration to maximize 

        opti.subject_to(X[0] == x_init)
        
        self.cost = -U[-1][joint]
        # Dynamics constraint
        opti.subject_to(X[1] == model.f_fun(X[0], U[0]))
        # Torque constraint
        opti.subject_to(opti.bounded(model.tau_min, model.tau_fun(X[0], U[0]), model.tau_max))
        for i in range(self.model.nq):
            if i!= joint:
                opti.subject_to(U[-1][i]==0)
        self.X = X
        self.U = U
        self.x_init = x_init
        self.additionalSetting()
        opti.minimize(self.cost)
        self.opti = opti
    
    def additionalSetting(self):
        self.cost = - self.U[-1][self.joint]
        
if __name__ == "__main__":
    now = datetime.datetime.now()

    params = parser.Parameters('z1')
    robot = adam_model.AdamModel(params,n_dofs=4)


    n_samples = 50000 # samples for each joint
    acc_max = [[] for _ in range(robot.nq)]
    acc_min = [[] for _ in range(robot.nq)]
    
    acc_min_x = [[] for _ in range(robot.nq)]
    acc_max_x = [[] for _ in range(robot.nq)]

    progress_bar = tqdm.tqdm(total=n_samples*robot.nq*2, desc='Sampling started')
    for k in range(robot.nq):
        i=0
        while i < n_samples:
            if i == 0:
                print('min acc')
            x0 = np.array((robot.x_max-robot.x_min)*np.random.random_sample((robot.nx,)) + robot.x_min*np.ones((robot.nx,)))

            min_problem = MinAccProblem(robot,k)
            min_solver = min_problem.instantiateProblem()
            min_solver.set_value(min_problem.x_init,x0)
            try:
                sol = min_solver.solve()
                acc_min[k].append(sol.value(min_problem.U[-1][k])) 
                #print(sol.value(min_problem.U[-1][k]))
                ddx_min = np.array(robot.jac(np.eye(4),sol.value(min_problem.X[0][:robot.nq]))[:3,6:]@sol.value(min_problem.U[-1])) + \
                          robot.jac_dot(np.eye(4),sol.value(min_problem.X[0][:robot.nq]),np.zeros(6),sol.value(min_problem.X[0][robot.nq:]))[:3,6:]@sol.value(min_problem.X[0][robot.nq:])
                acc_min_x[k].append(copy.copy(ddx_min))
                progress_bar.update(1)
                i+=1
            except:
                print('failed')
                
        i=0
        while i < n_samples:
            if i == 0:
                print('max acc')
            x0 = np.array((robot.x_max-robot.x_min)*np.random.random_sample((robot.nx,)) + robot.x_min*np.ones((robot.nx,)))

            max_problem = MaxAccProblem(robot,k)
            max_solver = max_problem.instantiateProblem()
            max_solver.set_value(max_problem.x_init,x0)
            try:
                sol = max_solver.solve()
                acc_max[k].append(sol.value(max_problem.U[-1][k])) 
                #print(sol.value(max_problem.U[-1][k]))
                ddx_max = np.array(robot.jac(np.eye(4),sol.value(max_problem.X[0][:robot.nq]))[:3,6:]@sol.value(max_problem.U[-1])) + \
                          robot.jac_dot(np.eye(4),sol.value(max_problem.X[0][:robot.nq]),np.zeros(6),sol.value(max_problem.X[0][robot.nq:]))[:3,6:]@sol.value(max_problem.X[0][robot.nq:])
                acc_max_x[k].append(copy.copy(ddx_max))
                progress_bar.update(1)
                i+=1
            except:
                print('failed')
                
    saving_date = str(datetime.datetime.now())
    with open(saving_date+'min.pkl', 'wb') as file:
        pickle.dump(acc_min, file)
    with open(saving_date+'max.pkl', 'wb') as file:
        pickle.dump(acc_max, file)
    with open(saving_date+'ddx_min.pkl', 'wb') as file:
        pickle.dump(acc_min_x, file)
    with open(saving_date+'ddx_max.pkl', 'wb') as file:
        pickle.dump(acc_max_x, file)
    progress_bar.close()