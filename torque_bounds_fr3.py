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
import matplotlib.pyplot as plt

class MaxVelOCP:
    """ Define OCP problem and solver (IpOpt) """
    def __init__(self, model, n_steps):
        self.params = model.params
        self.model = model
        self.nq = model.nq

        N = self.params.N
        Q=1e-2
        opti = cs.Opti()
        x_init = opti.parameter(model.nx)
        vel_dir = opti.parameter(model.nq)
        

        # Define decision variables
        X, U = [], []
        X += [opti.variable(model.nx)]
        opti.subject_to(opti.bounded(model.x_min, X[-1], model.x_max))
        for k in range(n_steps):
            X += [opti.variable(model.nx)]
            opti.subject_to(opti.bounded(model.x_min, X[-1], model.x_max))
            U += [opti.variable(model.nu)]

        opti.subject_to(X[0][:self.nq] == x_init[:self.nq])
        opti.subject_to(((cs.MX.eye(self.nq)-(vel_dir@vel_dir.T))@X[0][self.nq:])==cs.MX.zeros(self.nq,1))

        cost = -cs.dot(vel_dir,X[0][self.nq:])

        for k in range(n_steps):
            # Dynamics constraint
            opti.subject_to(X[k + 1] == model.f_fun(X[k], U[k]))
            cost+= Q*(X[k][self.nq:]**2)
            # Torque constraints
            opti.subject_to(opti.bounded(model.tau_min, model.tau_fun(X[k], U[k]), model.tau_max))
        #opti.subject_to(X[-1]==X[-2])
        opti.subject_to(X[-1][robot.nq:]==cs.MX.zeros(self.nq,1))

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
            'ipopt.max_iter': 1000,
            #'ipopt.linear_solver': 'ma57',
            'ipopt.sb': 'no'
        }

        opti.solver('ipopt', opts)
        return opti

if __name__ == "__main__":

    params = parser.Parameters('fr3')
    not_locked_joint =5
    robot = adam_model.AdamModel(params,n_dofs=1, not_locked=not_locked_joint)
    horizon_length = 40
    samples = 80
    samples_i = int(samples/10)
    samples_f = samples_i
    results_angle = []
    results_vel = []
    plot = False

    print(robot.x_min)
    #print(f'inertia {robot.mass(np.eye(4), np.zeros(7))[6:, 6:]}')
    print(f'ee_pos {robot.ee_fun(np.zeros(1*2))}')

    divider = 2.5
    robot.tau_max = robot.tau_max/divider
    robot.tau_min = robot.tau_min/divider

    x0_s_i = np.linspace(robot.x_min[0],robot.x_min[0]+0.1,samples_i)
    x0_s = np.linspace(robot.x_min[0],robot.x_max[0],samples)
    x0_s_f = np.linspace(robot.x_max[0]-0.1,robot.x_max[0],samples_f)
    x0_s=np.hstack((x0_s_i,x0_s,x0_s_f))
    progress_bar = tqdm.tqdm(total=x0_s.shape[0], desc='Sampling started')
    for i in range(x0_s.shape[0]):
        x0 = np.array((robot.x_max-robot.x_min)*np.random.random_sample((robot.nx,)) + robot.x_min*np.ones((robot.nx,)))
        x0[:robot.nq] = x0_s[i]
        vel_direction = x0[robot.nq:]/np.linalg.norm(x0[robot.nq:])

        ocp_form= MaxVelOCP(robot,horizon_length)
        ocp = ocp_form.instantiateProblem()
        ocp.set_value(ocp_form.x_init, x0)
        ocp.set_value(ocp_form.vel_dir, vel_direction)
        try:
            sol = ocp.solve()
            results_angle.append(x0[:robot.nq])
            results_vel.append(sol.value(ocp_form.X[0][robot.nq:]))
            print(sol.value(ocp_form.X[0][robot.nq:]))

            if plot:
                controls=[]
                for i in range(horizon_length):
                    controls.append(np.array(robot.tau_fun(sol.value(ocp_form.X[i]), sol.value(ocp_form.U[i])))[0][0])
                plt.figure(f'joint{not_locked_joint}, u_max {robot.tau_max}')
                plt.plot(controls,color='blue')
                plt.axhline(y=robot.tau_max, color='red', linestyle='--', label='Dashed Line')
                plt.axhline(y=robot.tau_min, color='red', linestyle='--', label='Dashed Line')
                vels=[]
                pos=[]
                for i in range(horizon_length+1):
                    vels.append(sol.value(ocp_form.X[i][robot.nq]))
                    pos.append(sol.value(ocp_form.X[i][0]))
                plt.figure(f'joint{not_locked_joint}, velocity')
                plt.plot(vels,color='blue')

                plt.figure(f'position')
                plt.plot(pos,color='red')
                plt.hlines([robot.x_min[0],robot.x_max[0]],xmin=0,xmax=len(pos))
                plt.show()
        except:
            print('Failed')
        
        ocp_form= MaxVelOCP(robot,horizon_length)
        ocp = ocp_form.instantiateProblem()
        ocp.set_value(ocp_form.x_init, x0)
        ocp.set_value(ocp_form.vel_dir, -vel_direction)
        try:
            sol = ocp.solve()
            results_angle.append(x0[:robot.nq])
            results_vel.append(sol.value(ocp_form.X[0][robot.nq:]))
            print(sol.value(ocp_form.X[0][robot.nq:]))
            if plot:
                controls=[]
                for i in range(horizon_length):
                    controls.append(np.array(robot.tau_fun(sol.value(ocp_form.X[i]), sol.value(ocp_form.U[i])))[0][0])
                plt.figure(f'joint{not_locked_joint}, u_max {robot.tau_max}')
                plt.plot(controls,color='blue')
                plt.axhline(y=robot.tau_max, color='red', linestyle='--', label='Dashed Line')
                plt.axhline(y=robot.tau_min, color='red', linestyle='--', label='Dashed Line')
                vels=[]
                pos=[]
                for i in range(horizon_length+1):
                    vels.append(sol.value(ocp_form.X[i][robot.nq]))
                    pos.append(sol.value(ocp_form.X[i][0]))
                plt.figure(f'joint{not_locked_joint}, velocity')
                plt.plot(vels,color='blue')

                plt.figure(f'position')
                plt.plot(pos,color='red')
                plt.hlines([robot.x_min[0],robot.x_max[0]],xmin=0,xmax=len(pos))
                plt.show()
        except:
            print('Failed')
        progress_bar.update(1)
    progress_bar.close()

    plt.figure(f'joint{not_locked_joint}, u_max {robot.tau_max}')
    plt.title(f'joint{not_locked_joint}, u_max {robot.tau_max}')
    plt.scatter(results_angle, results_vel,color='blue',marker='o')
    plt.hlines([robot.x_max[1], robot.x_min[1]], robot.x_min[0], robot.x_max[0], colors='red')
    plt.vlines([robot.x_max[0], robot.x_min[0]], robot.x_min[1], robot.x_max[1], colors='red')
    plt.show()



