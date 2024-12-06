import numpy as np
from neural_network import NeuralNetwork
from adam_model import AdamModel
import conf_z1 as conf
import casadi as cs
import matplotlib.pyplot as plt
from termcolor import colored
from numpy.random import uniform
import pickle
import orc.utils.plot_utils as plut

'''
    Look for initial conditions where we can solve the OCP problem. 
    To avoid using the NN, I use a longer horizon with a terminal constraint of
    zero velocity.
'''

def print_solver_stats(stats):
    if(stats["success"]):
        text_color = "green"
    else:
        text_color = "red"
    print("\t", colored(stats["return_status"], text_color), "N. iters", stats["iter_count"])

if __name__=="__main__":
    DO_PLOTS = 0
    N_PROBLEMS = 100
    conf.alpha = 0  # remove safety factor

    model = AdamModel(conf, conf.n_dofs)
    nq = model.nq

    N = conf.N
    N2 = conf.N
    ee_des = conf.ee_des

    # standard formulation with NN as terminal constraint
    model.define_problem(conf)
    opti_nn = model.instantiate_problem(conf)

    # new formulation w/o NN but with augmented horizon
    opti = cs.Opti()
    param_x_init = opti.parameter(model.nx)     # initial state
    cost = 0

    # create all the decision variables
    X, U = [], []
    X += [opti.variable(model.nx)]
    for k in range(N+N2): 
        X += [opti.variable(model.nx)]
        opti.subject_to( opti.bounded(model.x_min, X[-1], model.x_max) )
    for k in range(N+N2): 
        U += [opti.variable(model.nq)]

    opti.subject_to(X[0] == param_x_init)
    dist_b = []
    for k in range(N+N2+1):     
        if(k<N+1):
            ee_pos = model.ee_fun(X[k])
            cost +=  ((ee_pos - ee_des).T @ conf.Q @ (ee_pos - ee_des))

        if(k<N):
            cost += U[k].T @ conf.R @ U[k]

        if(k<N+N2):
            opti.subject_to(X[k+1] == model.f_fun(X[k], U[k]))
            opti.subject_to( opti.bounded(model.tau_min, model.tau_fun(X[k], U[k]), model.tau_max))
    
        if conf.obstacles is not None and conf.obs_flag:
            # Collision avoidance with obstacles
            for obs in conf.obstacles:
                t_glob = model.ee_fun(X[k])
                if obs['name'] == 'floor':
                    lb = obs['bounds'][0]
                    ub = obs['bounds'][1]
                    opti.subject_to( opti.bounded(lb, t_glob[2], ub))
                elif obs['name'] == 'ball':
                    lb = obs['bounds'][0]
                    ub = obs['bounds'][1]
                    dist_b.append((t_glob - obs['position']).T @ (t_glob - obs['position']))
                    opti.subject_to( opti.bounded(lb, dist_b[k], ub))

    opti.subject_to( X[-1][nq:] == 0.0)
    opti.minimize(cost)

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

    problem_counter = 0         # total number of problems
    # failed_problem_counter = 0  # total number of failed problems

    # problems solved directly with NN constraint
    prob_solved_w_nn = 0 
    # problems failed with NN constraint and also with augmented horizon
    prob_failed_w_nn_and_w_aug_hor = 0
    # problems failed with NN constraint, solved with aug horizon, but failed again with NN constraint and warm start
    prob_failed_w_nn_w_warm_start = 0
    # problems failed with NN constraint, solved with aug horizon, and solved with NN constraint and warm start
    prob_solved_w_nn_w_warm_start = 0

    def print_stats():
        print("\n\t**** STATISTICS ****")
        print("Problems solved directly with NN constraint", prob_solved_w_nn)
        print("Problems failed with NN constraint and also with augmented horizon", 
            prob_failed_w_nn_and_w_aug_hor)
        print("Problems failed with NN constraint, solved with aug horizon, but failed again with NN constraint and warm start",
            prob_failed_w_nn_w_warm_start)
        print("Problems failed with NN constraint, solved with aug horizon, and solved with NN constraint and warm start",
            prob_solved_w_nn_w_warm_start)
        print("\n")
    
    x_guess = np.zeros((N_PROBLEMS, N+1, model.nx))  # 100 x 46 x 8
    u_guess = np.zeros((N_PROBLEMS, N, nq))          # 100 x 45 x 4
    while(problem_counter < N_PROBLEMS):
        x0, nn_0 = model.sample_feasible_state(conf)
        print("Problem", problem_counter, "On x[0], NN constr %.3f"%nn_0) #, "floor dist %.3f"%floor_dist, "ball dist %.3f"%ball_dist)

        opti_nn.set_value(model.param_x_init, x0)
        try:
            sol = opti_nn.solve()
        except:
            sol = opti_nn.debug
        print_solver_stats(sol.stats())

        if(sol.stats()["success"]):
            prob_solved_w_nn += 1
        else:
            print("\tSolve problem with augmented horizon.")
            X_nn = np.array([sol.value(model.X[k]) for k in range(N+1)]) # 46 x 8
            U_nn = np.array([sol.value(model.U[k]) for k in range(N)])   # 45 x 4
            opti.set_value(param_x_init, x0)
            try:
                sol = opti.solve()
            except:
                sol = opti.debug
            print_solver_stats(sol.stats())

            if(sol.stats()["success"]):
                nn_val = model.nn_func(sol.value(X[N]))
                print(colored("\tNN constr(X[N]) "+str(nn_val), "yellow" if nn_val<0 else "white"))

                X_aug = np.array([sol.value(X[k]) for k in range(N+1)]) # 46 x 8
                U_aug = np.array([sol.value(U[k]) for k in range(N)])   # 45 x 4
                print("\tDifference state trajs X_nn-X_aug:", np.linalg.norm(X_nn-X_aug))

                print("\tSolve again problem with NN constraint with initial guess")
                for k in range(N):
                    opti_nn.set_initial(model.X[k], sol.value(X[k]))
                    opti_nn.set_initial(model.U[k], sol.value(U[k]))
                opti_nn.set_initial(model.X[N], sol.value(X[N]))
                try:
                    sol = opti_nn.solve()
                except:
                    sol = opti_nn.debug
                print_solver_stats(sol.stats())

                if(sol.stats()["success"]):
                    prob_solved_w_nn_w_warm_start += 1
                else:
                    prob_failed_w_nn_w_warm_start += 1

                X_nn2 = np.array([sol.value(model.X[k]) for k in range(N+1)]) # 46 x 8
                U_nn2 = np.array([sol.value(model.U[k]) for k in range(N)])   # 45 x 4
                print("\tDifference state trajs X_nn2-X_aug:", np.linalg.norm(X_nn2-X_aug))
                print("\tDifference state trajs X_nn-X_nn2:", np.linalg.norm(X_nn-X_nn2))
            else:
                prob_failed_w_nn_and_w_aug_hor += 1


        # if(sol.stats()["success"]):
            # x_guess[problem_counter, :, :] = np.array([sol.value(X[k]) for k in range(N+1)]) # 100 x 46 x 8
            # u_guess[problem_counter, :, :] = np.array([sol.value(U[k]) for k in range(N)])   # 100 x 45 x 4
        problem_counter += 1
        # else:
            # failed_problem_counter += 1
        if(problem_counter%10==0):
            print_stats()
    
    print("Number of solved problems:  ", problem_counter)
    # print("Number of unsolved problems:", failed_problem_counter)
    print_stats()
    

    
    # with open('initial_guess.pkl', 'wb') as f:
        # pickle.dump({'xg': x_guess, 'ug': u_guess}, f)