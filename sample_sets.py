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

def check_obstacle_free(state,obstacles):
    check = True
    if obstacles != None:
        if len(obstacles['walls'])>0:
            for i in obstacles['walls']:
                check = check and (i['lb'] <= robot.ee_fun(state)[i['axis']] <= i['ub'])
        if len(obstacles['objects'])>0:
            for i in obstacles['objects']:
                dist_vec = np.array(i['position'])-robot.ee_fun(state)
                check = check and (np.linalg.norm(dist_vec)>=i['radius'])
        return check
    else:
        return check
    
def check_analytic_set(state, obstacles):
    check = True
    for i in range(robot.nq):
        check = check and (-np.sqrt(2*ddq_max[i]*(state[i]-robot.x_min[i])) <= state[robot.nq+i] <= np.sqrt(2*ddq_max[i]*(robot.x_max[i]-x_sampled[i])))
    if obstacles != None:
        if len(obstacles['walls'])>0:
            for i in obstacles['walls']:
                distance = i['pos'] -robot.ee_fun(state)[i['axis']]
                dx_max = np.sqrt(2*ddx_max[i['axis']]*np.abs(distance))
                check = check and (((robot.jac(np.eye(4),state[:robot.nq])[:3,6:]@state[robot.nq:])[i['axis']])*np.sign(distance) <= dx_max)
                    
        if len(obstacles['objects'])>0:
            for i in obstacles['objects']:
                dist_vec = np.array(i['position'])-robot.ee_fun(state)
                check = check and (np.linalg.norm(dist_vec)>=i['radius'])
                dx_max = np.sqrt(2*np.multiply(ddx_max,cs.fabs(dist_vec)))  
                check = check and cs.dot((robot.jac(np.eye(4),state[:robot.nq])[:3,6:]@state[robot.nq:]),dist_vec/cs.norm_2(dist_vec))<= np.linalg.norm(dx_max)
        return check
    else:
        return check
    
def check_network_set(state):
    return(robot.nn_func(state)>=0)

if __name__ == '__main__':
    params = parser.Parameters('z1')
    robot = adam_model.AdamModel(params,n_dofs=4)
    robot.params.alpha=10
    robot.setNNmodel()

    ddq_max = np.array([0.3,3,5,7])/3
    ddx_max = np.array([0.1, 0.1, 0.1])/0.3

    n_samples = 100000

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

    analytic_and_network, analytic_not_network, network_not_analytic, not_both = [],[],[],[]
    k=0

    progress_bar = tqdm.tqdm(total=n_samples, desc='Progress')
    while k<n_samples:
        x_sampled = np.array((robot.x_max-robot.x_min)*np.random.random_sample((robot.nx,)) + robot.x_min*np.ones((robot.nx,)))
        if check_obstacle_free(x_sampled,obstacles):
            x_in_analytic = check_analytic_set(x_sampled,obstacles)
            x_in_network = check_network_set(x_sampled)
            if (x_in_analytic and x_in_network):
                analytic_and_network.append(x_sampled)
            elif x_in_analytic:
                analytic_not_network.append(x_sampled)
            elif x_in_network:
                network_not_analytic.append(x_sampled)
            else:
                not_both.append(x_sampled) 
            k+=1
            progress_bar.update(1)
        else:
            continue
    progress_bar.close()
    print(f'both: {len(analytic_and_network)}, analytic: {len(analytic_not_network)}, network: {len(network_not_analytic)}, none = {len(not_both)}, ratio: {(len(analytic_and_network)+len(network_not_analytic))/(len(analytic_and_network)+len(analytic_not_network))}')
    print(ddq_max)
    print(ddx_max)
