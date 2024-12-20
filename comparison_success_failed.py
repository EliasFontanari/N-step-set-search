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

def load(filename):
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

if __name__ == '__main__':

    params = parser.Parameters('z1')
    robot = adam_model.AdamModel(params,n_dofs=4)

    data_folder = os.path.join(os.getcwd(),'N-steps results') 
    x0_failed = load(os.path.join(data_folder,'x0_failed2024-12-16 18:06:38.682140.pkl'))
    x0_solved = load(os.path.join(data_folder,'x0_solved2024-12-16 18:06:38.682140.pkl'))

    # singular_ratio_failed, singular_ratio_success = [],[]
    # for i in range(len(x0_failed)):
    #     U, D, VT = np.linalg.svd(robot.jac(np.eye(4),x0_failed[i][:robot.nq])[:3,6:])
    #     singular_ratio_failed.append(np.abs(D[0]/D[-1])) 
    # for i in range(len(x0_solved)):
    #     U, D, VT = np.linalg.svd(robot.jac(np.eye(4),x0_solved[i][0][:robot.nq])[:3,6:])
    #     singular_ratio_success.append(np.abs(D[0]/D[-1])) 
    # print(f'mean failed = {np.max(singular_ratio_failed)}, mean success = {np.max(singular_ratio_success)}')
    
    # #plt.hist(singular_ratio_failed, bins=10, color='blue', edgecolor='black',alpha=0.5)
    # plt.hist(singular_ratio_success,bins=10000, color='yellow', edgecolor='black',alpha=0.5)
    # plt.show()

    joint_closeness_min_fails = []
    joint_closeness_fails = []
    for i in range(len(x0_failed)):
        closeness = np.minimum(np.abs(x0_failed[i][:robot.nq]-robot.x_min[:robot.nq]),np.abs(robot.x_max[:robot.nq]-x0_failed[i][:robot.nq]))/(robot.x_max[:robot.nq]-robot.x_min[:robot.nq])
        joint_closeness_fails.append(np.mean(closeness))
        joint_closeness_min_fails.append(np.amin(closeness)) 

    joint_closeness_min_succ = []
    joint_closeness_succ = []
    for i in range(len(x0_solved)):
        closeness = np.minimum(np.abs(x0_solved[i][0][:robot.nq]-robot.x_min[:robot.nq]),np.abs(robot.x_max[:robot.nq]-x0_solved[i][0][:robot.nq]))/(robot.x_max[:robot.nq]-robot.x_min[:robot.nq])
        joint_closeness_succ.append(np.mean(closeness))
        joint_closeness_min_succ.append(np.amin(closeness)) 

    plt.figure()
    plt.hist(joint_closeness_min_fails,density=True, bins=50, color='blue', edgecolor='black',alpha=0.5)
    
    plt.figure()
    plt.hist(joint_closeness_min_succ,density=True,  bins=50, color='yellow', edgecolor='black',alpha=0.5)
    
    plt.show()

    plt.figure()
    plt.hist(joint_closeness_fails,density=True, bins=50, color='blue', edgecolor='black',alpha=0.5)
    
    plt.figure()
    plt.hist(joint_closeness_succ,density=True,  bins=50, color='yellow', edgecolor='black',alpha=0.5)
    
    plt.show()

    joint_closeness_x_vel_fails = []
    for i in range(len(x0_failed)):
        closeness_x_vel= np.zeros(robot.nq)
        for k in range(closeness.shape[0]):
            if np.abs(x0_failed[i][k]-robot.x_min[k]) < np.abs(robot.x_max[k]-x0_failed[i][k]):
                closeness_x_vel = (-(robot.x_max[k]-robot.x_min[k])/(x0_failed[i][k]-robot.x_min[k]))*x0_failed[i][k+robot.nq]
            else:
                closeness_x_vel = (-(robot.x_max[k]-robot.x_min[k])/(x0_failed[i][k]-robot.x_max[k]))*x0_failed[i][k+robot.nq]
        joint_closeness_x_vel_fails.append(np.amax(closeness_x_vel))

    joint_closeness_x_vel_succ = []
    for i in range(len(x0_solved)):
        closeness_x_vel= np.zeros(robot.nq)
        for k in range(closeness.shape[0]):
            if np.abs(x0_solved[i][0][k]-robot.x_min[k]) < np.abs(robot.x_max[k]-x0_solved[i][0][k]):
                closeness_x_vel = (-(robot.x_max[k]-robot.x_min[k])/(x0_solved[i][0][k]-robot.x_min[k]))*x0_solved[i][0][k+robot.nq]
            else:
                closeness_x_vel = (-(robot.x_max[k]-robot.x_min[k])/(x0_solved[i][0][k]-robot.x_max[k]))*x0_solved[i][0][k+robot.nq]
        joint_closeness_x_vel_succ.append(np.amax(closeness_x_vel))
            
    plt.figure()
    plt.hist(joint_closeness_x_vel_fails,density=True, color='blue', edgecolor='black',alpha=0.5)
    
    plt.figure()
    # joint_closeness_x_vel_succ = np.array(joint_closeness_x_vel_succ)
    # joint_closeness_x_vel_succ = joint_closeness_x_vel_succ[joint_closeness_x_vel_succ>-100]
    # print(max(joint_closeness_x_vel_succ))
    plt.hist(joint_closeness_x_vel_succ,density=True, color='yellow', edgecolor='black',alpha=0.5)
    
    plt.show()