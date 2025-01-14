import numpy as np
import casadi as cs
import adam_model
import parser
import copy
import random
import pickle
import datetime
import os
import pinocchio as pin
import time
import meshcat


def gravity(state):
    opti = cs.Opti()
    x_init = opti.parameter(robot.nx)
    cost = 0

    # Define decision variables
    X, U = [], []
    X += [opti.variable(robot.nx)]
    X += [opti.variable(robot.nx)]
    opti.subject_to(opti.bounded(robot.x_min, X[-1], robot.x_max))
    U += [opti.variable(robot.nu)]

    opti.subject_to(X[0] == x_init)


    opti.subject_to(X[1] == robot.f_fun(X[0], U[0]))
    # Torque constraints
    opti.subject_to(opti.bounded(robot.tau_min, robot.tau_fun(X[0], U[0]), robot.tau_max))
    opti.subject_to(X[1][robot.nq:] == [0,0,0])
    
    opti.minimize(cost)

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
    opti.set_value(x_init, state)
    sol = opti.solve()
    u = robot.tau_fun(sol.value(X[0]),sol.value(U[0]))
    return u

def load(filename):
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def saturate_tau(u):
    # for torque in np.nditer(u):
    #     if np.abs(torque)>robot.tau_max
    for i in range(robot.nu):
        if np.abs(u[i])>robot.tau_max[i]:
            u[i] = np.sign(u[i])*robot.tau_max[i]
    return u

def spawn_obstacles():
    box0 = meshcat.geometry.Box([2, 2, 1e-3])
    viz.viewer['world/obstacle/floor0'].set_object(box0)
    viz.viewer['world/obstacle/floor0'].set_property('color', [0, 0, 1, 0.5])
    viz.viewer['world/obstacle/floor0'].set_property('visible', True)
    T_floor = np.eye(4)
    viz.viewer['world/obstacle/floor0'].set_transform(T_floor)

    # box1 = meshcat.geometry.Box([2, 2, 1e-3])
    # viz.viewer['world/obstacle/floor1'].set_object(box1)
    # viz.viewer['world/obstacle/floor1'].set_property('color', [0, 0, 1, 0.5])
    # viz.viewer['world/obstacle/floor1'].set_property('visible', True)
    # T_floor = np.eye(4)
    # T_floor[:3,3] = np.array([0,0,0.6])
    # viz.viewer['world/obstacle/floor1'].set_transform(T_floor)


    # box2 = meshcat.geometry.Box([1e-3, 2, 2])
    # viz.viewer['world/obstacle/floor2'].set_object(box2)
    # viz.viewer['world/obstacle/floor2'].set_property('color', [0, 0, 1, 0.5])
    # viz.viewer['world/obstacle/floor2'].set_property('visible', True)
    # T_floor = np.eye(4)
    # T_floor[:3,3] = np.array([0.5,0,0])
    # viz.viewer['world/obstacle/floor2'].set_transform(T_floor)

    sphere = meshcat.geometry.Sphere(0.12)
    viz.viewer['world/obstacle/sphere'].set_object(sphere)
    viz.viewer['world/obstacle/sphere'].set_property('color',[1, 0, 0, 1])
    viz.viewer['world/obstacle/sphere'].set_property('visible', True)
    T_obs = np.eye(4)
    T_obs[:3, 3] = np.array([0.6,0.,0.12])
    viz.viewer['world/obstacle/sphere'].set_transform(T_obs)

def check_collision(obstacles,state):
    check=True
    if len(obstacles['walls'])>0:
        for i in obstacles['walls']:
            check = check and (i['lb'] <= robot.ee_fun(state)[i['axis']] <= i['ub'])
    if len(obstacles['objects'])>0:
        for i in obstacles['objects']:
            check = check and (np.linalg.norm(robot.ee_fun(state) - np.array(i['position'])) >= i['radius'])
    return check

if __name__ == '__main__':
    robot_name = 'fr3'
    dof = 6
    params = parser.Parameters(robot_name)
    robot = adam_model.AdamModel(params,n_dofs=dof)
    if robot.params.urdf_name == 'fr3':
        robot.tau_max = np.array([17,87,8.7,34.8,2.4,4.8])
        robot.tau_min = -np.array([17,87,8.7,34.8,2.4,4.8])
    nq = robot.nq

    max_steps = 500
    kd = 200
    ee_radius = 0.075


    walls = [
    {'axis':2, 'lb':0+ee_radius, 'ub':1e6, 'pos':0+ee_radius },
    # {'axis':2, 'lb':-1e6, 'ub':0.5, 'pos':0.5},
    # {'axis':0, 'lb':-1e6, 'ub':0.5, 'pos':0.5},
    # {'axis':0, 'lb':-0.5, 'ub':1e5, 'pos':-0.5}
    ]
    #walls = None

    # objects modeled as spheres
    objects = [
        {'position':[0.6,0.,0.12], 'radius':0.12+ee_radius}
        #{'position':[0.3,-0.35,0.25], 'radius':0.1}
    ]

    obstacles = {'walls':walls,'objects':objects}

    data_folder = os.path.join(os.getcwd(),'N-steps results') 
    x0_s = load(os.path.join(data_folder,'x0_failed2025-01-13 09:00:07.809522.pkl'))


    description_dir = params.ROBOTS_DIR + f'{robot_name}_description'
    rmodel, collision, visual = pin.buildModelsFromUrdf(description_dir + f'/urdf/{robot_name}.urdf',
                                                        package_dirs=params.ROOT_DIR)
    geom = [collision, visual]

    lockIDs = []
    if robot_name == 'z1':
        lockNames = ['joint5', 'joint6', 'jointGripper']
    if robot_name == 'fr3':
        lockNames = ['fr3_joint5', 'fr3_joint6', 'fr3_joint7', 'fr3_finger_joint1','fr3_finger_joint2']
        lockNames = lockNames[(dof-4):]
    for name in lockNames:
        lockIDs.append(rmodel.getJointId(name))

    rmodel_red, geom_red = pin.buildReducedModel(rmodel, geom, lockIDs, np.zeros(9 if robot_name == 'fr3' else 'z1'))

    viz = pin.visualize.MeshcatVisualizer(rmodel_red, geom_red[0], geom_red[1])
    viz.initViewer(loadModel=True, open=True)
    viz.display(x0_s[0][:nq])
    spawn_obstacles()
    
    X0 = cs.MX.sym('X0', robot.nx)
    U = cs.MX.sym('U',robot.nu)
    Q = 0
    for j in range(4):
        k1 = robot.f_fun_forw(X0, U)
        k2 = robot.f_fun_forw(X0 + robot.params.dt/2 * k1, U)
        k3 = robot.f_fun_forw(X0 + robot.params.dt/2 * k2, U)
        k4 = robot.f_fun_forw(X0 + robot.params.dt * k3, U)
        X_F=X0+robot.params.dt/6*(k1 +2*k2 +2*k3 +k4)
    F_disc_tau = cs.Function('F', [X0, U], [X_F])

    
    print(f'number of simulations = {len(x0_s)}')
    for i in range(len(x0_s)):
        x_simu = np.zeros((robot.nx,max_steps))
        x_simu[:,0] = x0_s[i]
        #x_simu[nq:,0] = np.array([0,0,0])
        for k in range(1,max_steps):
            u_der = 0*robot.gravity(np.eye(4),x_simu[:nq,k-1])[-nq:] + \
                robot.mass(np.eye(4), x_simu[:nq,k-1])[6:, 6:] @ (-kd*x_simu[nq:,k-1])  + \
                robot.bias(np.eye(4), x_simu[:nq,k-1], np.zeros(6), x_simu[nq:,k-1])[6:]
            u_der_tau = saturate_tau(u_der)
            #u_der2 = gravity(x_simu[:,k-1])
             
            x_simu[:,k] = np.squeeze(F_disc_tau(x_simu[:,k-1],u_der_tau))

            
        # _, D,_ = np.linalg.svd(robot.jac(np.eye(4),x0_s[i][:robot.nq])[:3,6:])
        # print(f'singularity indicator {np.abs((D[0]/D[-1]))}') 
        print(f'{i} : {x0_s[i]}')
        print(robot.ee_fun(x0_s[i]))
        if (render:=True):
            for j in range(x_simu.shape[1]):
                if not(check_collision(obstacles,x_simu[:,j])):
                    print('Collision')
                    viz.display(x_simu[:nq,j])
                    time.sleep(5)
                    break
                viz.display(x_simu[:nq,j])
                time.sleep((1/5))
                print(j)
                if np.linalg.norm(x_simu[:,j]-x_simu[:,j-1]) < 1e-4:
                    break

        
