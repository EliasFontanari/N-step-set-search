import numpy as np

state_tol = 1e-4    # tolerance used for checking state constraints
tau_tol = 1e-4      # tolerance used for checking control constraints
solver_tol = 1e-4   # tolerance used by the solver
solver_max_iter = 300
obs_flag = True


urdf_name = "z1"
from example_robot_data.robots_loader import load
robot_pin = load("z1")
frame_name = "gripperMover"
robot_urdf = robot_pin.urdf
n_dofs = 4

NN_DIR = "./nn_dir/"
GEN_DIR= "./gen_dir/"    # build directory for L4Casadi
eps = 1e-6              # used to avoid division by zero in the NN function
alpha = 6               # safety factor used in the NN constraint

N = 45
dt = 5e-3

# Cost
ee_des = np.array([0.6, 0.28, 0.078])
Q = 1e2 * np.eye(3)
R = 5e-3 * np.eye(n_dofs)

ee_radius = 0.075
obstacles = []

obs = dict()
obs['name'] = 'floor'
obs['type'] = 'box'
obs['dimensions'] = [2, 2, 1e-3]
obs['color'] = [0, 0, 1, 1]
obs['position'] = np.array([0., 0., 0.])
obs['transform'] = np.eye(4)
obs['bounds'] = np.array([ee_radius, 1e6])      # lb , ub
obstacles.append(obs)

obs = dict()
obs['name'] = 'ball'
obs['type'] = 'sphere'
obs['radius'] = 0.12
obs['color'] = [0, 1, 1, 1]
obs['position'] = np.array([0.6, 0., 0.12])
T_ball = np.eye(4)
T_ball[:3, 3] = obs['position']
obs['transform'] = T_ball
obs['bounds'] = np.array([(ee_radius + obs['radius']) ** 2, 1e6])     
obstacles.append(obs)
