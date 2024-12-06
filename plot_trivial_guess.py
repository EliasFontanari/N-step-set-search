import numpy as np
import matplotlib.pyplot as plt
import pickle
from parser import Parameters
from adam_model import AdamModel



def compute_acc_bounds(j):
    q_lin = np.linspace(model.x_min[j], model.x_max[j], 100)
    dq_max = np.zeros(100)
    dq_min = np.zeros(100)
    for k in range(100):
        dq_max[k] = np.sqrt(2 * ddq * (model.x_max[j] - q_lin[k]))
        dq_min[k] = -np.sqrt(2 * ddq * (q_lin[k] - model.x_min[j])) 
    return q_lin, dq_max, dq_min
    


params = Parameters('z1', rti=False)
model = AdamModel(params, n_dofs=4)
data = pickle.load(open(f'{params.DATA_DIR}z1_trivial_guess.pkl', 'rb'))
x_guess = data['xg']    
u_guess = data['ug']

nq = model.nq
ddq = 10.
# for q in q_lin:
#     print(q)
#     print('\t', np.sqrt(2 * ddq * (model.x_max[0] - q)))
#     # print(-np.sqrt(2 * ddq * (q - model.x_min[0])))
# # dq_max = np.sqrt(2 * ddq * (model.x_max[:nq] - q_lin))
# # dq_min = -np.sqrt(2 * ddq * (q_lin - model.x_min[:nq])) 
# # print(dq_max.shape)
# exit()
for i in range(len(x_guess)):
    fig, ax = plt.subplots(2, 2)
    ax = ax.reshape(-1)
    for j in range(model.nq):
        ax[j].grid(True)
        ax[j].plot(x_guess[i][:, j], x_guess[i][:, j + nq])
        ax[j].scatter(x_guess[i][-1, j], x_guess[i][-1, j + nq], c='r', marker='x')
        q_lin, dq_max, dq_min = compute_acc_bounds(j)
        ax[j].plot(q_lin, dq_max, 'k--')
        ax[j].plot(q_lin, dq_min, 'r--')
        ax[j].set_title(f'Joint {j}')
        ax[j].set_xlabel('q')
        ax[j].set_ylabel('dq')
        ax[j].set_xlim([model.x_min[j], model.x_max[j]])
        ax[j].set_ylim([model.x_min[j + nq], model.x_max[j + nq]])
    plt.suptitle(f'Initial condition {i}')
    plt.tight_layout()
    plt.savefig(f'data/ic_{i}.png')
    plt.close()