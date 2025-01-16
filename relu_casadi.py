import casadi as cs
import torch
import torch.nn as nn
import l4casadi as l4c

class ReLuFun(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # Single neuron (1 output)
        self.relu = nn.ReLU()                 # ReLU activation

        with torch.no_grad():  # Disable gradient tracking
            self.linear.weight.fill_(1)  # Set weights to 1
            self.linear.bias.fill_(0)   

    def forward(self, x):
        x = self.linear(x)  # Pass input through the linear layer
        x = self.relu(x)    # Apply ReLU activation
        return x
    



pyTorch_model = ReLuFun()
l4c_model = l4c.L4CasADi(pyTorch_model, device='cpu')  # device='cuda' for GPU

x_sym = cs.MX.sym('x', 1, 1)
y_sym = l4c_model(x_sym)
f = cs.Function('y', [x_sym], [y_sym])
df = cs.Function('dy', [x_sym], [cs.jacobian(y_sym, x_sym)])
ddf = cs.Function('ddy', [x_sym], [cs.hessian(y_sym, x_sym)[0]])

x = cs.DM([[0.], [2.]])
print(l4c_model(x))
print(f(x))
print(df(x))
print(ddf(x))




