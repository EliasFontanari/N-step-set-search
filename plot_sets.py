import numpy as np
import casadi as cs
import adam_model
import parser
import copy
import random
import sympy


from sympy import symbols
from sympy.plotting import plot


q = symbols('q')
ddq_max = symbols('ddq')
q_max = sympy.sqrt(2*2*(sympy.pi-q))
q_min = -sympy.sqrt(2*2*(-q+sympy.pi))

print(q_max)

p1=plot(sympy.sqrt(4*(sympy.pi-q)),(q,-np.pi,np.pi))
#p2=plot(q_min,show =False)
#p1.append(p2[0])
p1.show()


x=symbols('x')
plot(x**2, (x, -5, 5))