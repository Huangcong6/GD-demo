import numpy as np
from utlis import compute_gradient
import copy 
#operator for one update

def gradient_descent_update(x, lr, type):
    grad_x1, grad_x2 = compute_gradient(x[0], x[1], type)
    x[0] = x[0] - lr * grad_x1
    x[1] = x[1] - lr * grad_x2

    new_x = copy.copy(x)
    grad = np.array([grad_x1, grad_x2])
    
    return new_x, grad