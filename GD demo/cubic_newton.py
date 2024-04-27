import numpy as np
from utlis import target_function, compute_gradient


# 全局上界函数的梯度
def gradient_global_upper_bound(y, x, L, t, type):
    grad_x1, grad_x2 = compute_gradient(x[0], x[1], type)
    approx_hv_1, approx_hv_2 = approx_hessian(x, y, type, t)

    get_gradiet_x1 = grad_x1 + approx_hv_1 + L/2*np.linalg.norm(y-x)*(y[0] - x[0]) 
    get_gradiet_x2 = grad_x2 + approx_hv_2 + L/2*np.linalg.norm(y-x)*(y[1] - x[1]) 
    return get_gradiet_x1, get_gradiet_x2

def approx_hessian(x, y, type, t):
    grad_x1, grad_x2 = compute_gradient(x[0], x[1], type)
    new_x = x + t * (y - x)
    new_grad_x1, new_grad_x2 = compute_gradient(new_x[0], new_x[1], type)

    approx_hv_1 = (new_grad_x1 - grad_x1) / t
    approx_hv_2 = (new_grad_x2 - grad_x2) / t
    return approx_hv_1, approx_hv_2

# operator for one update
def cubic_newton_update(inner_iter, x, inner_lr, L, t, type):
    y = x + 0.01
    # solve the inner sub-problem
    for i in range(inner_iter):
        gradient_y1, gradient_y2 = gradient_global_upper_bound(y, x, L, t, type)
        y[0] = y[0] - inner_lr * gradient_y1
        y[1] = y[1] - inner_lr * gradient_y2
    
    return y

