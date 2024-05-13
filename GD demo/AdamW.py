import numpy as np
from utlis import compute_gradient
import copy

def adamw_update(t, x, m, v, lr, beta, type, weigh_decay=1e-2):
    
    grad_x1, grad_x2 = compute_gradient(x[0], x[1], type)


    m[0] = beta[0] * m[0] + (1 - beta[0]) * grad_x1
    m[1] = beta[0] * m[1] + (1 - beta[0]) * grad_x2

    v[0] = beta[1] * v[0] + (1 - beta[1]) * grad_x1 ** 2
    v[1] = beta[1] * v[1] + (1 - beta[1]) * grad_x2 ** 2

    m_hat = m / (1 - beta[0] ** t)
    v_hat = v / (1 - beta[1] ** t)
    ad_lr = lr / (1-beta[1]*np.sqrt(grad_x1**2 + grad_x2**2))
    x = x - ad_lr * (m_hat + weigh_decay * x) / (np.sqrt(v_hat) + 1e-4) 
    
    new_x = copy.copy(x)
    new_m = copy.copy(m)
    new_v = copy.copy(v)

    grad = np.array([grad_x1, grad_x2])
    
    return new_x, new_m, new_v, grad

