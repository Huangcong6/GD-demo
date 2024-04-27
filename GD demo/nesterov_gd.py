import numpy as np
from utlis import compute_gradient
import copy

def nesterov_gd_update(x, v, lr, momentum, type):
    # 计算梯度
    grad_x1, grad_x2 = compute_gradient(x[0], x[1], type)


    # 更新动量项
    v[0] = momentum * v[0] - lr * grad_x1
    v[1] = momentum * v[1] - lr * grad_x2
    # 计算预估计的下一步位置
    pred_x1 = x[0] + v[0]
    pred_x2 = x[1] + v[1]
    # 计算预估计梯度
    pred_grad_x1, pred_grad_x2 = compute_gradient(pred_x1, pred_x2, type)
    # Nesterov加速梯度下降更新
    x[0] = x[0] - lr * (pred_grad_x1 + momentum*v[0])
    x[1] = x[1] - lr * (pred_grad_x2 + momentum*v[1])

    new_x = copy.copy(x)
    new_v = copy.copy(v)
    grad = np.array([grad_x1, grad_x2])
    return new_x, new_v , grad