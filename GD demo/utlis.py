import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.animation

# 定义目标函数
def target_function(x1, x2, type):
    if type == 1:
        return 4 * x1**2 - 2.1 * x1**4 + 1/3 * x1**6 + x1 * x2 - 4 * x2**2 + 4 * x2**4
    elif type == 2:
        return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2
    elif type == 3:
        return x1**4 - 2 * x1**2 + x2**2
    elif type == 4:
        return (x1**2 + x2**2 -1)**2
    elif type == 5:
        return (1-x1**2)**2 + 100*(x2-x1**2)**2
    

    # 计算梯度
def compute_gradient(x1, x2, type):
    if type == 1:
        grad_x1 = 8*x1 - 8.4*x1**3 + 2*x1**5 + x2
        grad_x2 = x1 - 8*x2 + 16*x2**3
        return grad_x1, grad_x2 

    elif type == 2:
        grad_x1 = 0.52*x1 - 0.48*x2
        grad_x2 = 0.52*x2 - 0.48*x1
        return grad_x1, grad_x2 

    elif type == 3:
        grad_x1 = 4*x1**3 - 4*x1
        grad_x2 = 2*x2
        return grad_x1, grad_x2 

    elif type == 4:
        grad_x1 = 2*(x1**2 + x2**2 -1) * 2*x1
        grad_x2 = 2*(x1**2 + x2**2 -1) * 2*x2
        return grad_x1, grad_x2 

    elif type == 5:
        grad_x1 = -4*x1*(1-x1) + 400*x1**3 - 400*x1*x2
        grad_x2 = 200*x2 - 200*x1**2
        return grad_x1, grad_x2 

def show_value_image(n_iter, fvalue, label):
    plt.plot(range(1, n_iter + 1), fvalue, label=label)
    # 添加函数值为saddle的水平线
    saddle_value = 1
    plt.axhline(saddle_value, color='r', linestyle='--', label='Saddle Value')
    
    # 添加网格线
    plt.grid(True, color='gray', linestyle='--')

    # 设置x轴和y轴的标签
    plt.xlabel("Iteration")
    plt.ylabel("Function value")
    
    
    # 显示图例
    plt.legend()
    
    # 显示图像
    plt.show()
    

def function_value_image(n_iter, gd_fvalue, nag_fvalue, adamw_fvalue, cubic_fvalue, saddle_value,
                         label1='GD', label2='NAG', label3='AdamW', label4='Cubic'):
    # 绘制第一个方法的函数值变化
    plt.plot(range(1, n_iter + 1), gd_fvalue, label=label1)
    
    # 绘制第二个方法的函数值变化
    plt.plot(range(1, n_iter + 1), nag_fvalue, label=label2)
    
    # 绘制第二个方法的函数值变化
    plt.plot(range(1, n_iter + 1), adamw_fvalue, label=label3)
    
    # 绘制第二个方法的函数值变化
    plt.plot(range(1, n_iter + 1), cubic_fvalue, label=label4)
    
    # 添加函数值为saddle的水平线
    plt.axhline(saddle_value, color='r', linestyle='--', label='Saddle Value')
    
    # 添加网格线
    plt.grid(True, color='gray', linestyle='--')

    # 设置x轴和y轴的标签
    plt.xlabel("Iteration")
    plt.ylabel("Function value")
    
    
    # 显示图例
    plt.legend()
    
    plt.savefig("Iteration_Fvalue.png")
    # 显示图像
    plt.show()

def function_value_image_non(n_iter, gd_fvalue, nag_fvalue, adamw_fvalue, cubic_fvalue,
                         label1='GD', label2='NAG', label3='AdamW', label4='Cubic'):
    # 绘制第一个方法的函数值变化
    plt.plot(range(1, n_iter + 1), gd_fvalue, label=label1)
    
    # 绘制第二个方法的函数值变化
    plt.plot(range(1, n_iter + 1), nag_fvalue, label=label2)
    
    # 绘制第二个方法的函数值变化
    plt.plot(range(1, n_iter + 1), adamw_fvalue, label=label3)
    
    # 绘制第二个方法的函数值变化
    plt.plot(range(1, n_iter + 1), cubic_fvalue, label=label4)

    
    # 添加网格线
    plt.grid(True, color='gray', linestyle='--')

    # 设置x轴和y轴的标签
    plt.xlabel("Iteration")
    plt.ylabel("Function value")
    
    
    # 显示图例
    plt.legend()
    
    plt.savefig("Iteration_Fvalue.png")
    # 显示图像
    plt.show()


#绘画梯度下降路径
def descent_path(x_history,method_type,func_type, local_min, saddle_point, step):
    
    # 创建数据点
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)

    Y = target_function(X1, X2, func_type)

    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contour(X1, X2, Y, 50)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    if method_type == 'GD':
        ax.set_title('Gradient Descent')
    elif method_type == 'NAG':
        ax.set_title('Nesterov Descent')
    elif method_type == 'AdamW':
        ax.set_title('AdamW Descent')
    elif method_type == "Cubic":
        ax.set_title("Cubic Newton")
    else:
        print('type参数错误')  
    
    # 初始化红点
    scatter = ax.scatter(x_history[0,0], x_history[0,1], s=100, color='red', label="Trajectory")
    # 添加一个黄色的鞍点
    ax.scatter(*saddle_point, s=100, color='blue', label='Saddle Point')
    # 添加一个局部最小值点
    ax.scatter(local_min[:, 0], local_min[:, 1], s=100, color="black", label="Local Minimum")

    # # 定义梯度下降路径
    x_history = x_history [::step]
    # x_history = np.array(x_history)  # 梯度下降路径的坐标点
    
    # 存储过去生成的红点的位置
    past_scatter_positions = []
    
    # 更新函数，每帧更新红点的位置
    def update(frame):
        x, y = x_history[frame]
        past_scatter_positions.append([x, y])  # 存储当前红点的位置
        scatter.set_offsets(past_scatter_positions)  # 更新所有已生成的红点的位置
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(x_history), interval=1000, repeat=False)
    
    # 显示动画
    plt.legend()
    plt.show()
    
    # writervideo = matplotlib.animation.writers["ffmpeg"](fps=30)
    # 保存动画
    if method_type == 'GD':
        anim.save('GD.gif', writer="pillow")
    elif method_type == 'NAG':
        anim.save('NAG.gif', writer="pillow")
    elif method_type == 'AdamW':
        anim.save('AdamW_noise.gif', writer="pillow")
    elif method_type == "Cubic":
        anim.save("CubicNewton.gif", writer="pillow")

    plt.close()

#绘画梯度下降路径
def descent_path_non(x_history,method_type,func_type, local_min , step):
    
    # 创建数据点
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)

    Y = target_function(X1, X2, func_type)

    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contour(X1, X2, Y, 50)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    if method_type == 'GD':
        ax.set_title('Gradient Descent')
    elif method_type == 'NAG':
        ax.set_title('Nesterov Descent')
    elif method_type == 'AdamW':
        ax.set_title('AdamW Descent')
    elif method_type == "Cubic":
        ax.set_title("Cubic Newton")
    else:
        print('type参数错误')  
    
    # 初始化红点
    scatter = ax.scatter(x_history[0,0], x_history[0,1], s=100, color='red', label="Trajectory")

    # 添加一个局部最小值点
    ax.scatter(local_min[:, 0], local_min[:, 1], s=100, color="black", label="Local Minimum")

    # # 定义梯度下降路径
    x_history = x_history [::step]
    # x_history = np.array(x_history)  # 梯度下降路径的坐标点
    
    # 存储过去生成的红点的位置
    past_scatter_positions = []
    
    # 更新函数，每帧更新红点的位置
    def update(frame):
        x, y = x_history[frame]
        past_scatter_positions.append([x, y])  # 存储当前红点的位置
        scatter.set_offsets(past_scatter_positions)  # 更新所有已生成的红点的位置
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(x_history), interval=1000, repeat=False)
    
    # 显示动画
    plt.legend()
    plt.show()
    
    # writervideo = matplotlib.animation.writers["ffmpeg"](fps=30)
    # 保存动画
    if method_type == 'GD':
        anim.save('GD.gif', writer="pillow")
    elif method_type == 'NAG':
        anim.save('NAG_noise.gif', writer="pillow")
    elif method_type == 'AdamW':
        anim.save('AdamW_noise.gif', writer="pillow")
    elif method_type == "Cubic":
        anim.save("CubicNewton.gif", writer="pillow")

    plt.close()


# 自适应噪声
def adaptive_noise_adam(grad, T, m_n, v_n, learning_rate, mu, beta1_n=0.9, beta2_n=0.999, epsilon = 0.0001):
    
    T = T + 1
    #mu =  mu * 2 #让mu随着添加噪声的次数，逐渐减少,让开始添加的噪声更大
    m_n = beta1_n * m_n + 0.5 * (1 - beta1_n) * grad[0] + 0.5 * (1 - beta1_n) * grad[1]
    v_n = beta2_n * v_n + 0.5 * (1 - beta2_n) * (grad[0] ** 2) + 0.5 * (1 - beta2_n) * (grad[1] ** 2)
    # 计算偏差修正项
    m_hat_n = m_n / (1 - beta1_n ** T)
    v_hat_n = v_n / (1 - beta2_n ** T)
    # 计算自适应步长
    adaptive_step_size = learning_rate * (np.sqrt(epsilon) / (np.sqrt(abs(v_hat_n)) + epsilon))

    # 计算自适应噪声，其标准差与自适应步长的平方根成比例
    noise = 30*np.random.uniform(-0.1,0.1,grad.shape) * np.sqrt(mu * adaptive_step_size)
    
    return m_n, v_n, noise,T, mu

def is_value_down(value_noise, value):
    if value_noise - value < 0:
        return True
    else:
        return False
        






    
 