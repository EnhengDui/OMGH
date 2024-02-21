from utils.loss import *
import time
# from update import *
def update_b_i(M1, M2, U1, U2, h1_i, h2_i, Z1, Z2, S_i, B, b_1, b_2, eta1, eta2):
    """
    参数:
    M, U, Z: 矩阵 M, U, Z
    h_i: 向量 h_i
    phi: 标量
    S: 矩阵 S
    b_1, b_2: 向量 b_1 和 b_2
    rho1, rho2, eta1, eta2: 正则化参数
    eta1, eta2是向量，其大小与b_1, b_2相同
    """
    sum_sij = torch.sum(S_i * B, dim=1)
    term1 = 2 * M1.T @ U1.T @ h1_i + 2 * M2.T @ U2.T @ h2_i
    term2 = 2 * phi * M1.T @ Z1 @ h1_i + 2 * phi * M2.T @ Z2 @ h2_i
    term3 = 2 * lambda_ * sum_sij
    term4 = rho1 * b_1 + rho2 * b_2 - eta1 - eta2

    b_i = (term1 + term2 + term3 + term4) / (rho1 + rho2)
    return b_i

# 投影到lp_box上
def project_box(x):
    xp = x
    xp[x > 1] = 1
    xp[x < -1] = -1
    return xp

def update_b1(b_i_next, eta1):
    b1 = project_box(b_i_next + eta1 / rho1)
    return b1

def project_shifted_Lp_ball(x, p):  # p=2
    # r是B的编码长度，x是待投影的输入 这里p=2
    norm_p = torch.linalg.norm(x, p)
    xp = (r**(1/p)) * x / norm_p
    return xp

def update_b2(b_i_next, eta2):
    b2 = project_shifted_Lp_ball(b_i_next + eta2 / rho2, 2)
    return b2


def update_B(M, U, H, Z, S, B, b1, b2, eta1, eta2):
    """更新矩阵B"""
    for i in range(B.shape[1]):  # 更新B的每一列
        if B.shape[1] != H.shape[1]:
            raise ValueError("B的列数应该等于输入的H的列数等于样本个数")

        h_i = H[:, i]
        S_i = S[i, :]
        # 1.更新b_i
        b_i = update_b_i(M, U, h_i, Z, S_i, B, b1, b2, eta1, eta2)

        # 2.更新b1
        b1 = update_b1(b_i, eta1)
        # 3.更新b2
        b2 = update_b2(b_i, eta2)

        # 4.更新 eta1
        eta1 = eta1 + rho1 * (b_i - b1)
        # 5.更新 eta2
        eta2 = eta2 + rho2 * (b_i - b2)

        # 更新B的第i列
        B[:, i] = b_i

    return B

def optimize_bi(i, M, U, H, Z, S, B, b1, b2, eta1, eta2):
    """更新bi"""
    # 更新B的第i列
    b_i = B[:, i]
    h_i = H[:, i]
    S_i = S[i, :]
    # 1.更新b_i
    b_i = update_b_i(M, U, h_i, Z, S_i, B, b1, b2, eta1, eta2)

    # 2.更新b1
    b1 = update_b1(b_i, eta1)
    # 3.更新b2
    b2 = update_b2(b_i, eta2)
    # 4.更新 eta1
    eta1 = eta1 + rho1 * (b_i - b1)
    # 5.更新 eta2
    eta2 = eta2 + rho2 * (b_i - b2)

    return b_i, b1, b2, eta1, eta2


def compute_std_obj(obj_list, history_size):
    obj_tensor = torch.tensor(obj_list[-history_size:], dtype=torch.float32)

    # 计算了 obj_list 中最后 history_size 个元素的标准差
    std_obj = torch.std(obj_tensor, unbiased=False)  # std_obj:[[0.00617]]

    # 标准差 std_obj 被归一化，即除以 obj_list 中最后一个元素的绝对值。
    # 这是为了将标准差的大小与目标函数值的大小放在相同的比例尺上
    std_obj = std_obj / torch.abs(obj_tensor[-1])
    # 返回的std_obj是一个标量,加了索引居然没报错 好像不是标量

    return std_obj

def ADMM_bi(M1, M2, U1, U2, H1, H2, Z1, Z2, S, B, all_params=None):
    initial_params = {'std_threshold': 1e-6, 'gamma_val': 1.0, 'gamma_factor': 0.99, \
                      'initial_rho': 5, 'learning_fact': 1 + 3 / 100, 'rho_upper_limit': 1000, 'history_size': 6,
                      'rho_change_step': 5, \
                      'rel_tol': 1e-5, 'stop_threshold': 1e-3, 'max_iters': 1e4, 'projection_lp': 2}

    if all_params == None:
        all_params = initial_params
    else:
        for k in initial_params.keys():
            if k not in all_params.keys():
                all_params[k] = initial_params[k]

    stop_threshold = all_params['stop_threshold']
    std_threshold = all_params['std_threshold']
    max_iters = all_params['max_iters']
    initial_rho = all_params['initial_rho']
    history_size = all_params['history_size']
    rho_change_step = all_params['rho_change_step']
    gamma_val = all_params['gamma_val']
    learning_fact = all_params['learning_fact']
    projection_lp = all_params['projection_lp']
    gamma_factor = all_params['gamma_factor']

    BEST_SOL = []
    B_SOL = []
    B1 = []
    B2 = []
    TIME_ELAPSED = []
    # i
    for i in range(B.size(1)):
        b_i = B[:, i]
        h1_i = H1[:, i]
        h2_i = H2[:, i]
        S_i = S[i, :]
        b_sol = b_i
        b1 = b_sol
        b2 = b_sol
        eta1 = torch.zeros_like(b1)
        eta2 = torch.zeros_like(b2)
        rho1 = initial_rho
        rho2 = rho1
        obj_list = []
        std_obj = 1

        # initiate the binary solution
        prev_idx = b_sol
        best_sol = prev_idx

        best_bin_obj = loss_function_bi(h1_i, h2_i, U1, U2, M1, M2, b_i, B, Z1, Z2, S_i)

        time_elapsed = 0
        for j in range(int(max_iters)):
            t1 = time.time()
            # 1.更新b_i
            b_sol = update_b_i(M1, M2, U1, U2, h1_i, h2_i, Z1, Z2, S_i, B, b1, b2, eta1, eta2)

            # 2.更新b1
            b1 = update_b1(b_i, eta1)
            # 3.更新b2
            b2 = update_b2(b_i, eta2)
            # 4.更新 eta1
            eta1 = eta1 + gamma_val * rho1 * (b_i - b1)
            # 5.更新 eta2
            eta2 = eta2 + gamma_val * rho2 * (b_i - b2)

            t2 = time.time()
            time_elapsed = time_elapsed + (t2 - t1)

            # increase rho1 and rho2
            if np.mod(j + 2, rho_change_step) == 0:
                rho1 = learning_fact * rho1
                rho2 = learning_fact * rho2
                gamma_val = max(gamma_val * gamma_factor, 1.0)

            # evaluate this iteration 计算相对误差 小于阈值则停止迭代
            temp1 = (torch.linalg.norm(b_sol - b1)) / max(torch.linalg.norm(b_sol), 2.2204e-16)
            temp2 = (torch.linalg.norm(b_sol - b2)) / max(torch.linalg.norm(b_sol), 2.2204e-16)
            if max(temp1.item(), temp2.item()) <= stop_threshold:
                print('iter: {j}, stop_threshold: {max(temp1, temp2):.6f}')
                break

            # 将当前解 `x_sol` 的目标函数值添加到一个列表中。
            # 如果列表长度达到 `history_size`，计算这些目标函数值的标准差。
            # 如果标准差小于或等于 `std_threshold`，则认为目标函数值已足够稳定，停止迭代。
            obj_list.append(loss_function_bi(h1_i, h2_i, U1, U2, M1, M2, b_sol, B, Z1, Z2, S_i)) # 连续的loss

            if len(obj_list) >= history_size:
                std_obj = compute_std_obj(obj_list, history_size)
                if std_obj <= std_threshold:
                    # print('iter: %d, std_threshold: %.6f' % (j, std_obj))
                    break

            # 生成一个布尔索引 cur_idx，它将解 x_sol 二值化：大于或等于0.5的值被视为1，小于0.5的值被视为0。
            # 计算二值化解的目标函数值，如果这个值比迄今为止记录的最好值 best_bin_obj 更低，则更新最好的目标函数值和对应的解。

            cur_idx = torch.where(b_sol >= 0.5, 1, -1)
            prev_idx = cur_idx  # 二进制解
            cur_obj = loss_function_bi(h1_i, h2_i, U1, U2, M1, M2, prev_idx, B, Z1, Z2, S_i) # 离散的loss

            # maintain the best binary solution so far; in case the cost function oscillates
            if best_bin_obj >= cur_obj:
                best_bin_obj = cur_obj
                best_sol = prev_idx

        # print(best_sol.shape)
        # print(best_sol)
        B[:, i] = best_sol
        # BEST_SOL.append(best_sol)
        # B_SOL.append(b_sol)
        # B1.append(b1)
        # B2.append(b2)
        TIME_ELAPSED.append(time_elapsed)

    # return BEST_SOL, B_SOL, B1, B2, TIME_ELAPSED
    return B

