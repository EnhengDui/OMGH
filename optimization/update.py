from lpbox_admm import *
from utils.loss import calculate_loss_hs
import os
import torch
import scipy.io as sio
from criterion.mAP import *
from criterion.criterion import *
from utils.evaluation import get_hash_code
from dataloader.load_data import *
from utils.evaluation import *

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

print(device)

def optimize(H_train, S_total, BATCH_SIZE,max_iters=1000, tol=1e-6):
# def optimize(params, compute_loss, update_params, max_iters=1000, tol=1e-6):
    """
    参数:
    - params: 优化参数的初始值。
    - compute_loss: 一个函数，根据当前的参数计算损失。
    - compute_gradient: 一个函数，根据当前的参数计算梯度。
    - update_params: 一个函数，根据当前的参数和梯度更新参数。
    - max_iters: 最大迭代次数。
    - tol: 收敛的容忍度，当连续两次迭代的参数变化小于这个值时停止迭代。

    返回:
    - params: 优化后的参数。
    - loss_history: 每次迭代的损失历史。
    """

    # 初始化M和Z, MZ有正交限制
    M1 = initialize_ortho_torch(k, r).float().to(device)

    M2 = initialize_ortho_torch(k, r).float().to(device)


    Z1 = initialize_ortho_torch(k, d).float().to(device)

    Z2 = initialize_ortho_torch(k, d).float().to(device)

    # U M Z B需要初始化 ， 均值为0，方差为1的正态分布
    U1 = torch.randn(d, k).float().to(device)

    U2 = torch.randn(d, k).float().to(device)

    # B = np.random.choice([-1, 1], (r, N))
    # B = torch.from_numpy(B)
    B = torch.randint(0, 2, (r, N))  # 左闭右开
    B = torch.where(B == 0, torch.tensor(-1.0), torch.tensor(1.0)).float().to(device)

    # 用均匀分布初始化B
    # B = np.random.uniform(size=(r, N))  # 均匀分布
    # B = np.where(B >= 0.5, 1, -1).astype('float32')

    # 均匀分布初始化B
    # B = torch.rand(r, N) # 0,1均匀分布,左闭右开
    # B = torch.where(B >= 0.5, torch.tensor(1.0), torch.tensor(-1.0))

    history_size = 5
    std_threshold = 1e-6
    loss_history = [] # 不用改，添加时记得使用item

    time_elapsed = 0
    num_batch = 0
    for H1, H2, S in generate_batches_torch(H_train, S_total, BATCH_SIZE):
    # for H1, H2, S in generate_subbatches(H_train, S_total):

        num_batch += 1
        print(f'-------------------------------------------------第 {num_batch}批数据正在优化-----------------------------------------')
        # if num_batch == 2:
        #     break
        for i in range(max_iters):
            t1 = time.time()
            # 计算局部拉普拉斯矩阵 Ls
            Ds = torch.diag(torch.sum(S, dim=1) / 2)  # numpy为axis
            Ls = Ds - (S + S.T) / 2
            # 计算当前参数的损失
            # print("更新前")
            loss = calculate_loss_hs(H1, H2, U1, U2, M1, M2, B, Z1, Z2, Ls, phi, lambda_, gamma)
            # loss = compute_loss(params)
            loss_history.append(loss.item())

            # 打印损失，或者进行其他形式的进度展示
            print(f'Iteration {i+1}/{max_iters}, Loss: {loss.item()}')

            # 如果满足收敛条件，则停止迭代
            # if i > 0 and np.abs(loss_history[-2] - loss_history[-1]) < tol:
            #     print('Convergence criteria met.')
            #     break

            if len(loss_history) >= history_size:
                std_obj = compute_std_obj(loss_history, history_size)
                if std_obj.item() <= std_threshold:
                    # print('iter: %d, std_threshold: %.6f' % (j, std_obj))
                    break

            # 更新参数

            # 更新B-bi
            # lb_b = loss_function_B(H1, U1, M1, Z1, H2, U2, M2, Z2, B, Ls, phi, lambda_)
            # print(f'更新前的B的loss为{lb_b}')
            B = ADMM_bi(M1, M2, U1, U2, H1, H2, Z1, Z2, S, B)
            # lb_a = loss_function_B(H1, U1, M1, Z1, H2, U2, M2, Z2, B, Ls, phi, lambda_)
            # print(f'更新后的B的loss为{lb_a}')

            # 1.更新M1、M2
            # lm_b = loss_M(M1, M2, H1, H2, Z1, Z2, U1, U2, B, phi)
            # print(f'更新前的M的loss为{lm_b}')

            M1_matrix = B @ H1.T @ ((phi * Z1.T) + U1)
            # 在PyTorch中，SVD返回的是(U, S, V)，其中V已经是V.T（在NumPy中是U, S, VH，其中VH是V.T）。
            L1, S1, R1T = torch.linalg.svd(M1_matrix, full_matrices=False)
            M1 = R1T.T @ L1.T

            M2_matrix = B @ H2.T @ ((phi * Z2.T) + U2)
            L2, S2, R2T = torch.linalg.svd(M2_matrix, full_matrices=False)
            M2 = R2T.T @ L2.T

            # lm_a = loss_M(M1, M2, H1, H2, Z1, Z2, U1, U2, B, phi)
            # print(f'更新后的M的loss为{lm_a}')

            # 2.更新Z
            # lz_b = loss_Z(M1, H1, Z1, M2, H2, Z2, B)
            # print(f'更新前的Z的loss为{lz_b}')
            Z1_matrix = H1 @ B.T @ M1.T
            L3, S3, R3T = torch.linalg.svd(Z1_matrix, full_matrices=False)
            Z1 = R3T.T @ L3.T

            Z2_matrix = H2 @ B.T @ M2.T
            L4, S4, R4T = torch.linalg.svd(Z2_matrix, full_matrices=False)
            Z2 = R4T.T @ L4.T

            # lz_a = loss_Z(M1, H1, Z1, M2, H2, Z2, B)
            # print(f'更新前的Z的loss为{lz_a}')

            # 3.更新U
            # lu_b = loss_U(H1, U1, M1, H2, U2, M2, B, gamma)
            # print(f'更新前的U的loss为{lu_b}')

            I1 = torch.eye(M1.size(0)).to(device) # 单位矩阵I，其大小与 M1 @ M1.T 相同
            inverse_term = torch.linalg.pinv(gamma * I1 + M1 @ B @ B.T @ M1.T)        # 伪逆
            U1 = H1 @ B.T @ M1.T @ inverse_term


            I2 = torch.eye(M2.size(0)).to(device)  # 单位矩阵I，其大小与 M1 @ M1.T 相同
            inverse_term = torch.linalg.pinv(gamma * I2 + M2 @ B @ B.T @ M2.T)
            U2 = H2 @ B.T @ M2.T @ inverse_term

            # lu_a = loss_U(H1, U1, M1, H2, U2, M2, B, gamma)
            # print(f'更新前的U的loss为{lu_a}')

            t2 = time.time()
            time_elapsed = time_elapsed + (t2 - t1)
            print("时间",time_elapsed)
            # print("更新后")
            # loss = calculate_loss_hs(H1, H2, U1, U2, M1, M2, B, Z1, Z2, Ls, phi, lambda_, gamma)
            print(loss)

    return M1, M2, Z1, Z2, B




def OMGH(dataloader,max_iters=1000):
    X1_acc = []
    B_acc = []
    X2_acc = []
    label_acc = []

    E1_t = torch.zeros(d1, r).float().to(device)
    E2_t = torch.zeros(d2, r).float().to(device)
    H_t = torch.zeros(r, r).float().to(device)
    
    E3_t = torch.zeros(d1, d1).float().to(device)
    E4_t = torch.zeros(d2, d2).float().to(device)

    U1 = torch.randn(d1, k).float().to(device)
    U2 = torch.randn(d2, k).float().to(device)
    M1 = torch.randn(k, r).float().to(device)
    M2 = torch.randn(k, r).float().to(device)
    P1 = torch.randn(k, d1).float().to(device)
    P2 = torch.randn(k, d2).float().to(device)

    for img, txt, label in dataloader:
        # 先加进去试试 否则刚开始没有Q
        X1_acc.append(img)
        X2_acc.append(txt)
        label_acc.append(label)

        X1_t = img.T.to(device)
        X2_t = txt.T.to(device)
        B_t = torch.randint(0, 2, (r, BATCH_SIZE)).float().to(device)
        B_t = torch.where(B_t == 0, torch.tensor(-1.0), torch.tensor(1.0)).to(device)

        E1_t = E1_t + X1_t @ B_t.T
        E2_t = E2_t + X2_t @ B_t.T
        H_t = H_t + B_t @ B_t.T
        
        E3_t = E3_t + X1_t @ X1_t.T
        E4_t = E4_t + X2_t @ X2_t.T

        Sxx = su_simlarity(label).to(device)
        # Sqx =

        # 从旧数据中随机选择一部分
        indices = torch.randperm(len(X1_acc))[:(Nq * Nq / (Nq + Nt))]
        selected_old_X1 = torch.stack([X1_acc[i] for i in indices])
        selected_old_X2 = torch.stack([X2_acc[i] for i in indices])
        if B_acc:
            selected_old_B = torch.stack([B_acc[i] for i in indices])
        else:
            selected_new_B = B_t[:, indices]

        # 从新数据中随机选择一批锚点
        indices = torch.randperm(img.size(0))[:(Nq * Nt / (Nq + Nt))]
        # selected_new_X1 = img[indices]
        # selected_new_X2 = txt[indices]
        selected_new_B = B_t[:, indices]
        selected_new_l = label[indices].to(device)

        for i in range(max_iters):
            # 更新U1 U2
            I = torch.eye(k).to(device)
            U1 = E1_t @ M1.T @ torch.pinverse((M1 @ H_t @ M1.T + gamma/alpha * I))

            U2 = E2_t @ M2.T @ torch.pinverse((M2 @ H_t @ M2.T + gamma/alpha * I))

            # 更新M1 M2
            # I = torch.eye(U1.size(1)).to(device)
            I = torch.eye(k).to(device)
            M1 = torch.pinverse(alpha * U1.T @ U1 + miu * I) @ (alpha * U1.T @ E1_t + miu * P1 @ E1_t) @ torch.pinverse(H_t)
            M2 = torch.pinverse(alpha * U2.T @ U2 + miu * I) @ (alpha * U2.T @ E2_t + miu * P2 @ E2_t) @ torch.pinverse(H_t)

            # 更新P1 P2
            I = torch.eye(E3_t.size(0),E3_t.size(1)).to(device)
            P1 = M1 @ E1_t.T @ torch.pinverse(E3_t + gamma/alpha * I)
            I = torch.eye(E4_t.size(0), E4_t.size(1)).to(device)
            P2 = M2 @ E2_t.T @ torch.pinverse(E4_t + gamma/alpha * I)

            # 更新B
            epsilon = 1e-10
            B_t = torch.sign(alpha * M1.T @ U1.T @ X1_t + (1 - alpha) * M2.T @ U2.T @ X2_t +
                        miu * M1.T @ P1 @ X1_t + miu * M2.T @ P2 @ X2_t
                        + lambda_ * B_t @ Sxx + epsilon)
            # + lambda_ * Bq @ Sqx)

            term1 = alpha * torch.norm(X1_t - U1 @ M1 @ B_t, 'fro') ** 2 + (1 - alpha) * torch.norm(X2_t - U2 @ M2 @ B_t, 'fro') ** 2 
            term2 = -lambda_ * torch.trace(B_t @ Sxx @ B_t.T) 
            term3 = miu * (torch.norm(P1 @ X1_t - M1 @ B_t, 'fro') ** 2 + torch.norm(P2 @ X2_t - M2 @ B_t, 'fro') ** 2)
            term4 = gamma * (torch.norm(U1, 'fro') ** 2 + torch.norm(U2, 'fro') ** 2 + torch.norm(M1, 'fro') ** 2 + torch.norm(M2, 'fro') ** 2 + torch.norm(P1, 'fro') ** 2 + torch.norm(P2, 'fro') ** 2)
            loss = term1 + term2 + term3 + term4
            print(f'第{i}次迭代的损失为{loss}')
            
if __name__ == "__main__":
    """
        初始化矩阵
    """
    print('...Data loading is beginning...')
    data = load_pascal(load_i_dir, load_t_dir, load_l_dir)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


    print('...Data loading is completed...')
    # H_train = H_train.to(device)
    # S_total = S_total.to(device)
    # M1_best, M2_best, Z1_best, Z2_best, B = optimize(H_train, S_total, BATCH_SIZE)
    OMGH(dataloader)



    # # for img, txt, labels in generate_test_batches(X1, X2, labels, BATCH_SIZE):
    # img_code = get_hash_code(M1_best, Z1_best, X1).T
    #
    # txt_code = get_hash_code(M2_best, Z2_best, X2).T
    #
    #
    # print('...Evaluation on testing data...')
    #
    # img_to_txt = calculate_map(img_code, txt_code, labels, labels)
    # # img_to_txt = calc_mAP(img_code, txt_code, labels)
    # print('...Image to Text MAP = {}'.format(img_to_txt.item()))
    #
    # txt_to_img = calculate_map(txt_code, img_code, labels, labels)
    # # txt_to_img = calc_mAP(txt_code, img_code, labels)
    # print('...Text to Image MAP = {}'.format(txt_to_img.item()))
    #
    # print('...Average MAP = {}'.format(((img_to_txt.item() + txt_to_img.item()) / 2.)))



