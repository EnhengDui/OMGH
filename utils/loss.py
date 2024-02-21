from config.settings import *
import numpy as np
import torch
def initialize_ortho(row, col):
    M = np.random.randn(row, col)
    M_ortho, _ = np.linalg.qr(M)  # 对M进行QR分解，得到正交矩阵Q
    return M_ortho

def initialize_ortho_torch(row, col):
    M = torch.randn(row, col)
    M_ortho, _ = torch.linalg.qr(M)  # 对M进行QR分解，得到正交矩阵Q
    return M_ortho
def calculate_loss_hs(H1, H2, U1, U2, M1, M2, B, Z1, Z2, Ls, phi, lambda_, gamma):
    """
    参数:
    U, M, Z, B, H, Ls 是 numpy 数组
    H: 矩阵 H(input)
    U: 矩阵 U
    M: 矩阵 M
    B: 矩阵 B
    Z: 矩阵 Z
    Ls: 第一步得到的相似度矩阵的拉普拉斯矩阵 Ls
    phi: 标量，与第二项相关的系数
    lambda_: 标量，与第三项相关的系数
    gamma: 标量，与第四项相关的系数
    """
    term1 = torch.linalg.norm(H1 - U1 @ M1 @ B, 'fro') ** 2 + torch.linalg.norm(H2 - U2 @ M2 @ B, 'fro') ** 2
    # print("term1", term1)

    term2 = phi * torch.linalg.norm(Z1 @ H1 - M1 @ B, 'fro') ** 2 + phi * torch.linalg.norm(Z2 @ H2 - M2 @ B, 'fro') ** 2
    # print("term2", term2)
    term3 = lambda_ * torch.trace(B @ Ls @ B.T)
    # print("term3", term3)
    term4 = gamma * torch.linalg.norm(U1, 'fro') ** 2 + gamma * torch.linalg.norm(U2, 'fro') ** 2
    # print("term4", term4)
    loss_hs = term1 + term2 + term3 + term4

    return loss_hs

def check_elements(B):
    """检查矩阵中的元素是否只包含-1和1"""
    return np.all(np.logical_or(B == -1, B == 1))

# B相关的loss函数 用的是矩阵
def loss_function_B(H1, U1, M1, Z1, H2, U2, M2, Z2, B,  Ls, phi, lambda_):
    """计算B的损失函数值"""
    # 确保B中的元素只有-1和1
    if not check_elements(B):
        raise ValueError("矩阵B中的元素不只包含-1和1")

    term1 = np.linalg.norm(H1 - U1 @ M1 @ B, 'fro') ** 2 + np.linalg.norm(H2 - U2 @ M2 @ B, 'fro') ** 2
    # print("term1", term1)
    term2 = phi * np.linalg.norm(Z1 @ H1 - M1 @ B, 'fro') ** 2 + phi * np.linalg.norm(Z2 @ H2 - M2 @ B, 'fro') ** 2
    # print("term2", term2)
    term3 = lambda_ * np.trace(B @ Ls @ B.T)
    # print("term3", term3)
    loss = term1 + term2 + term3

    return loss

# bi相关的矩阵 用的B的列向量
def loss_function_bi(h1_i, h2_i, U1, U2, M1, M2, b_i, B, Z1, Z2, S_i):
    """计算bi的损失函数值"""
    b_i = b_i.float()

    # S_i 的形状被广播以匹配 B 的形状，然后进行逐元素乘法，最后对元素按行求和
    sum_sij = torch.sum(S_i * B, dim=1)  # np axis=1 torch dim=1
    # # 确保b_i中的元素只有-1和1
    # if not check_elements(b_i):
    #     raise ValueError("向量b_i中的元素不只包含-1和1")


    term1 = torch.linalg.norm(h1_i - U1 @ M1 @ b_i, 2) ** 2 + torch.linalg.norm(h2_i - U2 @ M2 @ b_i, 2) ** 2
    term2 = phi * torch.linalg.norm(Z1 @ h1_i - M1 @ b_i, 2) ** 2 + phi * torch.linalg.norm(Z2 @ h2_i - M2 @ b_i, 2) ** 2

    term3 = -2 * lambda_ * b_i.T @ sum_sij  # @是矩阵乘法, *是逐元素乘法 向量转置有warning

    loss = term1 + term2 + term3
    return loss

# def loss_function_bi(h1_i, h2_i, U1, U2, M1, M2, b_i, B, Z1, Z2, S_i):
#     """计算bi的损失函数值"""
#     # S_i 的形状被广播以匹配 B 的形状，然后进行逐元素乘法，最后对元素按行求和
#     sum_sij = np.sum(S_i * B, axis=1)  # np axis=1 torch dim=1
#     # # 确保b_i中的元素只有-1和1
#     # if not check_elements(b_i):
#     #     raise ValueError("向量b_i中的元素不只包含-1和1")
#
#     term1 = np.linalg.norm(h1_i - U1 @ M1 @ b_i, 2) ** 2 + np.linalg.norm(h2_i - U2 @ M2 @ b_i, 2) ** 2
#     term2 = phi * np.linalg.norm(Z1 @ h1_i - M1 @ b_i, 2) ** 2 + phi * np.linalg.norm(Z2 @ h2_i - M2 @ b_i, 2) ** 2
#     term3 = -2 * lambda_ * b_i.T @ sum_sij  # @是矩阵乘法, *是逐元素乘法
#
#     loss = term1 + term2 + term3
#     return loss

def loss_M(M1, M2, H1, H2, Z1, Z2, U1, U2, B, phi):
    term = M1 @ B @ H1.T @ (phi*Z1 + U1) + M2 @ B @ H2.T @ (phi*Z2 + U2)
    loss = np.trace(term)
    return loss

def loss_Z(M1, H1, Z1, M2, H2, Z2, B):
    term = Z1 @ H1 @ B.T @ M1 + Z2 @ H2 @ B.T @ M2
    loss = np.trace(term)
    return loss

def loss_U(H1, U1, M1, H2, U2, M2, B, gamma):
    term1 = np.linalg.norm(H1 - U1 @ M1 @ B, 'fro') ** 2 + np.linalg.norm(H2 - U2 @ M2 @ B, 'fro') ** 2
    term2 = np.linalg.norm(U1,'fro') ** 2 + np.linalg.norm(U2,'fro') ** 2
    loss = term1 + gamma * term2
    return loss







