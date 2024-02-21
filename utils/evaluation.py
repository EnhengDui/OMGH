import numpy as np
import torch
def get_hash_code(M, Z, x):
    y = M.T @ Z @ x
    return torch.where(y >= 0, 1, -1)

def su_simlarity(labels):
    norms = torch.norm(labels, p=2, dim=1, keepdim=True) # 按行求2范数,保留原维度
    labels = labels / norms
    cosine_similarity = torch.mm(labels, labels.T)
    return cosine_similarity

# c*N1 c*N2
def sqx_simlarity(Lq, L_t):
    norms = torch.norm(Lq, p=2, dim=0, keepdim=True) # 按列求2范数,保留原维度
    Lq = Lq / norms
    norms = torch.norm(L_t, p=2, dim=0, keepdim=True)  # 按列求2范数,保留原维度
    L_t = L_t / norms
    cos_similarity = torch.mm(Lq.T, L_t)
    return cos_similarity