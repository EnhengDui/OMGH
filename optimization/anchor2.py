
# 随机初始化Q1、Q2、Bq 有Nq个锚点

# 锚点选择
# 从Q中随机选择N1个数据
# 从X中随机选择N2个数据
# 合成新的Q
# 计算Sqx
# 计算BqSqx 和 BqSqxB_t
import torch
from config.settings import *
from utils.evaluation import *
from dataloader.load_data import *
import torch.nn.functional as F

d1 = 4096
d2 = 300
c = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Q1 = torch.randn(d1, Nq).float().to(device)
Q2 = torch.randn(d2, Nq).float().to(device)
Bq = torch.randint(0, 2, (r, Nq)).float().to(device)
Bq = torch.where(Bq == 0, torch.tensor(-1.0), torch.tensor(1.0)).to(device)

# 生成一个随机的整数张量，其值在0到c-1之间
indices = torch.randint(0, c, (Nq,))
# 将这些整数转换为独热编码形式
Lq = F.one_hot(indices, num_classes=c).T.float().to(device)

data = load_pascal(load_i_dir, load_t_dir, load_l_dir)
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

for i, t, l in dataloader:
    print(torch.linalg.norm(Q1, 'fro') ** 2)
    print(torch.linalg.norm(Q2, 'fro') ** 2)
    print(torch.linalg.norm(Bq, 'fro') ** 2)
    print(torch.linalg.norm(Lq, 'fro') ** 2)
    indices = torch.randperm(Nq)[:N1].to(device)

    old_X1 = Q1[:, indices] # d1,22
    old_X2 = Q2[:, indices]  # d2,22
    old_B = Bq[:, indices] # 16,22
    old_L = Lq[:, indices]  # c,22

    X1_t = i.T.to(device)  # 4096,64
    X2_t = t.T.to(device)  # 300,64
    L_t = l.T.to(device)  # 20,64

    B_t = torch.randint(0, 2, (r, BATCH_SIZE)).float().to(device)
    B_t = torch.where(B_t == 0, torch.tensor(-1.0), torch.tensor(1.0)).to(device)  # 16,64
    indices = torch.randperm(Nt)[:N2].to(device) # 28,
    new_X1 = X1_t[:, indices]  # 4096,28
    new_X2 = X2_t[:, indices]  # 300,28
    new_B = B_t[:, indices]  # 16,28
    new_L = L_t[:, indices]  # 20,28

    # 将新旧数据按列拼接
    Q1 = torch.cat((old_X1, new_X1), dim=1)  # d1,Nq
    Q2 = torch.cat((old_X2, new_X2), dim=1)  # d2,Nq
    Bq = torch.cat((old_B, new_B), dim=1)  # r,Nq
    Lq = torch.cat((old_L, new_L), dim=1)  # 20,Nq

    # 计算Sqx N1*N2
    Sqx = sqx_simlarity(Lq, L_t).to(device)  # Nq,Nt
    # 计算BqSqx 和 BqSqxB_t
    BqSqx = Bq @ Sqx  # r,N2

    BqSqxB_t = Bq @ Sqx @ B_t.T  # r,r







