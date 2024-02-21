# batch_size
BATCH_SIZE = 64 # 如果分成四个子图 得提前算一下
# num_batches = N // BATCH_SIZE  # //表示整数除法

# 假设维度
# d = 16  # 输入特征的维度
d1 = 4096
d2 = 300
# N_total = 38000  # 样本数_total(图像+文本)
# N = N_total//2  # 样本数(图像或文本)
N = BATCH_SIZE # batch_size
k = 128  # 子空间的维度 128
# k1 = 100
# k2 = 50
r = 16  # 哈希编码长度

Nq = 50
Nt = BATCH_SIZE
N1 = round(Nq*Nq/(Nq+Nt))
N2 = round(Nq*Nt/(Nq+Nt))
if N1+N2 != Nq:
    raise ValueError("N1+N2 != Nq")

# Z和M有正交限制
# k不需要和输入的维度d相等，假设输入维度为128
# r可以等于 16、32、64、128

# H:d*N
# U:d*k
# V:k*N
# M:k*r
# Z:k*d
# B:r*N
# H = UV ZH=V MB=V
# M和Z是方阵 需满足k=r=d

# loss系数
alpha = 0.5  # 三因子分解的系数
gamma = 0.001  # 正则项系数
miu = 1 # 子空间二因子系数
lambda_ = 100  # 类似谱聚类项

# phi = 0.1
# # 更新B所需参数
# rho1 = 1.0
# rho2 = 1.0
