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


def OMGH(dataloader, max_iters=1000):
    X1_acc = []
    B_acc = []
    X2_acc = []
    label_acc = []

    E1_t = torch.zeros(d1, r).float().to(device)
    E2_t = torch.zeros(d2, r).float().to(device)
    H_t = torch.zeros(r, r).float().to(device)

    E3_t = torch.zeros(d1, d1).float().to(device)
    E4_t = torch.zeros(d2, d2).float().to(device)

    for img, txt, label in dataloader:
        X1_t = img.T.to(device)
        X2_t = txt.T.to(device)
        B_t = torch.randint(0, 2, (r, BATCH_SIZE)).float().to(device)
        B_t = torch.where(B_t == 0, torch.tensor(-1.0), torch.tensor(1.0)).to(device)
        print("B_t", torch.linalg.norm(B_t, 'fro') ** 2)

        Sxx = su_simlarity(label).to(device)

        # Sqx =
        # 从旧数据中随机选择一部分
        if B_acc:
            print(len(X1_acc))
            indices = torch.randperm(len(X1_acc))[:int(Nq * Nq / (Nq + Nt))]
            print("indices", indices.shape)
            selected_old_X1 = torch.stack([X1_acc[i] for i in indices]).squeeze(0)  # 64,4096

            selected_old_X2 = torch.stack([X2_acc[i] for i in indices]).squeeze(0)  # 64,300

            if B_acc:
                selected_old_B = torch.stack([B_acc[i] for i in indices])
            else:
                selected_old_B = B_t[:, indices]
            print("old_B",selected_old_B.shape)  # 16,1

            # 从新数据中随机选择一批锚点
            indices = torch.randperm(img.size(0))[:int(Nq * Nt / (Nq + Nt))]
            # selected_new_X1 = img[indices]
            # selected_new_X2 = txt[indices]
            selected_new_B = B_t[:, indices]
            print("new_B",selected_new_B.shape)
            selected_new_l = label[indices].to(device)
            print("new_l",selected_new_l.shape)

        for i in range(max_iters):

            # 更新B
            epsilon = 1e-10
            B_t = B_t + 1
            # + lambda_ * Bq @ Sqx)

        # 先加进去试试 否则刚开始没有Q
        X1_acc.append(img)
        X2_acc.append(txt)
        label_acc.append(label)




if __name__ == "__main__":
    """
        初始化矩阵
    """
    print('...Data loading is beginning...')
    data = load_pascal(load_i_dir, load_t_dir, load_l_dir)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print('...Data loading is completed...')

    OMGH(dataloader)




