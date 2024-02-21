import numpy as np
import torch

# 生成每个batch的数据的函数
# features:r*N numpy array N是图像和文本的总数
# labels: c*N
def generate_batches(features, similarity_matrix, batch_size):
    if features.shape[1] % 2 != 0:
        raise ValueError("features的个数即列数应该是偶数")
    n = int(features.shape[1]//2)

    # if features.shape[1] != labels.shape[1]:
    #     raise ValueError("features的个数应该等于labels的个数")

    for i in range(0, n, batch_size):
        # 如果剩余数据不足以填满一个完整的batch，则跳过这个batch
        if i + batch_size > n:
            continue
        batch_I_features = features[:, i:i+batch_size]  # (d, batch_size)
        batch_T_features = features[:, i+n:i+n+batch_size]
        # batch_labels = labels[:, i:i+batch_size]
        batch_similarity = similarity_matrix[i:i+batch_size, i:i+batch_size]
        yield batch_I_features, batch_T_features, batch_similarity
        # yield关键字表示这是一个生成器函数 生成器的作用是在迭代过程中不断返回数据

def generate_batches_torch(features, similarity_matrix, batch_size):
    # tensor.size(1)返回张量的列数
    if features.size(1) % 2 != 0:
        raise ValueError("features的列数应该是偶数")
    n = features.size(1) // 2

    for i in range(0, n, batch_size):
        if i + batch_size > n:
            continue
        batch_I_features = features[:, i:i+batch_size]  # (d, batch_size)
        batch_T_features = features[:, i+n:i+n+batch_size]
        batch_similarity = similarity_matrix[i:i+batch_size, i:i+batch_size]
        yield batch_I_features, batch_T_features, batch_similarity

# 把相似度图分成四个子图，对应数据为四个子图的特征
def generate_subbatches(features, similarity_matrix):
    # tensor.size(1)返回张量的列数
    if features.size(1) % 2 != 0:
        raise ValueError("features的列数应该是偶数")
    n = features.size(1) // 2  # n是图像或文本的个数
    batch_size = n//2
    for i in range(0, n, batch_size):
        if i + batch_size > n:
            continue
        batch_I_features = features[:, i:i+batch_size]  # (d, batch_size)
        for j in range(0, n, batch_size):
            batch_T_features = features[:, j+n:j+n+batch_size]
            batch_similarity = similarity_matrix[i:i+batch_size, j:j+batch_size]
            yield batch_I_features, batch_T_features, batch_similarity

# 使用
# 在训练过程中使用生成的每个batch的数据
# for batch_features, batch_similarity in generate_batches(features, similarity_matrix, batch_size):
#     # 在这里使用batch_features和batch_similarity进行训练
#     print("Batch Features Shape:", batch_features.shape)
#     print("Batch Similarity Matrix Shape:", batch_similarity.shape)
#     print("--------------------")

def generate_test_batches(img, txt, labels, batch_size):
    labels = labels.T
    if img.shape[1] != txt.shape[1]:
        raise ValueError("图像文本个数应该匹配")
    if img.shape[1] != labels.shape[1]:
        raise ValueError("图像标签个数应该匹配")

    n = img.shape[1]
    for i in range(0, n, batch_size):
        # 如果剩余数据不足以填满一个完整的batch，则跳过这个batch
        if i + batch_size > n:
            continue
        batch_I_features = img[:, i:i+batch_size]  # (d, batch_size)
        batch_T_features = txt[:, i+n:i+n+batch_size]
        batch_labels = labels[:, i:i+batch_size]

        yield batch_I_features, batch_T_features, batch_labels


import os
import scipy.io as sio
data_path = "../data/PASCAL/"



# print(img.shape)  # (800,4096)
# print(txt.shape)  # (800, 300)
# print(labels.shape)  # (800, 1)

from torch.utils.data import DataLoader, Dataset

def to_one_hot(labels, num_classes):

    one_hot_labels = torch.zeros(len(labels), num_classes)
    # one_hot_labels[range(len(labels)), labels] = 1  # 处理大规模数据时，没有scatter高效
    # dim index src dim是维度 1表示行，index是索引，src是要填充的值 即在一行的哪个位置填1
    one_hot_labels.scatter_(1, labels.long(), 1)
    return one_hot_labels

class load_pascal(Dataset):
    def __init__(self, load_i_dir, load_t_dir, load_l_dir):
        img_dict = sio.loadmat(load_i_dir)
        txt_dict = sio.loadmat(load_t_dir)
        labels_dict = sio.loadmat(load_l_dir)
        self.img = img_dict["train_img"]
        self.txt = txt_dict["train_txt"]
        self.labels = labels_dict["train_img_lab"]
        self.num_classes = max(self.labels.flatten())+1
    def __getitem__(self, index):
        img = self.img[index]
        txt = self.txt[index]
        label = self.labels[index]

        image = torch.tensor(img, dtype=torch.float32)
        text = torch.tensor(txt, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        # print(label.view(-1,1).shape)  # [1] ---> [1,1]
        label = to_one_hot(label.view(-1,1),self.num_classes)
        label = label.squeeze()
        return image, text, label
    def __len__(self):
        return len(self.labels)

load_i_dir = os.path.join(data_path, "train_img.mat")
load_t_dir = os.path.join(data_path,"train_txt.mat")
load_l_dir = os.path.join(data_path, "train_img_lab.mat")

# data = load_pascal(load_i_dir, load_t_dir, load_l_dir)
# img,txt,lab = data[0]
# print(img.shape)  # 4096
# print(txt.shape)  # 300
# print(lab.shape)  # 19

# dataloader = DataLoader(data, batch_size=64, shuffle=True)
# for img, txt, label in dataloader:
#     print(img.shape, txt.shape, label.shape)



