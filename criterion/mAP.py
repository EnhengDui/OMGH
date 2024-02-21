import numpy as np
import scipy.spatial

def calc_mAP(image, text, label, k=0):
    # dist[i, j]表示第i个图像样本与第j个文本样本之间的余弦距离
    image = image.T
    text = text.T
    # label = label.T # n*c
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
    # 排序
    ord = dist.argsort()
    num_imag = dist.shape[0]
    # top-k前k个，默认为所有的
    if k == 0:
        k = num_imag
    res = []
    for i in range(num_imag):
        order = ord[i]
        precision_sum = 0.0  # 用于保存累积精确率的和
        num_relevant = 0.0  # 用于记录与查询图像相关的文本描述数量
        for j in range(k):
            if label[i] == label[order[j]]:
                num_relevant += 1
                precision_sum += (num_relevant / (j + 1))
        if num_relevant > 0:
            res.append(precision_sum / num_relevant)  # 计算Average Precision，AP
        else:
            res.append(0)  # 否则AP为0
    return np.mean(res)

# 测试数据
# image = np.array([[1, 2, 3]])
# text = np.array([[1, 2, 4], [2, 2, 3]])
# label = np.array([1, 2])
# k = 1
# mAP = calc_mAP(image, text, label, k)
# print("mAP:", mAP)
