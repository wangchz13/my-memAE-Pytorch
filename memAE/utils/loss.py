from __future__ import absolute_import, print_function

import torch
from torch import nn
from torch.nn import functional as F

def calculate_cosine_similarity_sum(vectors):
    n = len(vectors)
    m = vectors[0].size(0)  # 假设所有向量具有相同的维度
    
    # 确保输入的维度是一致的
    assert all(vec.size(0) == m for vec in vectors), "All vectors must have the same dimensionality."
    
    # 初始化余弦相似度和为零
    cosine_similarity_sum = torch.tensor(0.0)
    
    # 计算每对向量之间的余弦相似度并求和
    for i in range(n):
        for j in range(i+1, n):  # 避免重复计算对称位置
            cosine_similarity = F.cosine_similarity(vectors[i], vectors[j], dim=0)
            cosine_similarity_sum += cosine_similarity
    
    return cosine_similarity_sum  # 转为标量

class MemAELoss(nn.Module):
    def __init__(self, regularization_parameter=0.0002):
        super(MemAELoss, self).__init__()
        self.reg_param = regularization_parameter

    def forward(self, prediction, ground_truth, training=False, testing=False, validating=False):
        attention_weights = prediction['att']
        loss = None
        if training:
            loss = F.mse_loss(input=ground_truth, target=prediction['output'])
            regularizer =  F.softmax(attention_weights, dim=1) * F.log_softmax(attention_weights, dim=1)
            loss += (-1.0 * self.reg_param * regularizer.sum())
            loss += calculate_cosine_similarity_sum(prediction['mem'])
        if validating:
            loss = F.mse_loss(input=ground_truth, target=prediction['output'])
        if testing:
            loss = F.mse_loss(input=ground_truth, target=prediction['output'], reduction='none')
        return loss