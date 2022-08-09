import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from 蟒蛇火炬_预训练.优化器 import 模型的自适应动量估计


def 训练(配置, 模型, 训练用迭代器, 验证用迭代器, 测试用迭代器):
    开始时间 = time.time()
    模型.train()
    参数_优化器 = list(模型.named_parameters())
    不_衰减 = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    优化器_分组_参数列表 = [
        {'params': [p for n, p in 参数_优化器 if not any(nd in n for nd in 不_衰减)], 'weight_decay': 0.01},
        {'params': [p for n, p in 参数_优化器 if any(nd in n for nd in 不_衰减)], 'weight_decay': 0.0},
    ]
    优化器 = 模型的自适应动量估计(优化器_分组_参数列表, 学习率=配置.学习率, 预热=0.05, 训练_总数=len(训练用迭代器) * 配置.轮回数)

    总计批次 = 0
    验证最佳损失值 = float('inf')
    最后_改善 = 0
    旗帜 = 0
    模型.train()
    for 轮回 in range(配置.轮回数):
        print('轮回 [{}/{}]'.format(轮回 + 1, 配置.轮回数))
        for i, (训练, 标签) in enumerate(训练用迭代器):
            输出 = 模型(训练)
            模型.zero_grad()
            损失值 = F.cross_entropy(输出, 标签)
            损失值.backward()
            优化器.step()
            if 总计批次 % 100 == 0:
                真 = 标签.data.cpu()
                预测 = torch.max(输出.data, 1)[1].cpu()
                训练_准确率 = metrics.accuracy_score(真, 预测)
                验证_准确率, 验证_损失值 = 评估(配置, 模型, 验证用迭代器)
                if......


def 评估(配置, 模型, 数据_迭代器, 测试=False):
    模型.eval()
    总损失值 = 0
    所有预测 = np.array([], dtype=int)
    所有标签 = np.array([], dtype=int)
    with torch.no_grad():
        for 文本, 标签 in 数据_迭代器:
            输出 = 模型(文本)
            损失值 = F.cross_entropy(输出, 标签)
            总损失值 += 损失值
            标签 = 标签.data.cpu().numpy()
            预测 = torch.max(输出.data, 1)[1].cpu().numpy()
            所有标签 = np.append(所有标签, 标签)
            所有预测 = np.append(所有预测, 预测)

    准确率 = metrics.accuracy_score(所有标签, 所有预测)
    if 测试:
        报告 = metrics.classification_report(所有标签, 所有预测, target_names=配置.类别名单, digits=4)
        混淆 = metrics.confusion_matrix(所有标签, 所有预测)
        return 准确率, 损失值 / len(数据_迭代器), 报告, 混淆
    return 准确率, 损失值 / len(数据_迭代器)
