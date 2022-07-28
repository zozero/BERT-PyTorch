import torch
from torch import nn

from 蟒蛇火炬_预训练 import 形变双向编码器表示法的分词器


class 配置:
    def __init__(self, 数据集):
        self.模型名 = '形变双向编码器表示法'
        self.训练_路径 = 数据集 + '/数据集/训练集.txt'
        self.验证_路径 = 数据集 + '/数据集/验证集.txt'
        self.测试_路径 = 数据集 + '/数据集/测试集.txt'
        self.类别名单 = [类别.strip() for 类别 in open(数据集 + '/数据集/类别名单.txt').readlines()]
        self.保存路径 = 数据集 + '/保存的字典/' + self.模型名 + '.ckpt'
        self.设备 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.无效改善阈值 = 1000
        self.类别数 = len(self.类别名单)
        self.轮回数 = 3
        self.每批数量 = 128
        self.句子长度 = 32
        self.学习率 = 5e-5
        self.形变双向编码器表示法_路径 = './形变双向编码器表示法_预训练模型'
        self.分词器 = 形变双向编码器表示法的分词器.从预训练开始(self.形变双向编码器表示法_路径)
        self.单隐层的长度 = 768


class 模型(nn.Module):
    def __init__(self,配置):
        super(模型, self).__init__()
        self.形变双向编码器表示法=