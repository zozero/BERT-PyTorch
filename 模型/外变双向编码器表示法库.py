import torch
from torch import nn

from 蟒蛇火炬_预训练 import 外变双向编码器表示法的分词器
from 蟒蛇火炬_预训练.模型类 import 外变双向编码器表示法的模型


class 配置:
    def __init__(self, 数据集):
        self.模型名 = '外变双向编码器表示法'
        # self.训练_路径 = 数据集 + '/数据集/训练集.txt'
        # self.验证_路径 = 数据集 + '/数据集/验证集.txt'
        # self.测试_路径 = 数据集 + '/数据集/测试集.txt'
        self.训练_路径 = 数据集 + '/数据集/临时 - 副本/训练集.txt'
        self.验证_路径 = 数据集 + '/数据集/临时 - 副本/验证集.txt'
        self.测试_路径 = 数据集 + '/数据集/临时 - 副本/测试集.txt'
        self.类别名单 = [类别.strip() for 类别 in open(数据集 + '/数据集/类别名单.txt').readlines()]
        self.保存路径 = 数据集 + '/保存的字典/' + self.模型名 + '.ckpt'
        self.设备 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.无效改善阈值 = 1000
        self.类别数 = len(self.类别名单)
        self.轮回数 = 3
        self.每批数量 = 128
        self.句子长度 = 32
        self.学习率 = 5e-5
        self.外变双向编码器表示法_路径 = './预训练的模型'
        self.分词器 = 外变双向编码器表示法的分词器.从预训练开始(self.外变双向编码器表示法_路径)
        self.隐藏层大小 = 768


class 模型(nn.Module):
    def __init__(self, 配置):
        super(模型, self).__init__()
        self.表示法模型 = 外变双向编码器表示法的模型.从预训练开始(配置.外变双向编码器表示法_路径)
        for 参数 in self.表示法模型.parameters():
            参数.requires_grad = True
        self.全连接层 = nn.Linear(配置.隐藏层大小, 配置.类别数)

    def forward(self, 输入):
        句子列表 = 输入[0]
        句子掩码列表 = 输入[2]
        _, 已池化 = self.表示法模型(句子列表, 句子掩码列表=句子掩码列表, 是否输出全部已编码的层=False)
        输出 = self.全连接层(已池化)
        return 输出
