import copy
import json
import logging
import math
import os.path
import shutil
import sys
import tarfile
import tempfile

import numpy as np
import torch
from torch import nn

from .文件_多个工具 import 形成缓存路径, 配置文件名, 权重文件名

记录器 = logging.getLogger(__name__)
预训练词汇档案映射 = {
    '基础-不分大小写': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    '大量-不分大小写': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    '基础-分大小写': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    '大量-分大小写': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    '基础-多语言的-不分大小写': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    '基础-多语言的-分大小写': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    '基础-汉语': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt"
}
模型的配置文件名 = '配置.json'
张量洪流_权重文件名 = '模型.ckpt'


def 在模型中载入张量洪流权重(模型, 张量洪流_检查点路径):
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("需要在蟒蛇火炬项目中载入张量洪流，张量洪流需要安装，请查看 https://www.tensorflow.org/install/ 的安装介绍")
        raise

    张量洪流_路径 = os.path.abspath(张量洪流_检查点路径)
    print("正在从{}中载入张量洪流载入点".format(张量洪流_路径))
    初始变量 = tf.train.list_variables(张量洪流_路径)
    名称列表 = []
    数组列表 = []
    for 名称, 形状 in 初始变量:
        print("正在载入形状是{}的张量洪流权重{}".format(形状, 名称))
        数组 = tf.train.load_variable(张量洪流_路径, 名称)
        名称列表.append(名称)
        数组列表.append(数组)

    for 名称, 数组 in zip(名称列表, 数组列表):
        名称 = 名称.split('/')
        if any(n in ["adam_v", "adam_m", "global_step"] for n in 名称):
            print("正在跳过{}".format('/'.join(名称)))
            continue
        指针 = 模型
        for 模型名 in 名称:
            if re.fullmatch(r'[A-Za-z]+_\d+', 模型):
                l = re.split(r'_(\d+)', 模型名)
            else:
                l = [模型名]
            if l[0] == '核心' or l[0] == '伽马':
                指针 = getattr(指针, 'weight')
            elif l[0] == '输出_偏置项' or l[0] == '贝塔':
                指针 = getattr(指针, 'bias')
            elif l[0] == '输出_权重':
                指针 = getattr(指针, 'weight')
            elif l[0] == '斯坦福问答回答数据集':  # squad
                指针 = getattr(指针, '分类器')
            else:
                try:
                    指针 = getattr(指针, l[0])
                except AttributeError:
                    print("正在跳过{}".format('/'.join(名称)))
                    continue
            if len(l) >= 2:
                数字 = int(l[1])
                指针 = 指针[数字]
        if 模型名[-11:] == '_字向量层':
            指针 = getattr(指针, 'weight')
        elif 模型名 == '核心':
            数组 = np.transpose(数组)
        try:
            assert 指针.shape == 数组.shape
        except AssertionError as e:
            e.args += (指针.shape, 数组.shape)
            raise
        print("已初始化蟒蛇火炬权重{}".format(名称))
        指针.data = torch.from_numpy(数组)
    return 模型


def 高斯误差线性单元(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def 嗖嗖(x):
    """
    英文为swish
    这是一个激活函数
    :return:
    """
    return x * torch.sigmoid(x)


动作转函数 = {"高斯误差线性单元": 高斯误差线性单元,"整流线性单元":torch.nn.functional.relu, "嗖嗖": 嗖嗖}


class 模型的配置:
    """
    模型的模型的配置信息收录
    """

    def __init__(self,
                 词汇数量或者简谱配置文件,
                 隐藏层大小=768,
                 隐藏层个数=12,
                 关注层的头数=12,
                 中间层大小=3072,
                 隐藏层的动作='高斯误差线性单元',
                 隐藏层失活率=0.1,
                 关注概率的失活率=0.1,
                 最大_位置_字向量层=512,
                 词汇类型数量=2,
                 初始化_范围=0.02
                 ):
        """

        :param 词汇数量或者简谱配置文件: 简谱（json 爪哇脚本对象简谱）
        :param 隐藏层大小:
        :param 隐藏层个数:
        :param 关注层的头数: 关注层的特征图数量，
        :param 中间层大小:
        :param 隐藏层的动作:
        :param 隐藏层失活率:
        :param 关注概率的失活率:
        :param 最大_位置_字向量层:
        :param 词汇类型数量:
        :param 初始化_范围:
        """
        if isinstance(词汇数量或者简谱配置文件, str):
            with open(词汇数量或者简谱配置文件, 'r', encoding='utf-8') as 读者:
                简谱_配置 = json.loads(读者.read())
                for 键, 值 in 简谱_配置.items():
                    self.__dict__[键] = 值
        elif isinstance(词汇数量或者简谱配置文件, int):
            self.词汇数量 = 词汇数量或者简谱配置文件
            self.隐藏层大小 = 隐藏层大小
            self.隐藏层个数 = 隐藏层个数
            self.关注层的头数 = 关注层的头数
            self.中间层大小 = 中间层大小
            self.隐藏层的动作 = 隐藏层的动作
            self.隐藏层失活率 = 隐藏层失活率
            self.关注概率的失活率 = 关注概率的失活率
            self.最大_位置_字向量层 = 最大_位置_字向量层
            self.词汇类型数量 = 词汇类型数量
            self.初始化_范围 = 初始化_范围
        else:
            raise ValueError("首先参数必须是词汇数量（int）或者预训练模型配置文件的路径（str）。")

    @classmethod
    def 从字典获取配置(cls, 简谱对象):
        配置 = 模型的配置(词汇数量或者简谱配置文件=-1)
        for 键, 值 in 简谱对象.items():
            配置.__dict__[键] = 值
        return 配置

    @classmethod
    def 从简谱获取配置(cls, 简谱文件):
        with open(简谱文件, 'r', encoding='utf-8') as 读者:
            文本 = 读者.read()
        return cls.从字典获取配置(json.loads(文本))

    def __repr__(self):
        return str(self.转成简谱字符串())

    def 转成字典(self):
        输出 = copy.deepcopy(self.__dict__)
        return 输出

    def 转成简谱字符串(self):
        return json.dumps(self.转成字典(), indent=2, sort_keys=True) + '\n'

    def 转成简谱文件(self, 简谱文件路径):
        with open(简谱文件路径, 'w', encoding='utf-8') as 作者:
            作者.write(self.转成简谱字符串())


try:
    from apex.normalization.fused_layer_norm import FusedLayerNzorm as 模型的层归一化
except ImportError:
    记录器.info("apex能够实现更快的速度。可以从 https://www.github.com/nvidia/apex 安装")


    class 模型的层归一化(nn.Module):
        def __init__(self, 隐藏层大小, 艾普西龙=1e-12):
            super(模型的层归一化, self).__init__()
            self.weight = nn.Parameter(torch.ones(隐藏层大小))
            self.bias = nn.Parameter(torch.zeros(隐藏层大小))
            self.艾普西龙方差 = 艾普西龙

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.艾普西龙方差)
            return self.weight * x + self.bias


class 模型的字向量层(nn.Module):
    """
    字向量层：使用中文以汉字为单位；embedding是嵌入的意思，本质是一个词向量，描述词的一串数字
    """

    def __init__(self, 配置):
        super(模型的字向量层, self).__init__()
        self.字_字向量层 = nn.Embedding(配置.词汇数量, 配置.隐藏层大小, padding_idx=0)
        self.位置_字向量层 = nn.Embedding(配置.最大_位置_字向量层, 配置.隐藏层大小)
        self.字符_类型_字向量层 = nn.Embedding(配置.词汇类型数量, 配置.隐藏层大小)

        self.层归一化 = 模型的层归一化(配置.隐藏层大小, 艾普西龙=1e-12)
        self.失活率 = nn.Dropout(配置.隐藏层失活率)

    def forward(self, 输入_标记张量, 字符_类型_标记张量=None):
        序列长度 = 输入_标记张量.size(1)
        位置_标记张量 = torch.arange(序列长度, dtype=torch.long, device=输入_标记张量.device)
        位置_标记张量 = 位置_标记张量.unsqueeze(0).expand_as(输入_标记张量)
        if 字符_类型_标记张量 is None:
            字符_类型_标记张量 = torch.zeros_like(输入_标记张量)

        字_字向量层 = self.字_字向量层(输入_标记张量)
        位置_字向量层 = self.位置_字向量层(位置_标记张量)
        字符_类型_字向量层 = self.字符_类型_字向量层(字符_类型_标记张量)

        字向量层 = 字_字向量层 + 位置_字向量层 + 字符_类型_字向量层
        字向量层 = self.层归一化(字向量层)
        字向量层 = self.失活率(字向量层)

        return 字向量层


class 模型的自身关注层(nn.Module):
    """
    关注层的头数：是指某一个字向量通过计算【软最大((查询的*被查的)/根号维度数)*特性信息】后得出的特征，这样的特征的数量，一般是8个
        根号维度数：防止因维度越长导致结果越大
        软最大：softmax函数
        查询的：q
        被查的：k
        特性信息：v
    多头机制：获得多个特征，将所有特征拼接，最后通过全连接层来降维
    """

    def __init__(self, 配置):
        super(模型的自身关注层, self).__init__()
        if 配置.隐藏层大小 % 配置.关注层的头数 != 0:
            raise ValueError("隐藏层的大小（%d）不是关注层的头数的倍数（%d）" % (配置.隐藏层大小, 配置.关注层的头数))
        self.关注层的头数 = 配置.关注层的头数
        self.倍数 = int(配置.隐藏层大小 / 配置.关注层的头数)
        # 如果倍数不是整数就直接报错了，所以这里其实就是隐藏层大小
        self.总头数 = self.关注层的头数 * self.倍数

        self.查询 = nn.Linear(配置.隐藏层大小, self.总头数)
        self.被查 = nn.Linear(配置.隐藏层大小, self.总头数)
        self.特征信息 = nn.Linear(配置.隐藏层大小, self.总头数)

        self.失活率 = nn.Dropout(配置.关注概率的失活率)

    def 改变分数张量的形状(self, x):
        新x的形状 = x.size()[:-1] + (self.关注层的头数, self.倍数)
        x = x.view(*新x的形状)
        # permute：置换
        return x.permute(0, 2, 1, 3)

    def forward(self, 隐藏层状态, 关注层掩码):
        混合_查询_层 = self.查询(隐藏层状态)
        混合_被查_层 = self.被查(隐藏层状态)
        混合_特征信息_层 = self.特征信息(隐藏层状态)

        查询_层 = self.改变分数张量的形状(混合_查询_层)
        被查_层 = self.改变分数张量的形状(混合_被查_层)
        特征信息_层 = self.改变分数张量的形状(混合_特征信息_层)

        # 计算查询和被查之间的点积获得原生的关注层分数
        # matmul：矩阵乘法
        关注层分数 = torch.matmul(查询_层, 被查_层.transpose(-1, -2))
        关注层分数 = 关注层分数 / math.sqrt(self.倍数)
        # 应用注意掩码（为 形变双向编码器表示法的模型 forward() 函数中的所有层预先计算）？？？
        关注层分数 = 关注层分数 + 关注层掩码

        # 标准化关注分数为概率值
        关注层概率 = nn.Softmax(dim=-1)(关注层分数)

        # 这实际上是丢弃了整个字符来处理，这可能看起来有点不寻常，但取自原始的 Transformer 论文。？？？
        关注层概率 = self.失活率(关注层概率)

        语境_层 = torch.matmul(关注层概率, 特征信息_层)
        语境_层 = 语境_层.permute(0, 2, 1, 3).contiguous()
        新的_语境_层_形状 = 语境_层.size()[:-2] + (self.总头数,)
        语境_层 = 语境_层.view(*新的_语境_层_形状)
        return 语境_层


class 模型自身关注层的输出(nn.Module):
    def __init__(self, 配置):
        super(模型自身关注层的输出, self).__init__()
        self.稠密层 = nn.Linear(配置.隐藏层大小, 配置.隐藏层大小)  # 全连接层
        self.层归一化 = 模型的层归一化(配置.隐藏层大小, 艾普西龙=1e-12)
        self.失活率 = nn.Dropout(配置.隐藏层失活率)

    def forward(self, 隐藏层状态, 输入的张量):
        隐藏层状态 = self.稠密层(隐藏层状态)
        隐藏层状态 = self.失活率(隐藏层状态)
        隐藏层状态 = self.层归一化(隐藏层状态 + 输入的张量)

        return 隐藏层状态


class 模型的关注层(nn.Module):
    def __init__(self, 配置):
        super(模型的关注层, self).__init__()
        self.自身 = 模型的自身关注层(配置)
        self.输出 = 模型自身关注层的输出(配置)

    def forward(self, 输入的张量, 关注层掩码):
        自身_输出 = self.自身(输入的张量, 关注层掩码)
        关注_输出 = self.输出(自身_输出, 输入的张量)
        return 关注_输出


class 模型的中间层(nn.Module):
    def __init__(self, 配置):
        super(模型的中间层, self).__init__()
        self.稠密层 = nn.Linear(配置.隐藏层大小, 配置.中间层大小)  # 全连接层
        if isinstance(配置.隐藏层的动作, str):
            self.中间层_动作_函数 = 动作转函数[配置.隐藏层的动作]
        else:
            self.中间层_动作_函数 = 配置.隐藏层的动作

    def forward(self, 隐藏层状态):
        隐藏层状态 = self.稠密层(隐藏层状态)
        隐藏层状态 = self.中间层_动作_函数(隐藏层状态)
        return 隐藏层状态


class 模型的输出层(nn.Module):
    def __init__(self, 配置):
        super(模型的输出层, self).__init__()
        self.稠密层 = nn.Linear(配置.中间层大小, 配置.隐藏层大小)  # 全连接层
        self.层归一化 = 模型的层归一化(配置.隐藏层大小, 艾普西龙=1e-12)
        self.失活率 = nn.Dropout(配置.隐藏层失活率)

    def forward(self, 隐藏层状态, 输入的张量):
        隐藏层状态 = self.稠密层(隐藏层状态)
        隐藏层状态 = self.失活率(隐藏层状态)
        隐藏层状态 = self.层归一化(隐藏层状态 + 输入的张量)

        return 隐藏层状态


class 模型的层(nn.Module):
    def __init__(self, 配置):
        super(模型的层, self).__init__()
        self.关注层 = 模型的关注层(配置)
        self.中间层 = 模型的中间层(配置)
        self.输出层 = 模型的输出层(配置)

    def forward(self, 隐藏层状态, 关注层掩码):
        关注层_输出 = self.关注层(隐藏层状态, 关注层掩码)
        中间层_输出 = self.中间层(关注层_输出)
        层_输出 = self.输出层(中间层_输出, 关注层_输出)
        return 层_输出


class 模型的编码器(nn.Module):
    def __init__(self, 配置):
        super(模型的编码器, self).__init__()
        层 = 模型的层(配置)
        self.层 = nn.ModuleList([copy.deepcopy(层) for _ in range(配置.隐藏层个数)])

    def forward(self, 隐藏层状态, 关注层掩码, 是否输出全部已编码的层=True):
        全部编码层 = []
        for 层_模块 in self.层:
            隐藏层状态 = 层_模块(隐藏层状态, 关注层掩码)
            if 是否输出全部已编码的层:
                全部编码层.append(隐藏层状态)
        if not 是否输出全部已编码的层:
            全部编码层.append(隐藏层状态)
        return 全部编码层


class 模型的池化层(nn.Module):
    def __init__(self, 配置):
        super(模型的池化层, self).__init__()
        self.稠密层 = nn.Linear(配置.隐藏层大小, 配置.隐藏层大小)  # 全连接层
        self.激活函数 = nn.Tanh()

    def forward(self, 隐藏层状态):
        第一个_字符_张量 = 隐藏层状态[:, 0]
        已池化的输出 = self.稠密层(第一个_字符_张量)
        已池化的输出 = self.激活函数(已池化的输出)

        return 已池化的输出


class 模型的预训练模型(nn.Module):
    """
    用于初始权重的抽象类和下载并加载预训练模型的简单接口
    """

    def __init__(self, 配置, *输入列表, **参数字典):
        super(模型的预训练模型, self).__init__()
        if not isinstance(配置, 模型的配置):
            raise ValueError(
                "配置文件（{}）参数应该是一个”模型的配置“的类。"
                "要创建谷歌预训练模型，请使用”模型={}.从预训练开始(预训练模型名)“"
                .format(self.__class__.__name__, self.__class__.__name__))
        self.配置 = 配置

    def 初始化模型的权重(self, 模块):
        if isinstance(模块, (nn.Linear, nn.Embedding)):
            模块.weight.data.normal_(mean=0.0, std=self.配置.初始化_范围)
        elif isinstance(模块, 模型的层归一化):
            模块.bias.data.zero_()
            模块.weight.data.fill_(1.0)
        if isinstance(模块, nn.Linear) and 模块.bias is not None:
            模块.bias.data.zero_()

    @classmethod
    def 从预训练开始(cls, 预训练模型名或路径, *输入列表, **参数字典):
        """
        实例化模型的模型的预训练模型
        :param 预训练模型名或路径:
        :param 输入列表:
        :param 参数字典:
        :return:
        """

        状态字典 = 参数字典.get('状态字典', None)
        参数字典.pop('状态字典', None)
        缓存目录 = 参数字典.get('缓存目录', None)
        参数字典.pop('缓存目录', None)
        来自_张量洪流 = 参数字典.get('来自_张量洪流', False)
        参数字典.pop('来自_张量洪流', None)

        if 预训练模型名或路径 in 预训练词汇档案映射:
            档案文件 = 预训练词汇档案映射[预训练模型名或路径]
        else:
            档案文件 = 预训练模型名或路径

        try:
            解析的档案文件 = 形成缓存路径(档案文件, 缓存_目录=缓存目录)
        except EnvironmentError:
            记录器.error("在模型列表中（{}）没有找到模型名‘{}’。"
                      "我们假设路径或网址是‘{}’，但不能找到如何与这个路径或网址相关联的文件"
                      .format(预训练模型名或路径, ','.join(预训练词汇档案映射.keys()), 档案文件))
            return None
        if 解析的档案文件 == 档案文件:
            记录器.info("正在载入档案文件{}".format(档案文件))
        else:
            记录器.info("正在从{}缓存中载入档案文件{}".format(解析的档案文件, 档案文件))
        临时目录 = None
        if os.path.isdir(解析的档案文件) or 来自_张量洪流:
            序列化_目录 = 解析的档案文件
        else:
            临时目录 = tempfile.mkdtemp()
            记录器.info("正在从临时目录{}中提取档案文件{}".format(临时目录, 解析的档案文件))
            with tarfile.open(解析的档案文件, 'r:gz') as 档案:
                档案.extractall(临时目录)
            序列化_目录 = 临时目录

        配置文件 = os.path.join(序列化_目录, 配置文件名)
        if not os.path.exists(配置文件):
            配置文件 = os.path.join(序列化_目录, 模型的配置文件名)
        配置 = 模型的配置.从简谱获取配置(配置文件)
        记录器.info("模型配置{}".format(配置))
        模型 = cls(配置, *输入列表, **参数字典)
        # for i in 模型.state_dict():
        #     print(i)
        if 状态字典 is None and not 来自_张量洪流:
            权重路径 = os.path.join(序列化_目录, 权重文件名)
            状态字典 = torch.load(权重路径, map_location='cpu')
        if 临时目录:
            shutil.rmtree(临时目录)
        if 来自_张量洪流:
            # 测试时可能没有用到该函数，所以里面的代码暂时未匹配模型
            权重路径 = os.path.join(序列化_目录, 张量洪流_权重文件名)
            return 在模型中载入张量洪流权重(模型, 权重路径)
        # 从蟒蛇火炬状态字典载入
        旧键值列表 = []
        新键值列表 = []

        # 可以通过这更换相应的字符，但好多阿
        for 键值 in 状态字典.keys():
            新键值 = 键值
            if 'gamma' in 键值:
                新键值 = 键值.replace('gamma', 'weight')
            if 'beta' in 键值:
                新键值 = 键值.replace('beta', 'bias')

            # 存在问题。。。。。。。，需要修改部分偏置项和权重
            # if 'bert' in 新键值:
            #     新键值 = 新键值.replace('bert', '表示法模型')
            # if 'embeddings' in 新键值:
            #     新键值 = 新键值.replace('embeddings', '字向量层')
            # if 'word' in 新键值:
            #     新键值 = 新键值.replace('word', '字')
            # if 'position' in 新键值:
            #     新键值 = 新键值.replace('position', '位置')
            # if 'token_type' in 新键值:
            #     新键值 = 新键值.replace('token_type', '字符_类型')
            # # 为了防止提前替换LayerNorm，得把它提到Layer之前
            # if 'LayerNorm' in 新键值:
            #     新键值 = 新键值.replace('LayerNorm', '层归一化')
            # if 'Layer' in 新键值:
            #     新键值 = 新键值.replace('Layer', '层')
            # if 'encoder' in 新键值:
            #     新键值 = 新键值.replace('encoder', '编码器')
            # if 'attention' in 新键值:
            #     新键值 = 新键值.replace('attention', '关注层')
            # if 'self' in 新键值:
            #     新键值 = 新键值.replace('self', '自身')
            # if 'query' in 新键值:
            #     新键值 = 新键值.replace('query', '查询')
            # if 'key' in 新键值:
            #     新键值 = 新键值.replace('key', '被查')
            # if 'value' in 新键值:
            #     新键值 = 新键值.replace('value', '特征信息')
            # if 'output' in 新键值:
            #     新键值 = 新键值.replace('output', '输出层')
            # if 'dense' in 新键值:
            #     新键值 = 新键值.replace('dense', '稠密层')
            # if 'intermediate' in 新键值:
            #     新键值 = 新键值.replace('intermediate', '中间层')
            # if 'pooler' in 新键值:
            #     新键值 = 新键值.replace('pooler', '池化层')
            # 以下暂没用到到
            # if 'cls' in 新键值:
            #     新键值 = 新键值.replace('cls', '类别')
            # if 'predictions' in 新键值:
            #     新键值 = 新键值.replace('predictions', '预测层')
            # if 'transform' in 新键值:
            #     新键值 = 新键值.replace('transform', '外变层')
            # if 'decoder' in 新键值:
            #     新键值 = 新键值.replace('decoder', '解码器')
            # if 'seq_relationship' in 新键值:
            #     新键值 = 新键值.replace('seq_relationship', '序列_关系层')

            if 新键值:
                旧键值列表.append(键值)
                新键值列表.append(新键值)
        for 旧键值, 新键值 in zip(旧键值列表, 新键值列表):
            状态字典[新键值] = 状态字典.pop(旧键值)

        失踪的键值列表 = []
        未预料的键值列表 = []
        错误消息列表 = []
        # 拷贝状态字典，以便通过_load_from_state_dict调整它
        元数据 = getattr(状态字典, '_metadata', None)
        状态字典 = 状态字典.copy()
        if 元数据 is not None:
            状态字典._metadata = 元数据

        def 载入(模块, 前缀=''):
            局部元数据 = {} if 元数据 is None else 元数据.get(前缀[:-1], {})
            模块._load_from_state_dict(状态字典, 前缀, 局部元数据, True, 失踪的键值列表, 未预料的键值列表, 错误消息列表)
            for 名称, 子级 in 模块._modules.items():
                if 子级 is not None:
                    载入(子级, 前缀 + 名称 + '.')

        起始前缀 = ''
        if not hasattr(模型, '表示法模型') and any(s.startswith('表示法模型.') for s in 状态字典.keys()):
            起始前缀 = '表示法模型.'
        载入(模型, 前缀=起始前缀)
        if len(失踪的键值列表) > 0:
            记录器.info("从预训练模型{}不能初始化{}的权重".format(失踪的键值列表, 模型.__class__.__name__))
        if len(未预料的键值列表) > 0:
            记录器.info("从预训练模型的权重不能用于{}：{}".format(模型.__class__.__name__, 未预料的键值列表))
        if len(错误消息列表) > 0:
            raise RuntimeError("{}:\n\t{}的状态字典载入错误".format(模型.__class__.__name__, '\n\t'.join(错误消息列表)))
        return 模型


class 形变双向编码器表示法的模型(模型的预训练模型):
    def __init__(self, 配置):
        super(形变双向编码器表示法的模型, self).__init__(配置)
        self.字向量层 = 模型的字向量层(配置)
        self.编码器 = 模型的编码器(配置)
        self.池化层 = 模型的池化层(配置)
        self.apply(self.初始化模型的权重)

    def forward(self, 输入的标记, 字符_类别_标记=None, 关注层_掩码=None, 是否输出全部已编码的层=True):
        if 关注层_掩码 is None:
            关注层_掩码 = torch.ones_like(输入的标记)
        if 字符_类别_标记 is None:
            字符_类别_标记 = torch.zeros_like(输入的标记)

        扩展的关注层掩码 = 关注层_掩码.unsqueeze(1).unsqueeze(2)

        扩展的关注层掩码 = 扩展的关注层掩码.to(dtype=next(self.parameters()).dtype)
        扩展的关注层掩码 = (1.0 - 扩展的关注层掩码) * -10000.0

        字向量层输出 = self.字向量层(输入的标记, 字符_类别_标记)
        # print(np.array(字向量层输出.data.cpu().numpy()).shape)
        已编码的层 = self.编码器(字向量层输出, 扩展的关注层掩码, 是否输出全部已编码的层=是否输出全部已编码的层)
        序列化的输出 = 已编码的层[-1]
        # print(np.array(序列化的输出.data.cpu().numpy()).shape)
        已池化的输出 = self.池化层(序列化的输出)
        if not 是否输出全部已编码的层:
            已编码的层 = 已编码的层[-1]
        return 已编码的层, 已池化的输出
