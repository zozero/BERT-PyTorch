import copy
import json
import logging
import os.path
import shutil
import tarfile
import tempfile

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
形变双向编码器表示法配置文件名 = 'bert_config.json'
张量洪流_权重文件名 = 'model.ckpt'


def 在形变双向编码器表示法模型中载入张量洪流权重(模型, 张量洪瑞_检查点路径):
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("需要在蟒蛇火炬项目中载入张量洪流，张量洪流需要安装，请查看 https://www.tensorflow.org/install/ 的安装介绍")
        raise

    张量洪流_路径 = os.path.abspath(张量洪瑞_检查点路径)
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
                指针 = getattr(指针, '权重')
            elif l[0] == '输出_偏置项' or l[0] == '贝塔':
                指针 = getattr(指针, '偏置项')
            elif l[0] == '输出_权重':
                指针 = getattr(指针, '权重')
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
        if 模型名[-11:] == '_嵌入层':
            指针 = getattr(指针, '权重')
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


class 形变双向编码器表示法配置:
    """
    形变双向编码器表示法模型的配置信息收录
    """

    def __init__(self,
                 词汇数量或者简谱配置文件,
                 隐藏层层数=768,
                 隐藏层个数=12,
                 关注的头数=12,
                 中间的数量=3072,
                 隐藏层的动作='gelu',
                 隐藏层失活率=0.1,
                 关注概率的失活率=0.1,
                 最大_位置_嵌入层=512,
                 词汇类型数量=2,
                 初始化_范围=0.02
                 ):
        """

        :param 词汇数量或者简谱配置文件: 简谱（json 爪哇脚本对象简谱）
        :param 隐藏层层数:
        :param 隐藏层个数:
        :param 关注的头数: 关注层的特征图数量，
        :param 中间的数量:
        :param 隐藏层的动作:
        :param 隐藏层失活率:
        :param 关注概率的失活率:
        :param 最大_位置_嵌入层:
        :param 词汇类型数量:
        :param 初始化_范围:
        """
        if isinstance(词汇数量或者简谱配置文件, int):
            self.词汇数量 = 词汇数量或者简谱配置文件
            self.隐藏层层数 = 隐藏层层数
            self.隐藏层个数 = 隐藏层个数
            self.关注的头数 = 关注的头数
            self.中间的数量 = 中间的数量
            self.隐藏层的动作 = 隐藏层的动作
            self.隐藏层失活率 = 隐藏层失活率
            self.关注概率的失活率 = 关注概率的失活率
            self.最大_位置_嵌入层 = 最大_位置_嵌入层
            self.词汇类型数量 = 词汇类型数量
            self.初始化_范围 = 初始化_范围
        else:
            raise ValueError("首先参数必须是词汇数量（int）或者预训练模型配置文件的路径（str）。")

    @classmethod
    def 从字典获取配置(cls, 简谱对象):
        配置 = 形变双向编码器表示法配置(词汇数量或者简谱配置文件=-1)
        for 键, 值 in 简谱对象.items():
            配置.__dict__[键] = 值
        return 配置

    @classmethod
    def 从简谱获取配置(cls, 简谱文件):
        with open(简谱文件, 'r', encoding='utf-8') as 读者:
            文本 = 读者.read()
        return cls.从字典获取配置(json.loads(文本))

    def __repr__(self):
        return str(self.转成简谱字符串)

    def 转成字典(self):
        输出 = copy.deepcopy(self.__dict__)
        return 输出

    def 转成简谱字符串(self):
        return json.dumps(self.转成字典, indent=2, sort_keys=True) + '\n'

    def 转成简谱文件(self, 简谱文件路径):
        with open(简谱文件路径, 'w', encoding='utf-8') as 作者:
            作者.write(self.转成简谱字符串())


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as 形变双向编码器表示法模型归一化层
except ImportError:
    记录器.info("apex能够实现更快的速度。可以从 https://www.github.com/nvidia/apex 安装")


    class 形变双向编码器表示法模型归一化层(nn.Module):
        def __init__(self, 隐藏层层数, 艾普西龙=1e-12):
            super(形变双向编码器表示法模型归一化层, self).__init__()
            self.权重 = nn.Parameter(torch.ones(隐藏层层数))
            self.偏置项 = nn.Parameter(torch.zeros(隐藏层层数))
            self.艾普西龙方差 = 艾普西龙

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            X = (x - u) / torch.sqrt(s + self.艾普西龙方差)
            return self.权重 * x + self.偏置项


class 形变双向编码器表示法模型的预训练模型(nn.Module):
    """
    用于初始权重的抽象类和下载并加载预训练模型的简单接口
    """

    def __init__(self, 配置, *输入列表, **参数字典):
        super(形变双向编码器表示法模型的预训练模型, self).__init__()
        if not isinstance(配置, 形变双向编码器表示法配置):
            raise ValueError(
                "配置文件（{}）参数应该是一个”形变双向编码器表示法配置“的类。"
                "要创建谷歌预训练模型，请使用”模型={}.从预训练开始(预训练模型名)“"
                .format(self.__class__.__name__, self.__class__.__name__))
        self.配置 = 配置

    def 初始化形变双向编码器表示法权重(self, 模块):
        if isinstance(模块, (nn.Linear, nn.Embedding)):
            模块.weight.data.normal_(mean=0.0, std=self.配置.初始化_范围)
        elif isinstance(模块, 形变双向编码器表示法模型归一化层):
            模块.偏置项.data.zero_()
            模块.权重.data.fill_(1.0)
        if isinstance(模块, nn.Linear) and 模块.bias is not None:
            模块.bias.data.zero_()

    @classmethod
    def 从预训练开始(cls, 预训练模型名或路径, *输入列表, **参数字典):
        """
        实例化形变双向编码器表示法模型的预训练模型
        :param 预训练模型名或路径:
        :param 输入列表:
        :param 参数字典:
        :return:
        """

        状态字典 = 参数字典.get('状态字典', None)
        参数字典.pop('状态字典', None)
        缓存目录 = 参数字典.get('缓存目录', None)
        参数字典.pop('缓存目录', None)
        来自_张量洪流 = 参数字典.get('来自_张量洪流', None)
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
            配置文件 = os.path.join(序列化_目录, 形变双向编码器表示法配置文件名)
        配置 = 形变双向编码器表示法配置.从简谱获取配置(配置文件)
        记录器.info("模型配置{}".format(配置文件))
        模型 = cls(配置, *输入列表, **参数字典)
        if 状态字典 is None and not 来自_张量洪流:
            权重路径 = os.path.join(序列化_目录, 权重文件名)
            状态字典 = torch.load(权重路径, map_location='cpu')
        if 临时目录:
            shutil.rmtree(临时目录)
        if 来自_张量洪流:
            权重路径 = os.path.join(序列化_目录, 张量洪流_权重文件名)
            return 在形变双向编码器表示法模型中载入张量洪流权重(模型, 权重路径)
        # 从蟒蛇火炬状态字典载入
        旧键值列表 = []
        新键值列表 = []
        for 键值 in 状态字典.keys():
            新键值 = None
            if '伽马' in 键值:
                新键值 = 键值.replace('伽马', '权重')
            if '贝塔' in 键值:
                新键值 = 键值.replace('贝塔', '偏置项')
            if 新键值:
                旧键值列表.append(键值)
                新键值列表.append(新键值)
        for 旧键值, 新键值 in zip(旧键值列表, 新键值列表):
            状态字典[新键值] = 状态字典.pop(旧键值)

        失踪的键值列表 = []
        未预料的键值列表 = []
        错误消息列表 = []
        # 拷贝状态字典，以便通过_load_from_state_dict调整它
        元数据 = getattr(状态字典, '_元数据', None)
        状态字典 = 状态字典.copy()
        if 元数据 is not None:
            状态字典._元数据 = 元数据

        def 载入(模块, 前缀=''):
            局部元数据 = {} if 元数据 is None else 元数据.get(前缀[:-1], {})
            模块._load_from_state_dict(状态字典, 前缀, 局部元数据, True, 失踪的键值列表, 未预料的键值列表, 错误消息列表)
            for 名称, 子级 in 模块._modules.items():
                if 子级 is not None:
                    载入(子级, 前缀 + 名称 + '.')

        起始前缀 = ''
        if not hasattr(模型, '形变双向编码器表示法模型') and any(s.starwitch('形变双向编码器表示法模型.') for s in 状态字典.keys()):
            起始前缀 = '形变双向编码器表示法模型.'
        载入(模型, 前缀=起始前缀)
        if len(失踪的键值列表) > 0:
            记录器.info("从预训练模型{}不能初始化{}的权重".format(失踪的键值列表, 模型.__class__.__name__))
        if len(未预料的键值列表) > 0:
            记录器.info("从预训练模型的权重不能用于{}：{}".format(模型.__class__.__name__, 未预料的键值列表))
        if len(错误消息列表) > 0:
            raise RuntimeError("{}:\n\t{}的状态字典载入错误".format(模型.__class__.__name__, '\n\t'.join(错误消息列表)))
        return 模型


class 形变双向编码器表示法模型(形变双向编码器表示法模型的预训练模型):
    def __init__(self,配置):
        super(形变双向编码器表示法模型, self).__init__(配置)
        self.....
