import argparse
import time
from importlib import import_module

import numpy as np
import torch

from 多个工具 import 构建数据集, 构建迭代器, 获得时间偏移量
from 蟒蛇火炬_预训练.模型类 import 形变双向编码器表示法的模型
from 训练和评估 import 训练

解析 = argparse.ArgumentParser(description='中文文本分类')
解析.add_argument('--模型', type=str, required=True, help='选择一个模型：形变双向编码器表示法库，文心大模型类')
复数参数 = 解析.parse_args()


def 缩减数据集(读取行数=512):
    """
    减少数据量，用于测试代码运行过程是否存在问题
    :return:
    """
    文件名列表 = ["测试集.txt", "训练集.txt", "验证集.txt"]
    for 文件名 in 文件名列表:
        行列表 = []
        with open("清华中文文本分类工具包/数据集/临时/" + 文件名, 'r', encoding="utf-8") as 文件:
            for i in range(读取行数):
                行列表.append(文件.readline())
            文件.close()
        with open("清华中文文本分类工具包/数据集/" + 文件名, 'w', encoding="utf-8") as 文件:
            文件.writelines(行列表)
            文件.close()

    exit()


def 修改预训练的模型():
    """
    路径_字典 = torch.load(路径.模型目录())
        状态_字典 = self.state_dict()
        for 名称 in 路径_字典:
            if 名称 not in 对应名称:
                continue
            状态_字典[对应名称[名称]] = 路径_字典
        self.load_state_dict(状态_字典)
    :return:
    """
    修改后的模型 = None
    列表 = []
    预训练的模型 = torch.load("形变双向编码器表示法_预训练模型/火炬_模型.bin")

    for i in 预训练的模型:
        print(i)

    # print(预训练的模型)
    exit()


def 载入自定义的模型():
    数据集 = '清华中文文本分类工具包'
    模型名 = 复数参数.模型
    x = import_module('模型.' + 模型名)
    配置 = x.配置(数据集)
    模型 = 形变双向编码器表示法的模型(配置)
    print(模型.state_dict())
    exit()


if __name__ == '__main__':
    # 缩减数据集()
    # 载入自定义的模型()
    # 修改预训练的模型()
    数据集 = '清华中文文本分类工具包'

    模型名 = 复数参数.模型
    x = import_module('模型.' + 模型名)
    配置 = x.配置(数据集)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    开始时间 = time.time()
    print("正在载入数据......")
    训练用数据, 验证用数据, 测试用数据 = 构建数据集(配置)
    # 验证用数据 = 构建数据集(配置)

    训练用迭代器 = 构建迭代器(训练用数据, 配置)
    验证用迭代器 = 构建迭代器(验证用数据, 配置)
    测试用迭代器 = 构建迭代器(测试用数据, 配置)
    # next(训练用迭代器)
    时间偏移量 = 获得时间偏移量(开始时间)
    print("花费的时间：", 时间偏移量)

    模型 = x.模型(配置).to(配置.设备)  # 形变双向编码器表示法库文件的模型类
    训练(配置, 模型, 训练用迭代器, 验证用迭代器, 测试用迭代器)
