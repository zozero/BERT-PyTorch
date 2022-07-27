import argparse
import time
from importlib import import_module

import numpy as np
import torch

from 多个工具 import 构建数据集, 构建迭代器

解析 = argparse.ArgumentParser(description='中文文本分类')
解析.add_argument('--模型', type=str, required=True, help='选择一个模型：形变双向编码器表示法库，文心大模型类')
复数参数 = 解析.parse_args()

if __name__ == '__main__':
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

    训练用迭代器 = 构建迭代器(训练用数据, 配置)
