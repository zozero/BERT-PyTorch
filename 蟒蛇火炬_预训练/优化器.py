import abc
import logging
import math
import sys

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_

记录器 = logging.getLogger(__name__)

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta("ABC", (), {})


class _学习率_进度表(ABC):
    警告_训练_总数 = False

    def __init__(self, 预热=0.002, 训练_总数=-1, **参数字典):
        super(_学习率_进度表, self).__init__(**参数字典)
        if 训练_总数 < 0:
            记录器.warning("总数{}的结果在进度表中没有应用".format(训练_总数))
        if not 0.0 <= 预热 < 1.0 and not 预热 == -1:
            raise ValueError("无效的预热：{}，应该取[0.0,1.0]之中".format(预热))
        预热 = max(预热, 0.)
        self.预热, self.训练_总数 = float(预热), float(训练_总数)
        self.进度_训练_总数_已警告 = -1

    def 获得_学习率(self, 步骤, 不预热=False):
        if self.训练_总数 < 0:
            return 1.
        进度 = float(步骤) / self.训练_总数
        返回值 = self.获得_学习率_(进度)
        if not 不预热 and self.警告_训练_总数 and 进度 > 1. and 进度 > self.进度_训练_总数_警告:
            记录器.warning("超过指定总数的训练。学习率的乘数设置为{}。请设置正确的训练_总数{}。".format(返回值, self.__class__.__name__))
            self.进度_训练_总数_已警告 = 进度
        return 返回值

    @abc.abstractmethod
    def 获得_学习率_(self, 进度):
        return 1.


class 常量学习率(_学习率_进度表):
    def 获得_学习率_(self, 进度):
        return 1.


class 预热余弦进度表(_学习率_进度表):
    警告_训练_总数 = True

    def __init__(self, 预热=0.002, 训练_总数=-1, 循环=.5, **参数字典):
        super(预热余弦进度表, self).__init__(预热=预热, 训练_总数=训练_总数, **参数字典)
        self.循环 = 循环

    def 获得_学习率_(self, 进度):
        if 进度 < self.预热:
            return 进度 / self.预热
        else:
            进度 = (进度 - self.预热) / (1 - self.预热)
            return 0.5 * (1. + math.cos(math.pi * self.循环 * 2 * 进度))


class 预热常量进度表(_学习率_进度表):
    def 获得_学习率_(self, 进度):
        if 进度 < self.预热:
            return 进度 / self.预热
        return 1.


class 预热线性进度表(_学习率_进度表):
    警告_训练_总数 = True

    def 获得_学习率_(self, 进度):
        if 进度 < self.预热:
            return 进度 / self.预热
        return max((进度 - 1.) / (self.预热 - 1.), 0.)


进度表字典 = {
    None: 常量学习率,
    "none": 常量学习率,
    "预热_余弦": 预热余弦进度表,
    "预热_常量": 预热常量进度表,
    "预热_线性": 预热线性进度表
}


class 模型的自适应估计矩(Optimizer):
    def __init__(self, 参数列表, 学习率=required, 预热=-1, 训练_总数=-1, 进度表="预热_线性", b1=0.9, b2=0.999, 艾普西龙=1e-6, 权重_衰减=0.01,
                 最大_梯度_范数=1.0, **参数字典):
        if 学习率 is not required and 学习率 < 0.0:
            raise ValueError("没用的学习率：{}，应该>=0.0".format(学习率))
        if not isinstance(进度表, _学习率_进度表) and 进度表 not in 进度表字典:
            raise ValueError("无效的进度表参数：{}".format(进度表))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("无效的b1参数：{}，应该取[0.0,1.0]之中".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("无效的b2参数：{}，应该取[0.0,1.0]之中".format(b2))
        if not 艾普西龙 >= 0.0:
            raise ValueError("无效的艾普西龙参数：{}，应该取>=0.0".format(艾普西龙))
        if not isinstance(进度表, _学习率_进度表):
            进度表_类型 = 进度表字典[进度表]
            进度表 = 进度表_类型(预热=预热, 训练_总数=训练_总数)
        else:
            if 预热 != -1 or 训练_总数 != -1:
                记录器.warning("在_学习率_进度表作为进度表提供时，优化器的预热和训练_总数是无效的。请在_学习率_进度表对象中指定自定义的预热和训练_总数")
        默认字典 = dict(lr=学习率, schedule=进度表, b1=b1, b2=b2, e=艾普西龙, weight_decay=权重_衰减, max_grad_norm=最大_梯度_范数)
        super(模型的自适应估计矩, self).__init__(参数列表, 默认字典)

    def 获得_学习率(self):
        学习率 = []
        for 组 in self.param_groups:
            for p in 组['params']:
                阶段 = self.state[p]
                if len(阶段) == 0:
                    return [0]
                学习率_进度表 = 组['lr']
                学习率_进度表 *= 组['schedule'].get_lr(阶段['step'])
                学习率.append(学习率_进度表)
        return 学习率

    def step(self, closure=None):
        损失值 = None
        if closure is not None:
            损失值 = closure()

        for 组 in self.param_groups:
            for p in 组['params']:
                if p.grad is None:
                    continue
                梯度 = p.grad.data
                if 梯度.is_sparse:
                    raise RuntimeError("自适应动量估计不支持稀疏梯度，请考虑使用（稀疏自适应动量估计）替换")

                阶段 = self.state[p]
                if len(阶段) == 0:
                    阶段['step'] = 0
                    阶段['next_m'] = torch.zeros_like(p.data)
                    阶段['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = 阶段['next_m'], 阶段['next_v']  # 暂时取名
                贝塔1, 贝塔2 = 组['b1'], 组['b2']

                if 组['max_grad_norm'] > 0:
                    clip_grad_norm_(p, 组['max_grad_norm'])

                next_m.mul_(贝塔1).add_(梯度, alpha=1 - 贝塔1)
                next_v.mul_(贝塔2).addcmul_(梯度, 梯度, value=1 - 贝塔2)
                更新 = next_m / (next_v.sqrt() + 组['e'])

                if 组['weight_decay'] > 0.0:
                    更新 += 组['weight_decay'] * p.data

                学习率_进度表 = 组['lr']
                学习率_进度表 *= 组['schedule'].获得_学习率(阶段['step'])

                更新_随着_学习率 = 学习率_进度表 * 更新
                p.data.add_(-更新_随着_学习率)

                阶段['step'] += 1

        return 损失值
