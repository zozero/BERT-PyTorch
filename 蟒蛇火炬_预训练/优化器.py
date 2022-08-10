import abc
import logging
import math
import sys

from torch.optim import Optimizer
from torch.optim.optimizer import required

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


class 模型的自适应动量估计(Optimizer):
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
        默认字典 = dict(学习率=学习率, 进度表=进度表, b1=b1, b2=b2, 艾普西龙=艾普西龙, 权重_衰减=权重_衰减, 最大_梯度_范数=最大_梯度_范数)
        super(模型的自适应动量估计, self).__init__(参数列表, 默认字典)

    def step(self,closure=None):
        ....
