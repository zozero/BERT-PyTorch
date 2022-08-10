import collections
import logging
import os.path

import unicodedata

from 蟒蛇火炬_预训练.文件_多个工具 import 形成缓存路径

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
预训练词汇嵌入层位置上的层数映射 = {
    '基础-不分大小写': 512,
    '大量-不分大小写': 512,
    '基础-分大小写': 512,
    '大量-分大小写': 512,
    '基础-多语言的-不分大小写': 512,
    '基础-多语言的-分大小写': 512,
    '基础-汉语': 512
}

词汇文件名 = '词汇.txt'


def 载入词汇(词汇文件):
    词汇 = collections.OrderedDict()
    索引 = 0
    with open(词汇文件, "r", encoding="utf-8") as 读者:
        while True:
            标志 = 读者.readline()
            if not 标志:
                break
            标志 = 标志.strip()
            词汇[标志] = 索引
            索引 += 1
    return 词汇


def 对空格分词(文本):
    文本 = 文本.strip()
    if not 文本:
        return []
    分词列表 = 文本.split()
    return 分词列表


class 形变双向编码器表示法的分词器:
    """
    分词：将句子分成一个个文字，放在一个列表中
    词：本义是将内心所想的外化表达，古人称最小表义单位为“词”，称由句子组成的完整表达为“辞”；
    词汇：是一种语言里所有的（或特定范围的）词和固定短语的总和，由于词汇文件中包含字和单词所以这里称为词汇
    词汇文件：放有许多句子的文件，路径：形变双向编码器表示法_预训练模型/词汇.txt
    词条：表示一段话或者一句话
    记录器：各种信息记录，相当于日志
    文本：一句话
    字符：很可能是一个汉字，也可能是一个标点符号
    """

    def __init__(self, 词汇文件, 是否小写=True, 最大长度=None, 是否分词=True, 绝不_分割=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if not os.path.isfile(词汇文件):
            raise ValueError(
                "不能在{}找到词汇文件。将从google预训练模型使用（tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)）中载入词汇"
                .format(词汇文件))
        self.词汇 = 载入词汇(词汇文件)
        self.序号映射标记 = collections.OrderedDict([(序号, 标记) for 标记, 序号 in self.词汇.items()])
        self.是否分词 = 是否分词
        if 是否分词:
            self.基础的分词器 = 基础的分词器(是否小写=是否小写, 绝不_分割=绝不_分割)
        self.词条_分词器 = 词条分词器(词汇=self.词汇)
        self.最大长度 = 最大长度 if 最大长度 is not None else int(1e12)

    def 分词(self, 文本):
        分割_分词列表 = []
        if self.是否分词:
            for 词 in self.基础的分词器.分词(文本):
                for 子词 in self.词条_分词器.分词(词):
                    分割_分词列表.append(子词)
        else:
            分割_分词列表 = self.词条_分词器.分词(文本)
        return 分割_分词列表

    def 把字符转换到标记(self, 字符列表):
        """
        和词汇文件相对应的行数
        :param 字符列表:
        :return:
        """
        标记列表 = []
        for 字符 in 字符列表:
            标记列表.append(self.词汇[字符])
        if len(标记列表) > self.最大长度:
            记录器.warning("字符列表的长度超过规定的最大长度（{}>{}）可能导致索引错误".format(len(标记列表), self.最大长度))
        return 标记列表

    def 把转标记换到字符(self, 标记列表):
        字符列表 = []
        for i in 标记列表:
            字符列表.append(self.序号映射标记[i])
        return 字符列表

    @classmethod
    def 从预训练开始(cls, 预训练模型或者路径, 缓存_目录=None, *复数输入, **参数字典):
        if 预训练模型或者路径 in 预训练词汇档案映射:
            词汇文件 = 预训练词汇档案映射[预训练模型或者路径]
            if '-分大小写' in 预训练模型或者路径 and 参数字典.get('是否小写', True):
                记录器.warning("你正在载入的预训练模型是不分大小写的模型，但你设置的是（是否小写=False）。我们将重新设置（是否小写=True），但你可能想检查这个行为。")
                参数字典['是否小写'] = True
            elif '-分大小写' not in 预训练模型或者路径 and not 参数字典.get('是否小写', True):
                记录器.warning("你正在载入的预训练模型是不分大小写的模型，但你设置的是（是否小写=False）。我们将重新设置（是否小写=True），但你可能想检查这个行为。")
                参数字典['是否小写'] = True
        else:
            词汇文件 = 预训练模型或者路径
        if os.path.isdir(词汇文件):
            词汇文件 = os.path.join(词汇文件, 词汇文件名)

        try:
            已解析的词汇文件 = 形成缓存路径(词汇文件, 缓存_目录=缓存_目录)
        except EnvironmentError:
            记录器.error(
                "模型名'{}'不能在模型名列表({})中找到。我们假定'{}'不能找到路径或者网址内的任何文件".format(预训练模型或者路径, ','.join(预训练词汇档案映射.keys()),
                                                                         词汇文件)
            )
            return None
        if 已解析的词汇文件 == 词汇文件:
            记录器.info("载入词汇文件{}".format(词汇文件))
        else:
            记录器.info("从{}的缓存中载入词汇文件{}".format(词汇文件, 已解析的词汇文件))
        if 预训练模型或者路径 in 预训练词汇嵌入层位置上的层数映射:
            最大长度 = 预训练词汇嵌入层位置上的层数映射[预训练模型或者路径]
            参数字典['最大长度'] = min(参数字典.get('最大长度', int(1e12)), 最大长度)
        分词器 = cls(已解析的词汇文件, *复数输入, **参数字典)
        return 分词器


class 基础的分词器:
    def __init__(self, 是否小写=True, 绝不_分割=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        self.是否小写 = 是否小写
        self.绝不_分割 = 绝不_分割

    def 分词(self, 文本):
        文本 = self._清洗文本(文本)
        文本 = self._对中文字符分词(文本)
        原始分词列表 = 对空格分词(文本)
        分割_分词列表 = []
        for 分词 in 原始分词列表:
            if self.是否小写 and 分词 not in self.绝不_分割:
                分词 = 分词.lower()
                分词 = self._清除重音(分词)
            分割_分词列表.extend(self._分割标点符号(分词))

        输出_分词列表 = 对空格分词(" ".join(分割_分词列表))
        return 输出_分词列表

    def _清除重音(self, 文本):
        文本 = unicodedata.normalize('NFD', 文本)
        输出 = []
        for 字符 in 文本:
            类别 = unicodedata.category(字符)
            if 类别 == 'Mn':
                continue
            输出.append(字符)
        return "".join(输出)

    def _分割标点符号(self, 文本):
        if 文本 in self.绝不_分割:
            return [文本]
        字符列表 = list(文本)
        i = 0
        是否开始新的字符 = True
        输出 = []
        while i < len(字符列表):
            字符 = 字符列表[i]
            if _判断是否为标点符号(字符):
                输出.append([字符])
                是否开始新的字符 = True
            else:
                if 是否开始新的字符:
                    输出.append([])
                是否开始新的字符 = False
                输出[-1].append(字符)
            i += 1
        return ["".join(x) for x in 输出]

    def _对中文字符分词(self, 文本):
        输出 = []
        for 字符 in 文本:
            码位 = ord(字符)
            if self._判断是否为中文字符(码位):
                输出.append(' ')
                输出.append(字符)
                输出.append(' ')
            else:
                输出.append(字符)
        return ''.join(输出)

    def _判断是否为中文字符(self, 码位):
        """

        :param 码位: (Unicode code point)
        :return:
        """
        if (
                (0x4E00 <= 码位 <= 0x9FFF) or
                (0x3400 <= 码位 <= 0x4DBF) or
                (0x20000 <= 码位 <= 0x2A6DF) or
                (0x2A700 <= 码位 <= 0x2B73F) or
                (0x2B740 <= 码位 <= 0x2B81F) or
                (0x2B820 <= 码位 <= 0x2CEAF) or
                (0xF900 <= 码位 <= 0xFAFF) or
                (0x2F800 <= 码位 <= 0x2FA1F)):
            return True
        return False

    def _清洗文本(self, 文本):
        输出 = []
        for 字符 in 文本:
            码位 = ord(字符)
            if 码位 == 0 or 码位 == 0xfffd or _判断是否为控制字符(字符):
                continue
            if _判断是否为空格(字符):
                输出.append(' ')
            else:
                输出.append(字符)
        return "".join(输出)


class 词条分词器:
    def __init__(self, 词汇, 未知字符="[UNK]", 每个词的最大字符数=100):
        self.词汇 = 词汇
        self.未知字符 = 未知字符
        self.每个词的最大字符数 = 每个词的最大字符数

    def 分词(self, 文本):
        输出的词汇列表 = []
        for 词汇 in 对空格分词(文本):
            字符列表 = list(词汇)
            if len(字符列表) > self.每个词的最大字符数:
                输出的词汇列表.append(self.未知字符)
                continue

            是否是坏的 = False
            起始 = 0
            子词汇列表 = []
            while 起始 < len(字符列表):
                末尾 = len(字符列表)
                当前子字符串 = None
                while 起始 < 末尾:
                    子字符串 = "".join(字符列表[起始:末尾])
                    if 起始 > 0:
                        子字符串 = "##" + 子字符串
                    if 子字符串 in self.词汇:
                        当前子字符串 = 子字符串
                        break
                    末尾 -= 1
                if 当前子字符串 is None:
                    是否是坏的 = True
                    break
                子词汇列表.append(当前子字符串)
                起始 = 末尾

            if 是否是坏的:
                输出的词汇列表.append(self.未知字符)
            else:
                输出的词汇列表.extend(子词汇列表)
        return 输出的词汇列表


def _判断是否为空格(字符):
    if 字符 == ' ' or 字符 == '\t' or 字符 == '\n' or 字符 == '\r':
        return True
    类别 = unicodedata.category(字符)
    if 类别 == 'Zs':
        return True
    return False


def _判断是否为控制字符(字符):
    if 字符 == '\t' or 字符 == '\n' or 字符 == '\r':
        return False
    类别 = unicodedata.category(字符)
    if 类别.startswith('C'):
        return True
    return False


def _判断是否为标点符号(字符):
    码位 = ord(字符)
    if (
            (33 <= 码位 <= 47) or
            (58 <= 码位 <= 64) or
            (91 <= 码位 <= 96) or
            (123 <= 码位 <= 126)
    ):
        return True
    类别 = unicodedata.category(字符)
    if 类别.startswith("P"):
        return True
    return False
