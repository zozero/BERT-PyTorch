import os.path
import collections
import logging

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


class 形变双向编码器表示法的分词器:
    def __init__(self, 词汇文件, 是否小写=True, 最大长度=None, 是否分词=True, 绝不_分割=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if not os.path.isfile(词汇文件):
            raise ValueError(
                "不能在{}找到词汇文件。将从google预训练模型使用（tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)）中载入词汇"
                .format(词汇文件))
        self.词汇 = 载入词汇(词汇文件)
        self.序号映射标记 = collections.OrderedDict([(序号, 标记) for 标记, 序号 in self.词汇.items()])
        self.是否分词 = 是否分词
        if 是否分词:
            self.基础的分词器 =

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

    def 分词(self,文本):
        。。。
