import fnmatch
import json
import logging
import os
import shutil
import sys
import tempfile
from functools import wraps
from hashlib import sha256

import requests
from botocore.exceptions import ClientError
import boto3

from urllib.parse import urlparse

from tqdm import tqdm

try:
    from pathlib import Path

    蟒蛇火炬_预训练_形变双向编码器表示法_缓存 = Path(os.getenv('蟒蛇火炬_预训练_形变双向编码器表示法_缓存',
                                            Path.home() / '.蟒蛇火炬_预训练_形变双向编码器表示法'))
except (AttributeError, ImportError):
    蟒蛇火炬_预训练_形变双向编码器表示法_缓存 = Path('蟒蛇火炬_预训练_形变双向编码器表示法_缓存',
                                  os.path.join(os.path.expanduser("~"), '.蟒蛇火炬_预训练_形变双向编码器表示法'))

记录器 = logging.getLogger(__name__)


def 网址转文件名(网址, 网络标签=None):
    网址_字节 = 网址.encode('utf-8')
    网址_哈希 = sha256(网址_字节)
    文件名 = 网址_哈希.hexdigest()

    if 网络标签:
        网址_字节 = 网络标签.encode('utf-8')
        网址_哈希 = sha256(网址_字节)
        文件名 += '.' + 网址_哈希.hexdigest()
    return 文件名


def 形成缓存路径(网址或文件名, 缓存_目录=None):
    if 缓存_目录 is None:
        缓存_目录 = 蟒蛇火炬_预训练_形变双向编码器表示法_缓存
    if sys.version_info[0] == 3 and isinstance(网址或文件名, Path):
        网址或文件名 = str(网址或文件名)
    if sys.version_info[0] == 3 and isinstance(缓存_目录, Path):
        缓存_目录 = str(缓存_目录)

    已解析 = urlparse(网址或文件名)

    if 已解析.scheme in ('http', 'https', 's3'):
        return 从缓存中获取(网址或文件名, 缓存_目录)
    elif os.path.exists(网址或文件名):
        return 网址或文件名
    elif 已解析.scheme == '':
        raise EnvironmentError("%s文件没有找到".format(网址或文件名))
    else:
        raise ValueError("%s无法解析成网址或者本地的路径".format(网址或文件名))


def 分割亚马逊简单存储服务路径(网址):
    已解析 = urlparse(网址)
    if not 已解析.netloc or not 已解析.path:
        raise ValueError("错误的亚马逊简单存储服务路径：{}".format(网址))
    水桶名 = 已解析.netloc
    亚马逊简单存储服务路径 = 已解析.path
    if 亚马逊简单存储服务路径.startswith("/"):
        亚马逊简单存储服务路径 = 亚马逊简单存储服务路径[1:]
    return 水桶名, 亚马逊简单存储服务路径


def 亚马逊简单存储服务_请求(函数):
    @wraps(函数)
    def 包装(网址, *参数列表, **参数字典):
        try:
            return 函数(网址, *参数列表, **参数字典)
        except ClientError as 错误:
            if int(错误.response['Error']['Code']) == 404:
                raise EnvironmentError("{}文件没有找到".format(网址))
            else:
                raise

    return 包装


@亚马逊简单存储服务_请求
def 亚马逊简单存储服务_网络标签(网址):
    亚马逊简单存储服务_资源 = boto3.resource("s3")
    水桶名, 亚马逊简单存储服务路径 = 分割亚马逊简单存储服务路径(网址)
    亚马逊简单存储服务对象 = 亚马逊简单存储服务_资源.Object(水桶名, 亚马逊简单存储服务路径)
    return 亚马逊简单存储服务对象.e_tag


@亚马逊简单存储服务_请求
def 亚马逊简单存储服务_获得(网址, 临时文件):
    亚马逊简单存储服务_资源 = boto3.resource("s3")
    水桶名, 亚马逊简单存储服务路径 = 分割亚马逊简单存储服务路径(网址)
    亚马逊简单存储服务_资源.Bucket(水桶名).download_fileobj(亚马逊简单存储服务路径, 临时文件)


def 超文本传输_获得(网址, 临时文件):
    请求 = requests.get(网址, stream=True)
    内容_长度 = 请求.headers.get('Content-Length')
    全部 = int(内容_长度) if 内容_长度 is not None else None
    进度 = tqdm(unit="B", total=全部)
    for 块 in 请求.iter_content(chunk_size=1024):
        if 块:
            进度.update(len(块))
            临时文件.write(块)
    进度.close()


def 从缓存中获取(网址, 缓存_目录=None):
    if 缓存_目录 is None:
        缓存_目录 = 蟒蛇火炬_预训练_形变双向编码器表示法_缓存
    if sys.version_info[0] == 3 and isinstance(缓存_目录, Path):
        缓存_目录 = str(缓存_目录)

    if not os.path.exists(缓存_目录):
        os.makedirs(缓存_目录)

    if 网址.startswith("s3://"):
        网络标签 = 亚马逊简单存储服务_网络标签(网址)
    else:
        try:
            响应 = requests.head(网址, allow_redirects=True)
            if 响应.status_code != 200:
                网络标签 = None
            else:
                网络标签 = 响应.headers.get("ETag")
        except EnvironmentError:
            网络标签 = None

    if sys.version_info[0] == 2 and 网络标签 is not None:
        网络标签 = 网络标签.decode('utf-8')
    文件名 = 网址转文件名(网址, 网络标签)

    缓存路径 = os.path.join(缓存_目录, 文件名)

    if not os.path.exists(缓存路径) and 网络标签 is None:
        匹配_文件 = fnmatch.filter(os.listdir(缓存_目录), '.*')
        匹配_文件 = list(filter(lambda s: not s.endswitch('.json'), 匹配_文件))
        if 匹配_文件:
            缓存路径 = os.path.join(缓存_目录, 匹配_文件[-1])

    if not os.path.exists(缓存路径):
        with tempfile.NamedTemporaryFile() as 临时文件:
            记录器.info("%s 在缓存中没有找到，正在下载到%s", 网址, 临时文件.name)

            if 网址.startswith("s3://"):
                亚马逊简单存储服务_获得(网址, 临时文件)
            else:
                超文本传输_获得(网址, 临时文件)

            临时文件.flush()
            临时文件.seek(0)

            记录器.info("复制%s到缓存%s中", 临时文件.name, 缓存路径)
            with open(缓存路径, 'wb') as 缓存文件:
                # 管理的货架工具
                shutil.copyfileobj(临时文件, 缓存文件)
            记录器.info("为文件%s创建元信息", 缓存路径)
            元信息 = {'url': 网址, 'etag': 网络标签}
            元信息_路径 = 缓存路径 + '.json'
            with open(元信息_路径, 'w') as 元信息_文件:
                输出字符串 = json.dumps(元信息)
                # if sys.version_info[0]==2 and isinstance(输出字符串,str):
                #     输出字符串=unicode(输出字符串,'utf-8')
                元信息_文件.write(输出字符串)
            记录器.info("删除临时文件%s", 临时文件)

    return 缓存路径
