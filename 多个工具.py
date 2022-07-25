from tqdm import tqdm


def 构建数据集(配置):
    def 载入数据(路径,填充大小=32):
        内容列表=[]
        with open(路径,'r',encoding='UTF-8') as 文件:
            for 行 in tqdm(文件):
                一行=行.strip()
                if not 一行:
                    continue
                内容,标签=一行.split('\t')
                标记=配置.分词器.
