### 项目说明
我再次重新修改了命名来，之前的命名不够准确，这一次看了论文，希望我的理解没有什么太大问题。

该代码是对Bert-Chinese-Text-Classification-Pytorch的仿写，只实现了部分代码，约四分之一。

需要添加参数`--模型 外变双向编码器表示法库`

[原项目地址](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)


[预训练模型下载地址](https://huggingface.co/zozero/my-bert-chinese/tree/main)。这是我进行修改后的模型，如果模型时从原项目下载的，需要运行`修改预训练的模型()`，来适配中文的命名方式，该函数在`运行.py`中。
模型需要放在[预训练的模型](%E9%A2%84%E8%AE%AD%E7%BB%83%E7%9A%84%E6%A8%A1%E5%9E%8B)文件夹中。

可以在[测试代码用](%E6%B8%85%E5%8D%8E%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E5%B7%A5%E5%85%B7%E5%8C%85%2F%E6%95%B0%E6%8D%AE%E9%9B%86%2F%E6%B5%8B%E8%AF%95%E4%BB%A3%E7%A0%81%E7%94%A8)
文件夹下找到我用来测试代码是否能跑通的数据集。当然你也可以下载原版的[数据集](http://thuctc.thunlp.org/)。

### 命名说明
变量名的统一性是必要，这里使得代码工整有序，而且无形中对变量进行了分类。

例如：文字的嵌入层、片段的嵌入层、位置的嵌入层；通过两个字知道了作用，通过后三个字知道了类型。

变量名字数的多少取决于是否能表达其意和是否能区分变量，当然也可以考虑是否顺口来取名。

例如：数量这个词就可以分别用数和量来表示，可以根据是否顺口、是否符合语境来具体使用哪个词。
