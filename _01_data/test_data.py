from collections import Counter

from config import Config
import pandas as pd

def get_data(path):
    # todo：1.读取数据 指定列名
    data = pd.read_csv(path, sep='\t', header=None)
    data.columns = ['text', 'label']

    # todo：2.了解类别分布情况是否均匀
    # print(data['label'].value_counts())
    counters = Counter(data['label'])
    print(counters) # 统计每个类别出现的次数
    # 占比
    for k, v in counters.items():
        print(k, v, v / len(data))

    # todo: 了解数据文本的分布长度
    # 获取每个文本的长度并作为一列存储到data中
    data['len'] = data['text'].apply(lambda x: len(x))
    print('获取每个文本的长度', data['len'])
    print(data['len'].describe())
    print(data)


if __name__ == '__main__':
    config = Config()
    get_data(config.train_path)
    print('=='*20)
    get_data(config.test_path)
    print('=='*20)
    get_data(config.dev_path)
    print('=='*20)
    # data = get_data(config.train_path)
    # print(data.head())
    # print(data['label'].head())
    # print(data['text'].head()