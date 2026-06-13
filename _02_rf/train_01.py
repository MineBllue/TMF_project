import pandas as pd
import jieba
from config import Config
config = Config()

def process_data(base_path, process_path):
    # 1. 读取数据
    df_data = pd.read_csv(base_path, sep='\t', header=None, names=['text', 'label'])
    # 2. 对每一行进行分词，并存储为一列
    df_data['words'] = df_data['text'].apply(lambda x: ' '.join(jieba.lcut(x)))
    # 3. 保存数据
    df_data.to_csv(process_path, sep='\t', index=None, header=True)

if __name__ == '__main__':
    # 处理训练集
    process_data(config.train_path, config.process_train_path)
    # 处理测试集
    process_data(config.test_path, config.process_test_path)
    # 处理验证集
    process_data(config.dev_path, config.process_dev_path)