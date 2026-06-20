import torch
from config import Config
config = Config()

def load_data_list(base_path):
    """
    :param base_path: 数据集路径
    原始数据: 新闻\t标签\n新闻\t标签\n...
    :return：[（新闻，标签)，（新闻，标签)...]
    """
    # 提前创建一个列表，用于存储（新闻，标签)元组
    data_list = []
    for line in open(base_path, 'r', encoding='utf8'):
        # 获取新闻和标签
        text, label = line.strip().split('\t')
        label = int(label)
        data_list.append((text, label))
    return data_list

# 自定义dataset 1个继承3个重写
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, index):
        return self.data_list[index]

# 自定义collate_fn
def my_collate_fn(batch):
    # zip = 迭代器
    texts, labels = zip(*batch)
    # 把数据进行tokenizer处理，转为 tensor
    batch_texts_tensor = config.bert_tokenizer(texts, truncation=True, padding=True, max_length=config.max_len, return_tensors='pt')
    # 把标签转为 tensor
    batch_labels_tensor = torch.tensor(labels)
    return batch_texts_tensor, batch_labels_tensor


def build_dataloader():
    # 训练集
    train_data_list = load_data_list(config.train_path)
    train_data_set = MyDataset(train_data_list)
    #--------------------------------------------------训练集需要打乱，但是测试集验证集不用------------shuffle= True 意思是打乱数据
    train_dataloader = torch.utils.data.DataLoader(train_data_set, batch_size=config.batch_size, shuffle=True, collate_fn=my_collate_fn)
    # 测试集
    test_data_list = load_data_list(config.test_path)
    test_data_set = MyDataset(test_data_list)
    #--------------------------------------------------------------------------------shuffle= True 意思是打乱数据
    test_dataloader = torch.utils.data.DataLoader(test_data_set, batch_size=config.batch_size, shuffle=False, collate_fn=my_collate_fn)
    # 验证集
    dev_data_list = load_data_list(config.dev_path)
    dev_data_set = MyDataset(dev_data_list)
    #--------------------------------------------------------------------------------shuffle= True 意思是打乱数据
    dev_dataloader = torch.utils.data.DataLoader(dev_data_set, batch_size=config.batch_size, shuffle=False, collate_fn=my_collate_fn)

    return train_dataloader, test_dataloader, dev_dataloader

if __name__ == '__main__':
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    print(f'训练集数量一共：{len(train_dataloader)}批，每批{config.batch_size}个数据')
    print(f'测试集数量一共：{len(test_dataloader)}批，每批{config.batch_size}个数据')
    print(f'验证集数量一共：{len(dev_dataloader)}批，每批{config.batch_size}个数据')
