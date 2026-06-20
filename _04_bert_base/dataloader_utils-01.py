# 导入PyTorch深度学习框架
import torch
# 从配置模块导入Config类,包含所有路径和超参数
from config import Config
# 实例化配置对象,加载BERT tokenizer、模型等
config = Config()


def load_data_list(base_path):
    """
    从文件中加载数据集
    文件格式: 每行是 "文本内容\t标签"(tab分隔)
    例如: "这部电影很好看\t0"
    """
    # 初始化空列表存储(文本, 标签)元组
    data_list = []
    # 逐行读取文件,encoding='utf8'处理中文
    for line in open(base_path, 'r', encoding='utf8'):
        # strip()去除首尾空白和换行符,\t按tab分割文本和标签
        text, label = line.strip().split('\t')
        # 将标签字符串转换为整数(如"0"->0)
        label = int(label)
        # 将(文本, 标签)元组添加到列表
        data_list.append((text, label))
    # 返回所有数据样本的列表
    return data_list


class MyDataset(torch.utils.data.Dataset):
    """
    自定义数据集类,继承PyTorch的Dataset
    用于封装数据,配合DataLoader实现批量加载
    """
    def __init__(self, data_list):
        """
        初始化方法
        :param data_list: 由load_data_list返回的[(文本, 标签), ...]列表
        """
        # 保存数据列表为实例属性
        self.data_list = data_list
    
    def __len__(self):
        """
        返回数据集的总样本数
        DataLoader需要此方法来确定迭代次数
        """
        return len(self.data_list)
    
    def __getitem__(self, index):
        """
        根据索引获取单个样本
        DataLoader会根据batch_size自动调用此方法获取多个样本
        :param index: 样本索引
        :return: (文本, 标签)元组
        """
        return self.data_list[index]


def my_collate_fn(batch):
    """
    自定义批次整理函数
    DataLoader将多个样本组合成batch后,调用此函数进行批处理
    :param batch: 由__getitem__返回的多个样本组成的列表
                  [(文本1, 标签1), (文本2, 标签2), ...]
    :return: (批次文本张量, 批次标签张量)
    """
    # zip(*batch)解包:将所有文本放入texts元组,所有标签放入labels元组
    # 例如: batch=[("你好",0), ("世界",1)] -> texts=("你好","世界"), labels=(0,1)
    texts, labels = zip(*batch)
    
    # 使用BERT tokenizer将文本列表转换为张量
    # truncation=True:超过max_length的文本截断
    # padding=True:不足max_length的文本填充到相同长度
    # max_length=24:最大序列长度
    # return_tensors='pt':返回PyTorch张量格式
    # 返回值是字典:{'input_ids':..., 'attention_mask':..., 'token_type_ids':...}
    batch_texts_tensor = config.bert_tokenizer(
        texts, 
        truncation=True, 
        padding=True, 
        max_length=config.max_len, 
        return_tensors='pt'
    )
    
    # 将标签元组转换为PyTorch张量
    batch_labels_tensor = torch.tensor(labels)
    
    # 返回批次数据(字典形式的文本张量, 标签张量)
    return batch_texts_tensor, batch_labels_tensor


def build_dataloader():
    """
    构建训练集、测试集、验证集的DataLoader
    DataLoader负责批量加载数据、打乱顺序、并行加载等
    :return: 三个DataLoader对象
    """
    # ==================== 训练集 ====================
    # 从train.txt加载训练数据列表
    train_data_list = load_data_list(config.train_path)
    # 封装为Dataset对象
    train_data_set = MyDataset(train_data_list)
    # 创建DataLoader
    # batch_size=64:每批64个样本
    # shuffle=True:每个epoch打乱数据顺序(对训练集重要)
    # collate_fn=my_collate_fn:使用自定义函数处理批次
    train_dataloader = torch.utils.data.DataLoader(
        train_data_set, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=my_collate_fn
    )

    # ==================== 测试集 ====================
    # 从test.txt加载测试数据
    test_data_list = load_data_list(config.test_path)
    test_data_set = MyDataset(test_data_list)
    # shuffle=True对测试集影响不大,通常设为False
    test_dataloader = torch.utils.data.DataLoader(
        test_data_set, 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=my_collate_fn
    )

    # ==================== 验证集 ====================
    # 从dev.txt加载验证数据
    dev_data_list = load_data_list(config.dev_path)
    dev_data_set = MyDataset(dev_data_list)
    dev_dataloader = torch.utils.data.DataLoader(
        dev_data_set, 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=my_collate_fn
    )

    # 返回三个DataLoader,供训练、测试、验证使用
    return train_dataloader, test_dataloader, dev_dataloader


# 主程序入口:当直接运行此文件时执行测试代码
if __name__ == '__main__':
    # 构建三个数据加载器
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    
    # 打印数据统计信息
    # len(dataloader)返回的是批次数,不是样本数
    # 总样本数 = 批次数 × batch_size(最后一批可能不足)
    print(f'训练集数量一共：{len(train_dataloader)}批，每批{config.batch_size}个数据')
    print(f'测试集数量一共：{len(test_dataloader)}批，每批{config.batch_size}个数据')
    print(f'验证集数量一共：{len(dev_dataloader)}批，每批{config.batch_size}个数据')
