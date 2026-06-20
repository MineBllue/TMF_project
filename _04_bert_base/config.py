import torch
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.configuration_bert import BertConfig

class Config:
    def __init__(self):
        # 原始数据路径
        self.data_path = '/Users/wangjie/PycharmProjects/TMP_Project'
        self.train_path = self.data_path + '/_01_data/train.txt'
        self.test_path = self.data_path + '/_01_data/test.txt'
        self.dev_path = self.data_path + '/_01_data/dev.txt'
        self.class_path = self.data_path + '/_01_data/class.txt'
        self.stop_word_path = self.data_path + '/_01_data/stopwords.txt'
        # 分类词表
        self.id2class = {index: class_name for index, class_name in enumerate(open(self.class_path, 'r', encoding='utf8'))}
        # bert相关路径和参数
        self.bert_base_path = self.data_path + '/_04_bert_base/bert-base-chinese'
        # 设备（必须在加载模型前定义）
        self.device = torch.device('mps')
        # 提前加载tokenizer和bert模型
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_base_path)
        self.bert_model = BertModel.from_pretrained(self.bert_base_path).to(self.device)
        # 提前加载config文件对象
        self.bert_config = BertConfig.from_pretrained(self.bert_base_path)
        # bert模型保存路径
        self.bert_classifier_model_save_path = self.data_path + '/_04_bert_base/model/bert_classifier_model.bin'
        # 参数
        self.epochs = 1
        self.batch_size = 64
        self.lr = 5e-5
        self.max_len = 24
        self.class_num = len(self.id2class)

if __name__ == '__main__':
    config = Config()
    print(config.bert_base_path)
    print(config.device)
    print(config.class_num)
