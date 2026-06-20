import torch
from config import Config
config = Config()
from dataloader_utils import build_dataloader

# 自定义分类模型， 1个继承2个重写
class MyBertClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 隐藏层使用bert的原始结构
        self.bert = config.bert_model
        # bert输出768维度不符合当前10分类需求，添加线性层转换
        self.linear = torch.nn.Linear(config.bert_config.hidden_size, config.class_num)

    def forward(self, x):
        result = self.bert(**x)
        # result结果两部分：1.pooler_output 2.hidden_states
        # 把bert的768维转换为10维
        logits = self.linear(result.pooler_output)
        return logits # 后续做多分类交叉熵损失自带softmax，所以不需要softmax


if __name__ == '__main__':
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    # 准备模型
    my_bert_model = MyBertClassifier()
    # collate_fn调用时机：训练时，每次迭代都会调用一次
    for batch_texts_tensor, batch_labels_tensor in train_dataloader:
        # 前向传播
        logits = my_bert_model(batch_texts_tensor)
        break
