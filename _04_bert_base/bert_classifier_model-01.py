# 导入PyTorch深度学习框架
import torch
# 从配置模块导入Config类
from config import Config
# 实例化配置对象,加载BERT tokenizer、模型、配置等
config = Config()
# 从数据加载工具模块导入build_dataloader函数
from dataloader_utils import build_dataloader


# ==================== 自定义BERT分类器 ====================
# 继承torch.nn.Module(所有PyTorch模型的基类)
# "1个继承2个重写"指:
#   - 继承nn.Module
#   - 重写__init__(初始化网络层)
#   - 重写forward(定义前向传播逻辑)
class MyBertClassifier(torch.nn.Module):
    """
    基于BERT的文本分类模型
    架构: BERT编码器 + 线性分类层
    """
    def __init__(self):
        """
        初始化方法:定义模型的网络层结构
        """
        # 调用父类(nn.Module)的构造函数,必须执行
        super().__init__()
        
        # ==================== BERT编码器层 ====================
        # 直接使用配置中预加载的BERT模型(来自bert-base-chinese)
        # BERT输出768维度的向量表示(hidden_size=768)
        self.bert = config.bert_model
        
        # ==================== 线性分类层 ====================
        # 创建全连接层(线性变换),将BERT的768维输出映射到分类数量
        # config.bert_config.hidden_size = 768 (BERT隐藏层维度)
        # config.class_num = 类别数量(根据class.txt自动计算,如10类)
        # 等价于: torch.nn.Linear(768, 10)
        self.linear = torch.nn.Linear(
            config.bert_config.hidden_size,  # 输入维度: 768
            config.class_num                  # 输出维度: 类别数
        )

    def forward(self, x):
        """
        前向传播:定义数据流经网络的计算过程
        :param x: 字典类型的批次数据,包含input_ids、attention_mask等
                  例如: {'input_ids': tensor(...), 'attention_mask': tensor(...)}
        :return: logits - 未经softmax的原始预测分数(形状: [batch_size, class_num])
        """
        # ==================== 步骤1: BERT编码 ====================
        # **x 解包字典,将键值对作为关键字参数传递给BERT模型
        # 等价于: self.bert(input_ids=..., attention_mask=..., token_type_ids=...)
        # result是NamedTuple类型,包含多个属性:
        #   - pooler_output: [batch_size, 768] CLS token的输出代表整个句子的语义表示,用于分类任务
        #   - last_hidden_state: [batch_size, seq_len, 768] 所有token的输出
        #   - hidden_states: 所有层的隐藏状态(如果output_hidden_states=True)
        result = self.bert(**x)
        
        # ==================== 步骤2: 线性分类 ====================
        # 提取pooler_output(CLS位置的向量表示,形状:[batch_size, 768])
        # 通过线性层转换为类别分数(形状:[batch_size, class_num])
        # 例如: batch_size=64, class_num=10 → 输出形状[64, 10]
        logits = self.linear(result.pooler_output)
        
        # 返回logits(未归一化的概率分数)
        # 注意:这里不应用softmax,因为后续使用CrossEntropyLoss损失函数
        # CrossEntropyLoss内部会自动做softmax,重复使用会导致数值不稳定
        return logits


# ==================== 测试代码 ====================
# 当直接运行此文件时执行(被import时不执行)
if __name__ == '__main__':
    # 构建三个数据加载器(训练集、测试集、验证集)
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    
    # 实例化自定义BERT分类模型
    my_bert_model = MyBertClassifier()
    
    # 遍历训练集的批次数据
    # batch_texts_tensor: 字典{'input_ids':..., 'attention_mask':...}
    # batch_labels_tensor: 标签张量[64]
    for batch_texts_tensor, batch_labels_tensor in train_dataloader:
        # 前向传播:获取模型预测结果
        # 注意:变量名有拼写错误,logtis应为logits
        logtis = my_bert_model(batch_texts_tensor)
        
        # 只测试一个批次就退出循环(用于调试)
