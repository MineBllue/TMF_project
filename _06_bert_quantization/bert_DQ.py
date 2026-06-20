# 导包
from config import Config
from dataloader_utils import build_dataloader
from bert_classifier_model import MyBertClassifier
from evaluate_model import model_evaluate
import torch

# 提前加载配置文件
config = Config()
# todo 1.准备数据
train_dataloader, dev_dataloader, ßtest_dataloader = build_dataloader()
# todo 2.准备模型 加载训练好的模型参数
my_bert_model = MyBertClassifier()
# todo 注意: 加载模型的时候,选择cpu方式
my_bert_model.load_state_dict(torch.load(config.bert_classifier_model_save_path, map_location=config.device))
# todo 注意: 模型放到指定设备上
my_bert_model.cpu()
my_bert_model.eval()
# todo 3.提前评估下模型
acc, pre, rec, f1 = model_evaluate(dev_dataloader, my_bert_model)
print(f"原始模型: 准确率:{acc},精确率:{pre},召回率:{rec},f1:{f1}")
print('===============================================开始量化======================================================')
# TODO 4.动态量化：只量化分类头 linear 层，不碰 BERT 内部
# 原因：ARM Mac (M1/M2/M3) 上 fbgemm/qnnpack 引擎可能对 BERT 内部操作不完全兼容
# 思路：创建一个只包含 linear 的临时模型做量化，然后替换回原模型
import copy
dq_bert_model = copy.deepcopy(my_bert_model)
dq_bert_model.cpu().eval()

# 方案：用 quantize_dynamic 量化一个只包含单层 Linear 的 wrapper
class LinearWrapper(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
    
    def forward(self, x):
        return self.linear(x)

wrapper = LinearWrapper(dq_bert_model.linear)
# 量化这个 wrapper（只有一层 Linear）
quantized_wrapper = torch.ao.quantization.quantize_dynamic(
    model=wrapper,
    qconfig_spec={torch.nn.Linear},
    dtype=torch.qint8
)
# 替换回量化后的 linear 层
dq_bert_model.linear = quantized_wrapper.linear
print('============================================量化评估和保存======================================================')
# todo 5.量化后评估下模型
acc, pre, rec, f1 = model_evaluate(dev_dataloader, dq_bert_model)

print(f"量化后模型: 准确率:{acc},精确率:{pre},召回率:{rec},f1:{f1}")
# todo 6.保存量化后的模型
torch.save(dq_bert_model.state_dict(), config.quantization_bert_model_save_path)    
print('量化后模型保存成功!')
