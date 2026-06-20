import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
config = Config()
import torch
from dataloader_utils import build_dataloader
from bert_classifier_model import MyBertClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def model_evaluate():
    # 2个准备
    # 准备数据
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    # 准备模型，并移到 MPS 设备
    my_bert_model = MyBertClassifier().to(config.device)
    my_bert_model.load_state_dict(torch.load(config.bert_classifier_model_save_path, map_location=config.device))
    my_bert_model.eval()
    # 评估不需要计算梯度，关闭计算梯度优化评估速度
    with torch.no_grad():
        # 提前创建两个列表，所有的预测标签跟真实标签
        all_perd_labels, all_true_labels = [], []
        # 1个遍历
        for index, (batch_texts_tensor, batch_labels_tensor) in enumerate(tqdm(test_dataloader), start=1):
            # 将数据移到 MPS 设备
            batch_texts_tensor = {k: v.to(config.device) for k, v in batch_texts_tensor.items()}
            batch_labels_tensor = batch_labels_tensor.to(config.device)
            # 前向传播
            logits = my_bert_model(batch_texts_tensor)
            # 累加每批的预测标签和真实标签
            all_true_labels.extend(batch_labels_tensor.tolist())
            batch_perd_label = torch.argmax(logits, dim=-1)
            all_perd_labels.extend(batch_perd_label.tolist())
        acc = accuracy_score(all_true_labels, all_perd_labels)
        pre = precision_score(all_true_labels, all_perd_labels, average="macro", zero_division=0)
        recall = recall_score(all_true_labels, all_perd_labels, average="macro", zero_division=0)
        f1 = f1_score(all_true_labels, all_perd_labels, average="macro", zero_division=0)

        print(f'准确率为：{acc}, 精确率为：{pre}, 召回率为：{recall}, F1为：{f1}')


if __name__ == '__main__':
    model_evaluate()