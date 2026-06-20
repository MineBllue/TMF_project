from config import Config
config = Config()
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def model_evaluate(dev_loader, model):
    model.eval()
    # 评估不需要计算梯度，关闭计算梯度优化评估速度
    with torch.no_grad():
        # 提前创建两个列表，所有的预测标签跟真是标签
        all_perd_labels, all_true_labels = [], []
        # 1个遍历
        for index, (batch_texts_tensor, batch_labels_tensor) in enumerate(tqdm(dev_loader), start=1):
            # 将数据移到 MPS 设备
            batch_texts_tensor = {k: v.to(config.device) for k, v in batch_texts_tensor.items()}
            batch_labels_tensor = batch_labels_tensor.to(config.device)
            # 前向传播
            logits = model(batch_texts_tensor)
            # 累加每批的预测标签和真是标签
            all_true_labels.extend(batch_labels_tensor.tolist())
            batch_perd_label = torch.argmax(logits, dim=-1)
            all_perd_labels.extend(batch_perd_label.tolist())
        acc = accuracy_score(all_true_labels, all_perd_labels)
        pre = precision_score(all_true_labels, all_perd_labels, average="macro")
        recall = recall_score(all_true_labels, all_perd_labels, average="macro")
        f1 = f1_score(all_true_labels, all_perd_labels, average="macro")
    return  acc, pre, recall, f1