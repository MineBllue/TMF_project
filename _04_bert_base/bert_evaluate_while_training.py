from config import Config
from dataloader_utils import build_dataloader
from bert_classifier_model import MyBertClassifier
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tqdm import tqdm
import time
from evaluate_model import model_evaluate

config = Config()


def compute_metrics(y_true, y_pred):
    """
    计算分类指标
    :param y_true: 真实标签列表
    :param y_pred: 预测标签列表
    :return: 准确率、精确率、召回率、F1分数字典
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average="macro", zero_division=0),
        'recall': recall_score(y_true, y_pred, average="macro", zero_division=0),
        'f1': f1_score(y_true, y_pred, average="macro", zero_division=0)
    }


def print_training_log(epoch, batch_idx, metrics, elapsed_time):
    """
    打印训练日志
    :param epoch: 当前轮次
    :param batch_idx: 批次索引
    :param metrics: 指标字典
    :param elapsed_time: 耗时(秒)
    """
    print(f'第{epoch}轮，第{batch_idx}批，损失为：{metrics["loss"]:.4f}')
    print(f'第{epoch}轮，第{batch_idx}批，准确率为：{metrics["accuracy"]:.4f}')
    print(f'第{epoch}轮，第{batch_idx}批，精确率为：{metrics["precision"]:.4f}')
    print(f'第{epoch}轮，第{batch_idx}批，召回率为：{metrics["recall"]:.4f}')
    print(f'第{epoch}轮，第{batch_idx}批，F1为：{metrics["f1"]:.4f}')
    print(f'第{epoch}轮，第{batch_idx}批，用时：{elapsed_time:.2f}秒')


def train_one_batch(model, batch_texts, batch_labels, loss_fn, optimizer, device):
    """
    训练单个批次
    :return: 损失值、预测标签
    """
    # 将数据移到 MPS 设备
    batch_texts = {k: v.to(device) for k, v in batch_texts.items()}
    batch_labels = batch_labels.to(device)

    logits = model(batch_texts)
    loss = loss_fn(logits, batch_labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    pred_labels = logits.argmax(dim=-1).tolist()
    return loss.item(), pred_labels


def evaluate_and_save(model, dev_dataloader, best_f1_score, save_path):
    """
    在验证集上评估并保存最佳模型
    :return: 更新后的最佳F1分数
    """
    acc, pre, recall, f1 = model_evaluate(dev_dataloader, model)
    print(f'评估日志：测试集准确率为：{acc:.4f}, 测试集精确率为：{pre:.4f}, '
          f'测试集召回率为：{recall:.4f}, 测试集F1为：{f1:.4f}')
    
    model.train()
    
    if f1 > best_f1_score:
        torch.save(model.state_dict(), save_path)
        print(f'保存模型，当前F1值为：{f1:.4f}')
        return f1
    
    return best_f1_score


def model_train():
    """训练BERT分类模型"""
    # 准备数据
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    # 准备模型，并移到 MPS 设备
    my_bert_model = MyBertClassifier().to(config.device)
    # 准备损失函数
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    # 准备优化器
    optimizer = torch.optim.AdamW(my_bert_model.parameters(), lr=config.lr)
    
    best_f1_score = 0
    
    for epoch in range(config.epochs):
        total_loss, batch_cnt = 0, 0
        all_pred_labels, all_true_labels = [], []
        start_time = time.time()
        
        for index, (batch_texts_tensor, batch_labels_tensor) in enumerate(
            tqdm(train_dataloader), start=1
        ):
            # 训练单个批次
            loss_value, pred_labels = train_one_batch(
                my_bert_model, batch_texts_tensor,
                batch_labels_tensor, loss_fn, optimizer, config.device
            )
            
            # 累积统计信息
            total_loss += loss_value
            batch_cnt += 1
            all_pred_labels.extend(pred_labels)
            all_true_labels.extend(batch_labels_tensor.tolist())
            
            # 每10批打印训练日志
            if index % 10 == 0 or index == len(train_dataloader):
                metrics = compute_metrics(all_true_labels, all_pred_labels)
                metrics['loss'] = total_loss / batch_cnt
                elapsed_time = time.time() - start_time
                
                print_training_log(epoch, index, metrics, elapsed_time)
                
                # 重置统计变量
                total_loss, batch_cnt = 0, 0
                all_pred_labels, all_true_labels = [], []
                start_time = time.time()
            
            # 每100批进行评估
            if index % 100 == 0 or index == len(train_dataloader):
                best_f1_score = evaluate_and_save(
                    my_bert_model, dev_dataloader, 
                    best_f1_score, config.bert_classifier_model_save_path
                )


if __name__ == '__main__':
    model_train()