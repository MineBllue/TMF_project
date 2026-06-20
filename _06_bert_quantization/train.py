"""
BERT 文本分类模型训练脚本
=======================
功能：从头训练 BERT 分类模型，支持训练过程中评估并保存最佳模型
训练好的模型可用于后续的量化流程 (bert_DQ.py)

依赖：
  - config.py        : 配置参数
  - dataloader_utils : 数据加载
  - bert_classifier_model : BERT 分类模型定义
  - evaluate_model   : 模型评估函数
"""

from config import Config
from dataloader_utils import build_dataloader
from bert_classifier_model import MyBertClassifier
from evaluate_model import model_evaluate
import torch
from tqdm import tqdm
import time
import os


def train_one_batch(model, batch_texts, batch_labels, loss_fn, optimizer, device):
    """
    训练单个批次

    :param model:       模型
    :param batch_texts:  批次文本数据 (dict of tensors)
    :param batch_labels: 批次标签数据 (tensor)
    :param loss_fn:      损失函数
    :param optimizer:    优化器
    :param device:       设备 (cpu)
    :return:             (损失值, 预测标签列表)
    """
    # 数据移到指定设备
    batch_texts = {k: v.to(device) for k, v in batch_texts.items()}
    batch_labels = batch_labels.to(device)

    # 前向传播
    logits = model(batch_texts)
    loss = loss_fn(logits, batch_labels)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pred_labels = logits.argmax(dim=-1).tolist()
    return loss.item(), pred_labels


def evaluate_and_save(model, dev_dataloader, best_f1_score, save_path):
    """
    在验证集上评估模型，若 F1 提升则保存模型

    :param model:          模型
    :param dev_dataloader: 验证集 DataLoader
    :param best_f1_score:  当前最佳 F1 分数
    :param save_path:      模型保存路径
    :return:               更新后的最佳 F1 分数
    """
    acc, pre, recall, f1 = model_evaluate(dev_dataloader, model)
    print(f'验证集 -> 准确率: {acc:.4f}, 精确率: {pre:.4f}, '
          f'召回率: {recall:.4f}, F1: {f1:.4f}')

    # 恢复训练模式
    model.train()

    if f1 > best_f1_score:
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(model.state_dict(), save_path)
        print(f'✓ 保存最佳模型 (F1: {f1:.4f} > {best_f1_score:.4f}) -> {save_path}')
        return f1

    return best_f1_score


def model_train():
    """
    训练 BERT 分类模型的主流程

    训练流程：
      1. 准备数据加载器
      2. 初始化模型、损失函数、优化器
      3. 逐轮训练，每批更新参数
      4. 定期打印训练日志
      5. 定期在验证集上评估并保存最佳模型
    """
    # ==================== 1. 加载配置 ====================
    config = Config()
    print(f'设备: {config.device}')
    print(f'训练轮数: {config.epochs}')
    print(f'批次大小: {config.batch_size}')
    print(f'学习率: {config.lr}')
    print(f'最大序列长度: {config.max_len}')
    print(f'分类数量: {config.class_num}')
    print(f'模型保存路径: {config.bert_classifier_model_save_path}')
    print('=' * 60)

    # ==================== 2. 准备数据 ====================
    print('加载数据...')
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    print(f'训练集: {len(train_dataloader)} 批')
    print(f'验证集: {len(dev_dataloader)} 批')
    print(f'测试集: {len(test_dataloader)} 批')
    print('=' * 60)

    # ==================== 3. 准备模型 ====================
    print('初始化模型...')
    my_bert_model = MyBertClassifier().to(config.device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(my_bert_model.parameters(), lr=config.lr)

    # ==================== 4. 开始训练 ====================
    print('开始训练...')
    print('=' * 60)
    best_f1_score = 0.0
    global_step = 0
    train_start_time = time.time()

    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        batch_count = 0

        # 使用 tqdm 显示进度条
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{config.epochs}')

        for batch_texts_tensor, batch_labels_tensor in progress_bar:
            # 训练一个批次
            loss_value, _ = train_one_batch(
                my_bert_model, batch_texts_tensor,
                batch_labels_tensor, loss_fn, optimizer, config.device
            )

            total_loss += loss_value
            batch_count += 1
            global_step += 1

            # 更新进度条显示当前损失
            progress_bar.set_postfix({'loss': f'{loss_value:.4f}'})

            # 每 10 步打印一次训练日志
            if global_step % 10 == 0:
                avg_loss = total_loss / batch_count
                elapsed = time.time() - epoch_start_time
                print(f'Epoch {epoch + 1}, Step {global_step}, '
                      f'Loss: {avg_loss:.4f}, 耗时: {elapsed:.2f}s')

            # 每 100 步在验证集上评估一次
            if global_step % 100 == 0:
                print(f'\n--- Step {global_step} 验证评估 ---')
                best_f1_score = evaluate_and_save(
                    my_bert_model, dev_dataloader,
                    best_f1_score, config.bert_classifier_model_save_path
                )
                print('--- 继续训练 ---\n')

        # 每轮结束打印总结
        epoch_elapsed = time.time() - epoch_start_time
        avg_epoch_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f'Epoch {epoch + 1} 完成, 平均 Loss: {avg_epoch_loss:.4f}, '
              f'耗时: {epoch_elapsed:.2f}s')

        # 每轮结束在验证集上评估
        print(f'\n--- Epoch {epoch + 1} 结束验证 ---')
        best_f1_score = evaluate_and_save(
            my_bert_model, dev_dataloader,
            best_f1_score, config.bert_classifier_model_save_path
        )
        print('=' * 60)

    # ==================== 5. 最终测试 ====================
    total_train_time = time.time() - train_start_time
    print(f'\n训练完成! 总耗时: {total_train_time:.2f}s')

    # 加载最佳模型在测试集上评估
    if os.path.exists(config.bert_classifier_model_save_path):
        print('\n加载最佳模型进行测试集评估...')
        my_bert_model.load_state_dict(
            torch.load(config.bert_classifier_model_save_path, map_location=config.device)
        )
        acc, pre, recall, f1 = model_evaluate(test_dataloader, my_bert_model)
        print(f'测试集 -> 准确率: {acc:.4f}, 精确率: {pre:.4f}, '
              f'召回率: {recall:.4f}, F1: {f1:.4f}')
    else:
        print('\n未保存模型，使用最终模型进行测试集评估...')
        acc, pre, recall, f1 = model_evaluate(test_dataloader, my_bert_model)
        print(f'测试集 -> 准确率: {acc:.4f}, 精确率: {pre:.4f}, '
              f'召回率: {recall:.4f}, F1: {f1:.4f}')

    print('\n训练流程全部完成!')


if __name__ == '__main__':
    model_train()
