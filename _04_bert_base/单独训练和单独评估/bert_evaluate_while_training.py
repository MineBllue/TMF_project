# 添加父目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from dataloader_utils import build_dataloader
from bert_classifier_model import MyBertClassifier
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tqdm import tqdm # 引入进度条包
import time
# 斌子法则 14251
# todo 1个配置文件
config = Config()

def model_train():
    # todo 4个准备
    # 准备数据
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
    # 准备模型
    my_bert_model = MyBertClassifier()
    # 准备损失函数
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    # 准备优化器 parameters() 返回一个迭代器,包含模型中所有需要训练的张量参数
    optimizer = torch.optim.AdamW(my_bert_model.parameters(), lr=config.lr)
    # todo  2个循环
    # 外层遍历轮次
    for epoch in range(config.epochs):
        # 额外添加日志 每10批打印一次 损失，准确率，精确率，召回率，F1
        total_loss, batch_cnt = 0, 0
        all_pred_labels, all_true_labels = [], []
        start_time = time.time()
        # 内层遍历批次
        for index, (batch_texts_tensor, batch_labels_tensor) in enumerate(tqdm(train_dataloader), start=1):
            # todo  5个核心
            # 前向传播
            logits = my_bert_model(batch_texts_tensor)
            # 计算损失
            loss = loss_fn(logits, batch_labels_tensor)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 额外计算日志变量
            total_loss += loss.item()
            batch_cnt += 1
            all_pred_labels.extend(logits.argmax(dim=-1).tolist())
            all_true_labels.extend(batch_labels_tensor.tolist())

            if index % 10 == 0 or index == len(train_dataloader):
                print(f'第{epoch}轮，第{index}批，损失为：{total_loss / batch_cnt}')
                print(f'第{epoch}轮，第{index}批，准确率为：{accuracy_score(all_true_labels, all_pred_labels)}')
                print(f'第{epoch}轮，第{index}批，精确率为：{precision_score(all_true_labels, all_pred_labels, average="macro")}')
                print(f'第{epoch}轮，第{index}批，召回率为：{recall_score(all_true_labels, all_pred_labels, average="macro")}')
                print(f'第{epoch}轮，第{index}批，F1为：{f1_score(all_true_labels, all_pred_labels, average="macro")}')
                print(f'第{epoch}轮，第{index}批，用时：{time.time() - start_time}秒')
                # 清空日志变量
                total_loss, batch_cnt = 0, 0
                all_pred_labels, all_true_labels = [], []


    # todo  1个保存
    torch.save(my_bert_model.state_dict(), config.bert_classifier_model_save_path)

if __name__ == '__main__':
    model_train()