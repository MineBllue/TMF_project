from config import Config
from dataloader_utils import build_dataloader
from bert_classifier_model import MyBertClassifier
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import time

# 斌子法则14251
config = Config()

def model_train():
    train_dataloader,test_dataloader,dev_dataloader = build_dataloader()
    my_bert_model = MyBertClassifier()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(my_bert_model.parameters(), lr=config.lr)
    for epoch in range(config.epochs):
        for batch_texts_tensor,batch_labels_tensor in tqdm(train_dataloader):
            # 5个核心
            logits = my_bert_model(batch_texts_tensor)
            loss = loss_fn(logits, batch_labels_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(my_bert_model.state_dict(), config.bert_classifier_model_save_path)



if __name__ == '__main__':
    model_train()