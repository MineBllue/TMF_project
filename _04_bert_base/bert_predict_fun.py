from config import Config
config = Config()
from bert_classifier_model import MyBertClassifier
import torch
from dataloader_utils import build_dataloader
bert_model = MyBertClassifier()
bert_model.load_state_dict(torch.load(config.bert_classifier_model_save_path))

def predict_fun(data):
    text = data['text']
    text_tensor = config.bert_tokenizer(text, return_tensors='pt', max_length=config.max_len, padding=True, truncation=True)
    # 模型预测拿到样本对应的10个分数
    logits = bert_model(text_tensor)
    y_predict = torch.argmax(logits, dim=-1).item()
    y_predict_class_name = config.id2class[y_predict]
    data['predict_class'] = y_predict_class_name
    return data




if __name__ == '__main__':
    # 输入文本
    text = input('请输入文本：')
    data = {'text': text}
    result = predict_fun(data)
    print(result)