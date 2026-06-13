import pandas as pd
from torch.fx.experimental.unification.core import seq

from config import Config
config = Config()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
import joblib
import pickle

# 1. 读取测试集分词后数据，同时获取特征和标签
df_data = pd.read_csv(config.process_test_path, sep='\t')
x_words = df_data['words']
y_label = df_data['label']
# 2. 加载tfidf 然后把文本数据转为数值特征
try:
    tfidf = joblib.load(config.tfidf_save_path)
except Exception:
    try:
        with open(config.tfidf_save_path, 'rb') as f:
            tfidf = pickle.load(f)
    except Exception:
        raise ValueError(f"无法加载 TF-IDF 模型文件: {config.tfidf_save_path}")
new_x_words = tfidf.transform(x_words)
# 3. 加载rf模型 然后预测
try:
    model = joblib.load(config.rf_save_model_path)
except Exception:
    try:
        with open(config.rf_save_model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception:
        raise ValueError(f"无法加载 RF 模型文件: {config.rf_save_model_path}")
y_predict = model.predict(new_x_words)

# 4. 评估
print(f'准确率：{accuracy_score(y_label, y_predict)}')
print(f"精确率：{precision_score(y_label, y_predict, average='macro')}")
print(f"召回率：{recall_score(y_label, y_predict, average='macro')}")
print(f"f1分数：{f1_score(y_label, y_predict, average='macro')}")