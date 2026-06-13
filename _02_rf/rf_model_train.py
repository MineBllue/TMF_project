import time
import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import pickle
import time
from config import Config
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
config = Config()

# 1. 读取分词后的数据文件
df_data = pd.read_csv(config.process_train_path, sep='\t')
# print(df_data.head())
# 2. 单独获取特征和标签
x_words = df_data['words']
y_words = df_data['label']
# 3. 创建TF-IDF向量化器 文本特征->数值特征
# 3.1 提前把停用词放到列表中
stop_words = [line.strip() for line in open(config.stop_word_path, encoding='utf-8').readlines()]
# 3.2 创建TF-IDF向量化器 并设置停用词
tfidf = sklearn.feature_extraction.text.TfidfVectorizer(stop_words=stop_words)
# 3.3 训练向量化器  文本特征->数值特征
words_feature = tfidf.fit_transform(x_words)
# 4. 创建随机森林 模型训练
# 4.1 先切割数据
X_train, X_test, y_train, y_test = train_test_split(words_feature, y_words, test_size=0.2, random_state=42)
# 4.2 创建随机森林
model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, verbose=3, n_jobs=-1)

# 4.3 模型训练
start_time = time.time()
print('===开始训练===')
model.fit(X_train, y_train)
print(f'===训练结束，耗时{(time.time() - start_time):2f}s')
y_predict = model.predict(X_test)

# 模型评估
print(f'准确率：{accuracy_score(y_test, y_predict)}')
print(f"精确率：{precision_score(y_test, y_predict, average='macro')}")
print(f"召回率：{recall_score(y_test, y_predict, average='macro')}")
print(f"f1分数：{f1_score(y_test, y_predict, average='macro')}")

# 模型保存
joblib.dump(model, config.rf_save_model_path)
joblib.dump(tfidf, config.tfidf_save_path)
