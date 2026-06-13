# 导入时间模块，用于记录模型训练耗时
import time
# 导入joblib库，用于保存和加载机器学习模型（比pickle更适合大型numpy数组）
import joblib
# 导入pandas库并简写为pd，用于数据处理和分析
import pandas as pd
# 导入sklearn库，scikit-learn是Python的机器学习库
import sklearn
# 从sklearn.model_selection模块导入train_test_split函数，用于将数据集划分为训练集和测试集
from sklearn.model_selection import train_test_split
# 导入pickle模块，用于Python对象的序列化和反序列化
import pickle
# 再次导入时间模块（重复导入，可以删除）
import time
# 从当前目录的config.py文件导入Config配置类
from config import Config
# 从sklearn.metrics模块导入评估指标函数：准确率、召回率、精确率、F1分数
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# 创建Config类的实例，用于访问配置文件中的路径和参数
config = Config()

# ==================== 第一步：读取分词后的数据文件 ====================
# 使用pandas读取经过预处理的训练数据文件，文件以制表符(\t)分隔
# 数据应该包含'words'（分词后的文本）和'label'（标签）两列
df_data = pd.read_csv(config.process_train_path, sep='\t')

# ==================== 第二步：单独获取特征和标签 ====================
# 从数据框中提取'words'列作为文本特征（分词后的文本数据）
x_words = df_data['words']
# 从数据框中提取'label'列作为标签（分类目标变量）
y_words = df_data['label']

    # ==================== 第三步：创建TF-IDF向量化器（将文本特征转换为数值特征）====================
# 3.1 提前把停用词放到列表中
# 读取停用词文件，逐行读取并去除每行首尾的空白字符（如换行符、空格）
# 停用词是指在文本中频繁出现但对分类无意义的词（如"的"、"是"、"了"等）
stop_words = [line.strip() for line in open(config.stop_word_path, encoding='utf-8').readlines()]

# 3.2 创建TF-IDF向量化器并设置停用词
# TF-IDF（Term Frequency-Inverse Document Frequency）是一种文本特征提取方法
# 它将文本转换为数值向量，同时过滤掉停用词
tfidf = sklearn.feature_extraction.text.TfidfVectorizer(stop_words=stop_words)

# 3.3 训练向量化器，将文本特征转换为数值特征
# fit_transform方法会先学习词汇表（fit），然后将文本转换为TF-IDF矩阵（transform）
# 返回的是一个稀疏矩阵，每行代表一个样本，每列代表一个词的TF-IDF值
words_feature = tfidf.fit_transform(x_words)

# ==================== 第四步：创建随机森林模型并进行训练 ====================
# 4.1 先切割数据
# 将数据集按8:2的比例划分为训练集和测试集
# words_feature: 完整的TF-IDF特征矩阵
# y_words: 完整的标签数据
# test_size=0.2: 测试集占总数据的20%
# random_state=42: 设置随机种子，保证每次划分结果一致
X_train, X_test, y_train, y_test = train_test_split(words_feature, y_words, test_size=0.2, random_state=42)

# 4.2 创建随机森林分类器
# n_estimators=100: 随机森林中包含100棵决策树
# verbose=3: 设置详细输出级别，训练过程中会打印进度信息
# n_jobs=-1: 使用所有CPU核心并行训练，加快训练速度
model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, verbose=3, n_jobs=-1)

# 4.3 模型训练
# 记录训练开始时间，用于计算训练耗时
start_time = time.time()
print('===开始训练===')
# 调用fit方法进行模型训练，传入训练集特征和标签
# 随机森林会根据训练数据学习分类规则
model.fit(X_train, y_train)
# 打印训练完成信息，并计算训练耗时（保留2位小数）
print(f'===训练结束=== 耗时{(time.time() - start_time):2f}s')

# 使用训练好的模型对测试集进行预测
y_predict = model.predict(X_test)

# ==================== 第五步：模型评估 ====================
# 计算并打印准确率（Accuracy）：正确预测的样本数占总样本数的比例
print(f'准确率：{accuracy_score(y_test, y_predict)}')

# 计算并打印精确率（Precision）：使用macro平均方式（不考虑类别数量，直接求各类别精确率的平均值）
# 精确率 = TP / (TP + FP)，表示预测为正的样本中有多少是真的正样本
print(f"精确率：{precision_score(y_test, y_predict, average='macro')}")

# 计算并打印召回率（Recall）：使用macro平均方式
# 召回率 = TP / (TP + FN)，表示真正的正样本中有多少被预测出来了
print(f"召回率：{recall_score(y_test, y_predict, average='macro')}")

# 计算并打印F1分数：精确率和召回率的调和平均数，综合评估模型性能
# F1 = 2 * (Precision * Recall) / (Precision + Recall)
print(f"f1分数：{f1_score(y_test, y_predict, average='macro')}")

# ==================== 第六步：模型保存 ====================
# 使用joblib保存训练好的随机森林模型到指定路径
# joblib相比pickle更适合保存包含大型numpy数组的对象
joblib.dump(model, config.rf_save_model_path)

# 保存训练好的TF-IDF向量化器到指定路径
# 后续预测时需要使用相同的向量化器将文本转换为数值特征
joblib.dump(tfidf, config.tfidf_save_path)
