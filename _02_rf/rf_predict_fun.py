from config import Config
config = Config()
import joblib
import jieba

# 加载tfidf
tfidf = joblib.load(config.tfidf_save_path)
# 加载rf模型
model = joblib.load(config.rf_save_model_path)

def predict_fun(data):
    # 对输入的文本进行分词
    words = ' '.join(jieba.lcut(data['text']))
    # 将文本特征转为数值特征
    x_words = tfidf.transform([words])
    # rf预测模型类别索引
    y_predict_list = model.predict(x_words)
    y_predict_idx = y_predict_list[0]
    # 根据索引去查询类别
    id2class = {index: class_name for index, class_name in enumerate(open(config.class_path, 'r', encoding='utf8'))}
    y_predict_class_name = id2class[y_predict_idx]

    # 返回结果
    data['class_name'] = y_predict_class_name
    return data




if __name__ == '__main__':
    # 输入文本
    text = input('请输入文本：')
    data = {'text': text}
    result = predict_fun(data)
    print(result)