from config import Config
config = Config()
import fasttext
import jieba

# 加载模型
model = fasttext.load_model(config.fasttext_word_auto_model_path)

def predict_fun(data):
    # 对输入的文本进行分词
    words = ' '.join(jieba.lcut(data['text']))
    # 模型预测
    label = model.predict(words)
    # 获取预测结果
    y_predict_class_name = label[0][0][9:]
    # 返回结果
    data['class_name'] = y_predict_class_name
    return data




if __name__ == '__main__':
    # 输入文本
    text = input('请输入文本：')
    data = {'text': text}
    result = predict_fun(data)
    print(result)