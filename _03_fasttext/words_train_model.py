import fasttext

from config import Config
config = Config()

# 模型训练
model = fasttext.train_supervised(input=config.process_train_path_words,verbose=3)
# 保存模型
model.save_model(config.fasttext_word_default_model_path)
print('===' * 10)
# 理论上应该分两个文件，先加载，这里直接评估
# model = fasttext.load_model(config.fasttext_word_default_model_path)
result = model.test(config.process_test_path_words)
# 打印结果 数据个数，精确率，召回率
print('模型评估结果是', result) #  (10000, 0.908, 0.908)
print('===' * 10)
# 模型预测
label = model.predict('“ 手机 钱包 ” 亮相 科博会')
print('模型预测结果是', label) # (('__label__science',), array([0.97062206]))
