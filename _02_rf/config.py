class Config():
    def __init__(self):
        # 原始数据路径
        self.data_path = '/Users/wangjie/PycharmProjects/TMP_Project'
        self.train_path = self.data_path + '/_01_data/train.txt'
        self.test_path = self.data_path + '/_01_data/test.txt'
        self.dev_path = self.data_path + '/_01_data/dev.txt'
        self.class_path = self.data_path + '/_01_data/class.txt'
        self.stop_word_path = self.data_path + '/_01_data/stopwords.txt'
        # 随机森林处理后数据存放路径
        self.process_train_path = self.data_path + '/_02_rf/process_data/train.txt'
        self.process_test_path = self.data_path + '/_02_rf/process_data/test.txt'
        self.process_dev_path = self.data_path + '/_02_rf/process_data/dev.txt'
        # 随机森林模型保存路径
        self.rf_save_model_path = self.data_path + '/_02_rf/model/rf.pkl'
        self.tfidf_save_path = self.data_path + '/_02_rf/model/tfidf.pkl'


if __name__ == '__main__':
    config = Config()
    print(config.train_path)
    print(config.test_path)
    print(config.dev_path)
