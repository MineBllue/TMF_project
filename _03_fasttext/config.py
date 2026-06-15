class Config():
    def __init__(self):
        # 原始数据路径
        self.data_path = '/Users/wangjie/PycharmProjects/TMP_Project'
        self.train_path = self.data_path + '/_01_data/train.txt'
        self.test_path = self.data_path + '/_01_data/test.txt'
        self.dev_path = self.data_path + '/_01_data/dev.txt'
        self.class_path = self.data_path + '/_01_data/class.txt'
        self.stop_word_path = self.data_path + '/_01_data/stopwords.txt'
        # fasttext处理后数据存放路径
        self.process_train_path_chars = self.data_path + '/_03_fasttext/process_data/train_chars.txt'
        self.process_test_path_chars = self.data_path + '/_03_fasttext/process_data/test_chars.txt'
        self.process_dev_path_chars = self.data_path + '/_03_fasttext/process_data/dev_chars.txt'

        self.process_train_path_words = self.data_path + '/_03_fasttext/process_data/train_words.txt'
        self.process_test_path_words = self.data_path + '/_03_fasttext/process_data/test_words.txt'
        self.process_dev_path_words = self.data_path + '/_03_fasttext/process_data/dev_words.txt'
        # fasttext模型保存路径
        self.fasttext_char_default_model_path = self.data_path + '/_03_fasttext/model/fasttext_char_default.bin'
        self.fasttext_char_auto_model_path = self.data_path + '/_03_fasttext/model/fasttext_char_auto.bin'
        self.fasttext_word_default_model_path = self.data_path + '/_03_fasttext/model/fasttext_word_default.bin'
        self.fasttext_word_auto_model_path = self.data_path + '/_03_fasttext/model/fasttext_word_auto.bin'
        # 分类词表
        self.id2class = {index: class_name for index, class_name in enumerate(open(self.class_path, 'r', encoding='utf8'))}


if __name__ == '__main__':
    config = Config()
    print(config.process_train_path_chars)
    print(config.process_test_path_chars)
    print(config.process_dev_path_chars)
    print(config.process_train_path_words)
    print(config.process_test_path_words)
    print(config.process_dev_path_words)
