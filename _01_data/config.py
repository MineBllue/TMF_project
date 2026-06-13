class Config():
    def __init__(self):
        # 各种路径
        self.data_path = '/Users/wangjie/PycharmProjects/TMP_Project'
        self.train_path = self.data_path + '/_01_data/train.txt'
        self.test_path = self.data_path + '/_01_data/test.txt'
        self.dev_path = self.data_path + '/_01_data/dev.txt'


if __name__ == '__main__':
    config = Config()
    print(config.train_path)
    print(config.test_path)
    print(config.dev_path)
