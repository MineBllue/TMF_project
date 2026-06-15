from config import Config
config = Config()
import jieba

def process_data(base_path, process_path, is_jieba):
    # 判断是否是jieba分词
    if is_jieba:
        # 读取数据
        with open(base_path, 'r', encoding='utf8') as fr:
            with open(process_path, 'a', encoding='utf8') as fw:
                for line in fr.readlines():
                    # 先去除\n，然后切割成列表，然后拆包出text和label
                    text, label = line.strip().split('\t')
                    # 根据label索引获取对应的分类名
                    class_name = config.id2class[int(label)].strip()
                    words = " ".join(jieba.lcut(text))
                    # 拼接成fasttext需要的分类数据格式
                    ft_line = '__label__' + class_name + " " + words + '\n'
                    fw.write(ft_line)
    else:
        with open(base_path, 'r', encoding='utf8') as fr:
            with open(process_path, 'a', encoding='utf8') as fw:
                for line in fr.readlines():
                    # 先去除\n，然后切割成列表，然后拆包出text和label
                    text, label = line.strip().split('\t')
                    # 根据label索引获取对应的分类名
                    class_name = config.id2class[int(label)].strip()
                    words = " ".join(list(text))
                    # 拼接成fasttext需要的分类数据格式
                    ft_line = '__label__' + class_name + " " + words + '\n'
                    fw.write(ft_line)




if __name__ == '__main__':
    process_data(config.train_path, config.process_train_path_chars, is_jieba=False)
    process_data(config.train_path, config.process_train_path_words, is_jieba=True)

    process_data(config.test_path, config.process_test_path_chars, is_jieba=False)
    process_data(config.test_path, config.process_test_path_words, is_jieba=True)

    process_data(config.dev_path, config.process_dev_path_chars, is_jieba=False)
    process_data(config.dev_path, config.process_dev_path_words, is_jieba=True)