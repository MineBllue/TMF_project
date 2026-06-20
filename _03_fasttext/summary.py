"""
随机森林
封装API接口
1: 加载配置文件、训练好的 TF-IDF 转换器与随机森林模型
2:预测方法:
先对传入的json进行分词, join拼接,
通过tfidf做 拼接好的词转化成向量,
通过模型对向量进行预测
得到 数字标签映射为业务分类名称
拼接到json中返回

web服务器知识
url四部分组成: 协议部分/ip地址:端口号/资源路径部分?查询参数部分
请求方式包括post/get等

后端框架接触到flask,fastapi
其中都是 导包、创建应用实例、定义接口、启动服务

1:导包
2:创建对象,Flask,fastapi
3:定义接口
Flask的方式 @app.route(接口路径, 方法类别)
fastapi 的方式 @app.post(接口路径 )
def 方法(入参):
 返回
4:启动服务,包括ip+端口

其中区别
fastapi 异步高并发、自带自动接口文档，新项目更推荐使用

提示词生成代码
通过提示词生成对应的前端代码,和后端flask,fastapi代码,
通过页面调用 就可以了

fasttext
我们通过两种分词方式,
一种是按字,通过 单字切割分词 ,
一种是按词,通过jieba分词

两种训练方式,一种是默认的参数训练 一种是自动寻找最优的参数 但是都是一样的思路去做的
一种是默认的参数训练
一种是自动寻找最优的参数
但是都是一样的思路去做的

数据处理
将数据转化成需要的格式
fasttext本次使用的是单标签多分类,
__label__标签名  分词后的文本

细节思路(可以不背)
1:with open 读取数据和写入数据
2:读取每一行数据,
其中切分成label和word,
将label转化成英文名字,
将句子进行分词并且拼接
得到了对应的fasttext的格式
进行保存

训练模型
默认参数
使用fasttext.train_supervised(训练集路径,verbose=2)
自动参数就是将 autotuneValidationFile 指向评估集地址

保存模型
model.save_model(模型保存路径)

模型评估
先 fasttext.load_model(加载模型)
再model.test(测试集的路径) 得到测试的数据
其中会看到样本数量,精确率,召回率等

模型预测
model.predict(输入分词后的文本) 得到预测类别.

API页面
1:需要写预测方法,
加载于当前的fasttext最优模型,
预测对应传入的问题,并且拼接到json中,返回
通过提示词工程生成前端和flask和fastapi,接口调用 预测方法
就可以了.

Streamlit、Flask、FastAPI、fasttext
  Streamlit是专为数据科学和机器学习设计的交互式web应用。适用机器学习模型演示。
  Flask 是一个用 Python 编写的轻量级 Web 应用框架。适用于构建轻量级 API、个人开发。
  FastAPI 是一个高性能的 Python Web 框架，它是专为 构建 API 而生。
  fasttext是一款用于训练词向量和文本分类的工具。它的核心思想就是把词拆解成一组字符级的n-gram,并为每个n-gram学习一个向量。这样即使有一个词从未见过，只要它的子词出现过，都可以组成合理的向量。
  fasttext模型构建并训练使用流程分为以下几个步骤：
      1、创建配置文件
      2、数据处理，对数据进行分词并拼接成fasttext的数据格式 (__label__ 标签 词1 词2 ...)
      3、通过fasttext.train_supervised对模型进行训练
      4、save_model保存模型
      5、fasttext.load_model加载模型
      6、model.test评估模型
      7、最后使用model.predict进行模型预测
      8、最后可以使用web应用框架构建前端页面，使用训练好的模型。
  fasttext模型训练可以设置两种调参方式，分别是自动调参和默认调参，它们的区别就是：
    默认调参是直接使用fasttext预设的固定超参数。
    自动调参只需提供一份验证集，它就会自动优化超参数。
"""
"""
铭哥：   流利，讲的也很细，有自己的理解
麦丽华： 讲的也挺流利的，但是知识点没讲全
王俊：   拉完了
刘明儒：  
"""