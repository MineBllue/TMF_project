#θ.安装：pip install flask
# #1.导包
from flask import Flask
#2.创建对象
app = Flask(__name__)

#3.使用对象接收web请求，并给出响应结果
@app.route('/index.html', methods=['GET'])
def index():
    return 'hello world'

@app.route('/avatar_anime.ico', methods=['GET'])
def index1():
    with open('avatar_anime.ico', 'rb') as f:
        data = f.read()
    return data
#根据请求资源路径，获取对应的资源return资源结果
#4.启动服务器
app.run(host='0.0.0.0', port=8081, debug=True)