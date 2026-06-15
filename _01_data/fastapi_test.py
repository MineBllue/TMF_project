# 0.安装：pip install fastapi uvicorn
# #1.导包
from fastapi import FastAPI
from fastapi import Response
# 导入uvicorn用于启动服务器
import uvicorn

# 2.创建对象
app = FastAPI()

# 3.使用对象接收web请求，并给出响应结果
@app.get('/index.html')
def index():
    return Response('hello world')
# 根据请求资源路径，获取对应的资源return资源结果
@app.get('/avatar_anime.ico')
def img():
    with open('resource/avatar_anime.ico', 'rb') as f:
        data = f.read()
    return Response(data, media_type='image/x-icon')
# 4.启动服务器（使用uvicorn替代app.run）
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8082)
