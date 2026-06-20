"""
新闻分类预测服务 - DeepSeek API 后端
使用 DeepSeek 大模型进行新闻标题分类，提供 API、历史记录管理、前端页面服务
"""
from __future__ import annotations

import os
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

from openai import OpenAI
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# ── 配置 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
MAX_HISTORY = 200

# DeepSeek 分类类别
CLASS_NAMES = [
    "finance", "realty", "stocks", "education", "science",
    "society", "politics", "sports", "game", "entertainment"
]

SYSTEM_PROMPT = """你是一个专业的新闻分类助手。你的任务是将用户输入的新闻标题准确分类到以下10个指定类别中的一个：

finance, realty, stocks, education, science, society, politics, sports, game, entertainment

分类规则：
1. 仔细分析新闻标题的核心主题，从上述10个类别中选择最匹配的一个。
2. 如果新闻标题的内容无法归入上述10个类别中的任何一个，请默认返回 society。
3. 输出结果必须且只能是上述类别中的一个英文单词，严禁包含任何标点符号、解释性文字或其他装饰性输出。

新闻标题：{用户输入的新闻标题}
分类结果："""

# ── DeepSeek 客户端 ───────────────────────────────────
deepseek_client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

# ── FastAPI 应用 ──────────────────────────────────────
app = FastAPI(title="新闻分类预测服务 (DeepSeek)", version="1.0.0")


# ── 请求/响应模型 ─────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="待分类文本")


class PredictResult(BaseModel):
    id: str = Field(..., description="记录唯一ID")
    text: str
    class_name: str
    reasoning: Optional[str] = Field(None, description="模型推理过程")
    timestamp: str
    cost_ms: float


class HistoryResponse(BaseModel):
    total: int
    items: list[PredictResult]


# ── 内存历史存储 ──────────────────────────────────────
_history: deque[PredictResult] = deque(maxlen=MAX_HISTORY)


def predict_fun(text: str) -> tuple[str, Optional[str]]:
    """调用 DeepSeek API 对输入文本进行分类，返回 (类别名, 推理过程)。"""
    response = deepseek_client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )

    message = response.choices[0].message
    y_predict = message.content.strip() if message.content else "society"

    # 提取推理过程（DeepSeek 思考模式）
    reasoning = None
    if hasattr(message, 'reasoning_content') and message.reasoning_content:
        reasoning = message.reasoning_content

    # 验证分类结果是否在合法类别中，不在则默认 society
    if y_predict.lower() not in CLASS_NAMES:
        y_predict = "society"

    return y_predict.lower(), reasoning


# ── API 路由 ──────────────────────────────────────────
@app.post("/api/predict", response_model=PredictResult)
async def api_predict(req: PredictRequest):
    """新闻文本分类预测"""
    t0 = time.perf_counter()
    class_name, reasoning = predict_fun(req.text)
    cost_ms = round((time.perf_counter() - t0) * 1000, 2)

    result = PredictResult(
        id=uuid.uuid4().hex[:12],
        text=req.text,
        class_name=class_name,
        reasoning=reasoning,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        cost_ms=cost_ms,
    )

    _history.appendleft(result)
    return result


@app.get("/api/history", response_model=HistoryResponse)
async def api_history(q: Optional[str] = None, limit: int = 50):
    """获取预测历史，支持按类别名搜索"""
    items = list(_history)
    if q:
        items = [r for r in items if q.lower() in r.class_name.lower()]
    return HistoryResponse(total=len(items), items=items[:limit])


@app.delete("/api/history")
async def api_clear_history():
    """清空历史记录"""
    _history.clear()
    return JSONResponse({"ok": True})


@app.get("/api/categories")
async def api_categories():
    """获取所有类别列表"""
    return JSONResponse(CLASS_NAMES)


# ── 前端页面 ──────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "deepseek.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return """<!DOCTYPE html><html><body><h1>前端页面未找到</h1><p>请将 deepseek.html 放入 static/ 目录</p></body></html>"""


# ── 启动入口 ──────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8089, log_level="info")
