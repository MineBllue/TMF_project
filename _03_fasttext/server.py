"""
文本分类预测服务 - FastAPI 后端
提供文本分类预测 API、历史记录管理、前端页面服务
"""
from __future__ import annotations

import json
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

import jieba
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import fasttext

from config import Config

# ── 配置 ──────────────────────────────────────────────
config = Config()
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
MAX_HISTORY = 200  # 最多保留的历史记录数

# ── 加载模型 ──────────────────────────────────────────
model = fasttext.load_model(config.fasttext_word_auto_model_path)

# ── 加载类别映射 ──────────────────────────────────────
with open(config.class_path, "r", encoding="utf-8") as f:
    _classes = [line.strip() for line in f if line.strip()]
id2class: dict[int, str] = {i: name for i, name in enumerate(_classes)}
class2id: dict[str, int] = {name: i for i, name in enumerate(_classes)}
CLASS_NAMES: list[str] = _classes

# ── FastAPI 应用 ──────────────────────────────────────
app = FastAPI(title="文本分类预测服务", version="1.0.0")


# ── 请求/响应模型 ─────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="待分类文本")


class PredictResult(BaseModel):
    id: str = Field(..., description="记录唯一ID")
    text: str
    class_name: str
    class_id: int
    confidence: float
    probabilities: dict[str, float]
    timestamp: str
    cost_ms: float


class HistoryResponse(BaseModel):
    total: int
    items: list[PredictResult]


# ── 内存历史存储 ──────────────────────────────────────
_history: deque[PredictResult] = deque(maxlen=MAX_HISTORY)


def predict_fun(text: str) -> tuple[str, int, float, dict[str, float]]:
    """对输入文本进行分类预测，返回 (类别名, 类别ID, 置信度, 所有类别概率)。"""
    # fasttext 输入：分词后用空格连接
    words = " ".join(jieba.lcut(text))

    # fasttext predict 返回 (labels, probs)，labels 格式为 __label__类别名
    k = len(CLASS_NAMES)
    labels, probs = model.predict(words, k=k)

    # 解析 top-1 结果（label 格式为 __label__类别名）
    top_label: str = labels[0]
    y_name = top_label.replace("__label__", "")
    y_idx = class2id.get(y_name, -1)
    confidence = float(probs[0])

    # 构建所有类别概率字典
    probabilities: dict[str, float] = {}
    for label, prob in zip(labels, probs):
        cls_name = label.replace("__label__", "")
        probabilities[cls_name] = round(float(prob), 4)

    return y_name, y_idx, confidence, probabilities


# ── API 路由 ──────────────────────────────────────────
@app.post("/api/predict", response_model=PredictResult)
async def api_predict(req: PredictRequest):
    """文本分类预测"""
    t0 = time.perf_counter()
    class_name, class_id, confidence, probabilities = predict_fun(req.text)
    cost_ms = round((time.perf_counter() - t0) * 1000, 2)

    result = PredictResult(
        id=uuid.uuid4().hex[:12],
        text=req.text,
        class_name=class_name,
        class_id=class_id,
        confidence=confidence,
        probabilities=probabilities,
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
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return """<!DOCTYPE html><html><body><h1>前端页面未找到</h1><p>请将 index.html 放入 static/ 目录</p></body></html>"""


# ── 启动入口 ──────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8088, log_level="info")
