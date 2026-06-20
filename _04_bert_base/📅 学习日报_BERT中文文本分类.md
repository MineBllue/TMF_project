# 📅 2026-06-16 学习日报 | BERT 中文文本分类实战
> 📁 `_04_bert_base` | 共 4 个文件

## 📊 今日概要
基于 HuggingFace `transformers` 库加载 `bert-base-chinese` 预训练模型，构建一个 **10 类中文新闻文本分类器**。核心思路：冻结 BERT 作为特征提取器，在其上方接一个 Linear 层将 768 维 `pooler_output` 映射到 10 分类。完成了从数据加载 → Dataset 封装 → collate_fn tokenization → 自定义模型 → 训练循环（含每 10 批打印 acc/precision/recall/F1）的完整 pipeline，最终模型保存为 409MB 的 `.bin` 文件。

## 💡 API / 知识点
| 标记 | API / 函数 | 作用 | 关键参数 | 文件 |
|------|------------|------|----------|------|
| 🔴 | `BertTokenizer.from_pretrained()` | 加载 BERT 分词器 | `path/to/vocab` | `config.py` |
| 🔴 | `BertModel.from_pretrained()` | 加载 BERT 预训练模型权重 | `path/to/model` | `config.py` |
| 🔴 | `BertConfig.from_pretrained()` | 加载 BERT 配置（hidden_size=768 等） | `path/to/config` | `config.py` |
| 🔴 | `torch.utils.data.Dataset` | 自定义数据集基类 | 需重写 `__len__`, `__getitem__` | `dataloader_utils.py` |
| 🔴 | `torch.utils.data.DataLoader` | 批量数据迭代器 | `batch_size`, `shuffle`, `collate_fn` | `dataloader_utils.py` |
| 🟡 | `collate_fn` | 自定义批次组装函数 | 接收 batch list，返回 tensor | `dataloader_utils.py` |
| 🔴 | `tokenizer(texts, truncation, padding, max_length, return_tensors)` | 批量文本 → token ids + attention_mask | `truncation=True`, `padding=True`, `max_length`, `return_tensors='pt'` | `dataloader_utils.py` |
| 🔴 | `torch.nn.Linear(in, out)` | 线性变换层 | `in_features=768`, `out_features=10` | `bert_classifier_model.py` |
| 🟡 | `result.pooler_output` | BERT 输出的句子级表示（768 维） | 经 pooler 层处理后的 `[CLS]` 向量 | `bert_classifier_model.py` |
| 🔴 | `torch.nn.CrossEntropyLoss` | 多分类交叉熵损失 | `reduction='mean'` | `bert_evaluate_while_training.py` |
| 🔴 | `torch.optim.AdamW` | AdamW 优化器 | `params`, `lr=5e-5` | `bert_evaluate_while_training.py` |
| 🔴 | `optimizer.zero_grad()` / `loss.backward()` / `optimizer.step()` | 标准训练三步走 | — | `bert_evaluate_while_training.py` |
| 🟡 | `logits.argmax(dim=-1)` | 取 logits 最大索引作为预测类别 | `dim=-1` 沿最后一维取 argmax | `bert_evaluate_while_training.py` |
| 🔴 | `accuracy_score` / `precision_score` / `recall_score` / `f1_score` (sklearn) | 分类评估指标 | `average='macro'` | `bert_evaluate_while_training.py` |
| 🔴 | `torch.save(model.state_dict(), path)` | 仅保存模型参数（推荐方式） | `state_dict()` | `bert_evaluate_while_training.py` |

## 🛠 代码实操
| 文件 | 做了什么 | 核心步骤（每步用 → 连接） |
|------|----------|---------------------------|
| `config.py` | 全局配置类 | (1) 指定数据/模型路径 → (2) 加载 tokenizer + bert_model + bert_config → (3) 定义超参数（epochs=10, bs=64, lr=5e-5, max_len=24, device='mps'） |
| `dataloader_utils.py` | 数据加载工具 | (1) `load_data_list` 读文本→ [(text,label),...] → (2) `MyDataset` 继承 Dataset 重写3方法 → (3) `my_collate_fn` 用 tokenizer 把 batch 文本转 tensor → (4) `build_dataloader` 组装 train/test/dev 三个 DataLoader |
| `bert_classifier_model.py` | 自定义 BERT 分类模型 | (1) `MyBertClassifier` 继承 nn.Module → (2) `__init__` 挂载 config.bert_model + Linear(768→10) → (3) `forward` 调用 `self.bert(**x)` → 取 `.pooler_output` → 过 Linear 得 logits |
| `bert_evaluate_while_training.py` | 训练 + 边训练边评估 | (1) 4个准备（数据/模型/损失/优化器）→ (2) 外层 epochs 循环 → (3) 内层 batch 循环执行5步核心（前向→算loss→zero_grad→backward→step）→ (4) 每10批计算并打印 acc/precision/recall/F1 → (5) `torch.save` 保存模型 |

## ⚠️ 注意事项
| 标记 | 注意点 | 说明 | 文件 |
|------|--------|------|------|
| 🔴重点 | `collate_fn` 是 DataLoader 的核心扩展点 | 接收一个 list of `(text, label)` 元组，负责把一个 batch 的原始数据转为模型可接受的 tensor。这里用 `zip(*batch)` 解包，再用 tokenizer 批量编码文本 | `dataloader_utils.py:30-37` |
| 🔴重点 | tokenizer 返回的是 `BatchEncoding` (dict)，需 `**` 解包传入 BERT | `self.bert(**x)` 将 dict 解包为 `input_ids`, `attention_mask` 等关键字参数。如果直接传 `x` 会报错 | `bert_classifier_model.py:16` |
| 🟡难点 | BERT 输出的 `pooler_output` vs `last_hidden_state` | `last_hidden_state` 是每个 token 的 768 维向量 (batch, seq_len, 768)；`pooler_output` 是对 `[CLS]` 做 tanh+Linear 后的句子级表示 (batch, 768)。文本分类用后者 | `bert_classifier_model.py:19` |
| 🟠易混淆 | CrossEntropyLoss 自带 softmax | logits 不需要先过 softmax，`CrossEntropyLoss` 内部会自动做 log_softmax + NLLLoss。如果多加了 softmax 会导致梯度消失 | `bert_classifier_model.py:20` 注释 |
| 🟠易混淆 | `shuffle=True` 只对训练集 | 训练集需要打乱防止模型记忆顺序，测试集和验证集不需要打乱，否则评估结果不可复现 | `dataloader_utils.py:45,50,55` |
| 🔴重点 | `optimizer.zero_grad()` 必须放在 `loss.backward()` 之前 | 如果顺序写反，梯度会累积（默认累加），导致模型无法收敛。每批数据训练前必须清空上一批的梯度 | `bert_evaluate_while_training.py:37-40` |
| 🟡难点 | `average='macro'` 的含义 | 对每个类别分别计算指标后取平均，不考虑类别样本数差异。适合类别不均衡场景。与之相对的是 `'micro'`（全局计算）和 `'weighted'`（按样本数加权） | `bert_evaluate_while_training.py:51-53` |
| 🔴重点 | `torch.save(model.state_dict(), path)` vs `torch.save(model, path)` | 前者只保存参数（推荐），后者保存整个模型对象（依赖代码结构，不灵活）。加载时需先实例化模型再 `load_state_dict` | `bert_evaluate_while_training.py:61` |
| 🔴重点 | 学习率 5e-5 是 BERT 微调的标配 | BERT 预训练权重已经很好了，微调时 lr 不宜过大（通常 2e-5 ~ 5e-5），否则会破坏预训练知识（catastrophic forgetting） | `config.py:30` |
| 🟡难点 | `result.hidden_states` 默认不返回 | 需要在 `BertConfig` 或 `BertModel.from_pretrained` 中设置 `output_hidden_states=True` 才会输出中间层 hidden states | `bert_classifier_model.py:17` 注释 |

## ❓ 课后回顾
- [ ] 手动推导 `CrossEntropyLoss = log_softmax + NLLLoss`，理解为什么 logits 不需要先 softmax
- [ ] 对比 `result.pooler_output` 和 `result.last_hidden_state[:, 0, :]`（直接取 CLS 向量）的分类效果差异
- [ ] 尝试修改 `max_len`（24 → 48/128），观察训练速度和准确率变化
- [ ] 在 `collate_fn` 中打印一次 tokenizer 的输出结构，理解 `input_ids`、`attention_mask`、`token_type_ids` 各自的含义
- [ ] 把 `average='macro'` 改成 `'micro'` 和 `'weighted'`，对比指标差异

## 🎯 今日面试题

---

**第 1 题：BERT 模型的输入输出结构是怎样的？`pooler_output` 和 `last_hidden_state` 有什么区别？**

> 💬 面试话术：先说 BERT 输入三件套（input_ids / attention_mask / token_type_ids），再说两种输出的维度和用途，最后强调文本分类为什么用 pooler_output。

📝 **参考答案：**

**输入（3 个关键字段，封装在 BatchEncoding 中）：**
- `input_ids`: (batch, seq_len)，每个 token 对应的词典 ID，`[CLS]`=101, `[SEP]`=102
- `attention_mask`: (batch, seq_len)，1=有效 token，0=padding（让模型忽略填充位）
- `token_type_ids`: (batch, seq_len)，句子分段标识，单句分类全为 0

**输出（BaseModelOutputWithPooling）：**
- `last_hidden_state`: (batch, seq_len, 768) — **每个 token** 的上下文表示，适合序列标注任务（如 NER）
- `pooler_output`: (batch, 768) — **整个句子** 的语义表示，由 `[CLS]` token 经 `tanh` + `Linear(768→768)` 得到，专为分类任务设计

**文本分类为什么用 `pooler_output`？**
- `pooler_output` 已经过一层非线性变换，聚合了整句的语义信息
- 维度固定为 768，不受 `max_len` 影响，方便接分类头
- 也可以直接用 `last_hidden_state[:, 0, :]`（CLS 原始输出），但 `pooler_output` 是 BERT 官方推荐的分类输入

---

**第 2 题：为什么 BERT 微调时学习率要设成 5e-5 这么小？lr 太大的后果是什么？**

> 💬 面试话术：从"预训练权重已经是局部最优解"切入，引出 catastrophic forgetting 概念，最后给 lr 的经验范围。

📝 **参考答案：**

**为什么要小学习率：**
1. **BERT 预训练权重已经是极好的语义特征提取器**，包含丰富的语言知识（语法、语义、常识），微调只是在目标任务上做小幅调整
2. 如果 lr 过大（如 1e-3），梯度更新步长太大，会快速偏离预训练分布，导致**灾难性遗忘**（catastrophic forgetting）——模型忘记了预训练阶段学到的通用语言知识
3. BERT 论文官方推荐的微调 lr 范围：**2e-5 ~ 5e-5**，这是一个经过大量实验验证的安全区间

**lr 太大的后果：**
- 训练 loss 剧烈震荡，难以收敛
- 验证集指标先升后降（过拟合）
- 极端情况下，模型退化到不如随机初始化的效果

**lr 太小（如 1e-6）的后果：**
- 收敛速度极慢，需要更多 epochs
- 可能陷入局部最优，无法充分适应目标任务

---

**第 3 题：PyTorch 中 `collate_fn` 的作用是什么？为什么在 NLP 任务中几乎必须自定义它？**

> 💬 面试话术：先说 DataLoader 默认行为，再讲 NLP 数据的特殊性（变长文本需要 tokenization），最后给出自定义 collate_fn 的模板写法。

📝 **参考答案：**

**`collate_fn` 的作用：**
- DataLoader 从 Dataset 中取 `batch_size` 个样本后，调用 `collate_fn` 将这些样本**拼装成一个 batch tensor**
- 默认的 `collate_fn` 只能处理等长的数值型数据（如图像），直接把 list 沿第 0 维 stack

**为什么 NLP 任务必须自定义：**
1. **文本长度不固定**：一个 batch 中有"今天天气真好"（6 字）和"人工智能是计算机科学的一个分支..."（20+ 字），默认 collate 无法处理不等长
2. **需要 tokenization + padding**：文本要先转为 token ids，再 padding 到统一长度（本项目中 `max_len=24`）
3. **需要生成 attention_mask**：标记哪些是真实 token、哪些是 padding，防止模型关注无意义的填充位
4. **文本和标签是不同数据类型**：文本是 str，标签是 int，需要分别处理

**本项目中的标准写法：**
```python
def my_collate_fn(batch):
    texts, labels = zip(*batch)  # 解包 [(text,label), ...]
    batch_texts_tensor = tokenizer(
        list(texts), truncation=True, padding=True,
        max_length=24, return_tensors='pt'
    )
    batch_labels_tensor = torch.tensor(labels)
    return batch_texts_tensor, batch_labels_tensor
```

---
> 📝 自动生成 | 分析 4 个文件
