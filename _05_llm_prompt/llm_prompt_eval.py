import os
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

client = OpenAI(
    # api_key="sk-975783c3a4934192a6245f737670d217",
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

SYSTEM_PROMPT = """
你是一个专业的新闻分类助手。你的任务是将用户输入的新闻标题准确分类到以下10个指定类别中的一个：

finance, realty, stocks, education, science, society, politics, sports, game, entertainment

分类规则：
1. 仔细分析新闻标题的核心主题，从上述10个类别中选择最匹配的一个。
2. 如果新闻标题的内容无法归入上述10个类别中的任何一个，请默认返回 society。
3. 输出结果必须且只能是上述类别中的一个英文单词，严禁包含任何标点符号、解释性文字或其他装饰性输出。

新闻标题：{用户输入的新闻标题}
分类结果：
"""
def load_data(base_path):
    data_list = []
    for line in open(base_path, 'r', encoding='utf-8'):
        text,label_str = line.strip().split('\t')
        label = int(label_str)
        data_list.append(text, label)
    texts, labels = zip(*data_list)
    return texts, labels

def llm_prompt_eval(path):
    texts, labels = load_data(path)
    all_pred_labels = []
    for q in texts:
        y_pred_label = get_llm_response(q)
        all_pred_labels.append(y_pred_label)

    acc = accuracy_score(all_true_labels, all_perd_labels)
    pre = precision_score(all_true_labels, all_perd_labels, average="macro", zero_division=0)
    recall = recall_score(all_true_labels, all_perd_labels, average="macro", zero_division=0)
    f1 = f1_score(all_true_labels, all_perd_labels, average="macro", zero_division=0)

    print(f'准确率为：{acc}, 精确率为：{pre}, 召回率为：{recall}, F1为：{f1}')


def get_llm_response(query):
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query['text']},
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )

    y_predict = response.choices[0].message.content
    data['predict_class'] = y_predict

    return data



if __name__ == '__main__':
    llm_prompt_eval()