import os
from openai import OpenAI

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
    text = input('请输入新闻标题：')
    data = {'text': text}
    result = get_llm_response(data)
    print(result)