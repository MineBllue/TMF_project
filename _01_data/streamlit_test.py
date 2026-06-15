import streamlit as st

# 设置标题
st.title("投满分分类项目")
# 获取用户输入的文本
text = st.text_input("请输入要查询分类的文本")
if text:
    st.write(f"你输入的文本是:{text}")
    print(text)