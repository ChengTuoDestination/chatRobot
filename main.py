import streamlit as st
import mdtex2html
from mymodel import model, tokenizer

history = []
chatbot = []

def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

def format_chatbot(question, answer):
    html = '<div style="display: flex; flex-direction: column; align-items: flex-start;">'
    html += f'<div style="background-color: #eee; border-radius: 5px; padding: 8px 12px; margin-bottom: 8px;"><span style="font-weight: bold;">You:</span> {question}</div>'
    html += f'<div style="background-color: #f4f4f4; border-radius: 5px; padding: 8px 12px; margin-bottom: 8px;"><span style="font-weight: bold;">ChatGLM-6B:</span> {answer}</div>'
    html += '</div>'
    return html

def predict(input_text, history, chatbot, max_length, top_p, temperature):
    input_text = input_text.strip()
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    sample_outputs = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=top_p,
        temperature=temperature,
        num_return_sequences=1,
    )
    for i, sample_output in enumerate(sample_outputs):
        output = tokenizer.decode(sample_output, skip_special_tokens=True)
        chatbot.append((input_text, output[len(input_text)+1:]))#修改回答展示，去掉 question 部分
    new_history = history + chatbot
    st.session_state.history = new_history
    output_text = ""
    for question, answer in new_history:
        output_text += format_chatbot(question, answer)
    return chatbot, new_history, output_text

col1, col2, col3 = st.columns([1, 1, 1])
max_length = st.sidebar.slider("Maximum length", 0, 4096, 2048, 1, key="max_length_slider")
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.7, 0.1, key="top_p_slider")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.95, 0.01, key="temperature_slider")

# 初始化输出记录列表
if "output" not in st.session_state:
    st.session_state.output = []
output_text = st.empty()
# 初始化输入框记录列表
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
# 显示历史聊天记录的 HTML
if "history" in st.session_state:
    output_text.markdown(st.session_state.history_html, unsafe_allow_html=True)

with st.sidebar:
    input_text = st.text_area("Input", height=100, value=st.session_state.input_text)
    st.session_state.input_text = ""
    if st.button("Submit"):
        chatbot, history, output = predict(input_text, history, chatbot, max_length, top_p, temperature)
        # 更新历史聊天记录的 HTML，并将其保存到 Streamlit 的 session state 中
        st.session_state.history_html = output
        st.session_state.output.append(output)
        st.session_state.input_text = ""
        input_text = ""
        del st.session_state.input_text
    
# 显示输出记录的 HTML
if st.session_state.output:
    output_html = ""
    for o in st.session_state.output:
        output_html += o
        #output_html=st.write(o)
    output_text.markdown(output_html, unsafe_allow_html=True)   
