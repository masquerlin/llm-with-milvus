import openai
import gradio as gr
from milvus_server import get_data_milvus, update_milvus
pre_prompt = """
# CONTEXT(上下文) 
我想根据用户的问题和提供的相关信息,用户问题的回答
###
# OBJECTIVE(目标) 
根据用户的问题和提供的相关信息, 进行用户问题的回答

# ATTENTION(注意点) 
1、回答尽量利用提供的相关信息
# STYLE(风格)
幽默，详细，精准
###
# TONE(语调) 
humorous, professional
###
# AUDIENCE(受众) 
想了解相关关联信息的用户
###
# RESPONSE(响应) 
尽量利用相关关联的信息来对用户问题进行回答

######
用户的问题: {}

提供的相关信息: {}

######
Answer:
"""
def run_llm(prompt, history=[], functions=[], sys_prompt= f"You are an useful AI assistant that helps people solve the problem step by step."):

    openai.api_base = "http://localhost:8009/v1"
    openai.api_key = 'none'
    openai.api_request_timeout = 1 # 设置超时时间为10秒
    messages=[{"role": "system", "content": sys_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": "" + prompt})
    response = openai.ChatCompletion.create(
        model = "qwen_code",
        messages = messages,
        temperature=0.65,
        max_tokens=2048,
        functions=functions
        )
    data_res = response['choices'][0]['message']['content']
    function_call = response['choices'][0]['message']['function_call']
    return data_res, function_call
def model_chat(query, collection_name, history=[]):
    responses = list()
    if len(history) > 0:
        for history_msg in history:
            responses.append(history_msg)
    yield responses
    mylist = list()
    mylist.append(query)
    vector_recall = get_data_milvus(query, collection_name)
    prompt = pre_prompt.format(query, vector_recall)
    answer, functions_call = run_llm(prompt=prompt)
    mylist.append(answer)
    
    
    responses.append(mylist)
    yield responses
def clear_session():
    return '', [], None

def main():
    with gr.Blocks(css="footer {visibility: hidden}",theme=gr.themes.Soft()) as demo:
        gr.Markdown("""<center><font size=10>Code analyze</center>""")
        with gr.Row(equal_height=False):
            file_output = gr.Files(height=200)
            name_box = gr.Textbox()
            output_list = gr.List()
            
        with gr.Row(equal_height=False):
            chatbot = gr.Chatbot(label='智能bot回答',scale=1)
        with gr.Row():
            textbox = gr.Textbox(lines=3, label='提出你的问题吧')
        with gr.Row():
            with gr.Column():
                clear_history = gr.Button("🧹 clear")
                sumbit = gr.Button("🚀 submit")
        file_output.upload(update_milvus, inputs= [file_output, name_box],outputs=[output_list])
        sumbit.click(model_chat, [textbox, name_box, chatbot], [chatbot])
        clear_history.click(fn=clear_session,
                            inputs=[],
                            outputs=[textbox, chatbot])
    demo.queue(api_open=False).launch(server_name='localhost', server_port=7860,max_threads=10, height=800, share=False)
if __name__ == "__main__":
    main()