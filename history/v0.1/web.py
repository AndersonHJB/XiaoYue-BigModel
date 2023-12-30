import gradio as gr
import service  # 项目服务

s = service.Service()  # 项目服务初始化

with gr.Blocks() as demo:

    gr.HTML("""<h1 align="center">小悦 v0.1 - 纯 LLM 驱动</h1>""")  # 项目标题

    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])


    def respond(message, chat_history):
        bot_message = s.simple_answer(message, chat_history)  # 改变 chat 信息的传入接口

        chat_history.append((message, bot_message))

        return "", chat_history


    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")  # 启动 gradio 分享，允许外网访问。
