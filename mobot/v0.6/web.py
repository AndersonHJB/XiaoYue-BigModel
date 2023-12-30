import gradio as gr
import service

s = service.Service()

with gr.Blocks() as demo:

    gr.HTML("""<h1 align="center">小墨 v0.6 - 具有自我思维的智能体</h1>""")

    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])


    def respond(message, chat_history):
        bot_message = s.reasoning_action_answer(message, chat_history)

        chat_history.append((message, bot_message))

        return "", chat_history


    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")
