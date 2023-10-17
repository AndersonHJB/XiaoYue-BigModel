import prompt
import util
import config


class Service:
    def __init__(self):
        super(Service, self).__init__()
        self.util = util.Util()
        self.configs = config.ConfigParser()

    def simple_answer(self, message, history):
        # 1.组装系统提示，历史对话，用户当前问题
        system_prompt = prompt.SIMPLE_SYSTEM_PROMPT

        messages = self.util.concat_chat_message(system_prompt, history, message)

        # 2. 去调用 OpenAI 的接口完成任务
        response = self.util.ChatOpenAI(messages)

        return response.content
