import prompt
import util
import config


class Service:
    def __init__(self):
        super(Service, self).__init__()
        self.util = util.Util()
        self.configs = config.ConfigParser()

    def retrival_inference_answer(self, message, history):

        # 0. 获取用户真正的答案
        system_prompt = prompt.REASONING_QUESTION_SYSTEM_PROMPT
        user_prompt = f"当前问题提问: {message}\n实际上用户真正想问的问题是:"
        messages = self.util.concat_chat_message(system_prompt, history, user_prompt)
        # 替换问题成推理出来的，用于后续的任务
        message = self.util.ChatOpenAI(messages).content

        # 1. 把用户问题的文本向量向量
        vector = self.util.embed_query(message)

        # 2. 向量去数据量检索向量数据库
        documents = self.util.similarity_search(vector)

        # 3. 抑制置信度比较低的答案
        retrival = ""
        if len(documents['matches']) == 0:
            retrival = "没有找到相关的数据"

        for doc in documents['matches']:
            if float(doc["score"]) > 0.75:
                retrival += f'问题:{doc["metadata"]["question"]} 答案:{doc["metadata"]["answer"]}\n'

        # 4. 对于时效性问题采用搜索引擎检索
        record = self.util.search_engine(message)
        result = ""
        for item in record.get('results'):
            result += f"title: {item['title']}\ncontent: {item['content']}\n\n"

        # 5. 组装系统提示，历史对话，用户当前问题
        system_prompt = prompt.GENERIC_SYSTEM_PROMPT

        user_prompt = f"""
            知识库:\n{retrival}\n
            搜索引擎结果:\n{result}\n
            请你根据知识库、搜索引擎结果回答用户问题:\n{message}
        """

        messages = self.util.concat_chat_message(system_prompt, history, user_prompt)

        # 5.  去调用 OpenAI 的接口进行汇总推理
        response = self.util.ChatOpenAI(messages)

        # 6. 返回结果
        return response.content

