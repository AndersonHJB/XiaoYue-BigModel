import prompt
import util
import config


class Service:
    def __init__(self):
        super(Service, self).__init__()
        self.util = util.Util()
        self.configs = config.ConfigParser()

    def retrival_inference_answer(self, message, history):

        # 1. 把用户问题的文本向量向量
        question_vector = self.util.EmbeddingOpenAI.embed_query(message)

        # 2. 向量去数据量检索向量数据库
        index = self.util.VDBPinecone.get_pinecone_index("mobot")
        documents = index.query(
            top_k=5,
            include_values=False,
            include_metadata=True,
            vector=question_vector
        )

        # 3. 抑制置信度比较低的答案
        retrival = ""
        if len(documents['matches']) == 0:
            retrival = "没有找到相关的数据"

        for doc in documents['matches']:
            if float(doc["score"]) > 0.75:
                retrival += f'问题:{doc["metadata"]["question"]} 答案:{doc["metadata"]["answer"]}\n'

        # 4. 组装系统提示，历史对话，用户当前问题
        system_prompt = prompt.GENERIC_SYSTEM_PROMPT

        user_prompt = f"历史对话:\n{history}\n\n知识库:\n{retrival}\n用户问题:\n{message}"

        messages = self.util.concat_chat_message(system_prompt, [], user_prompt)

        # 5.  去调用 OpenAI 的接口进行汇总推理
        response = self.util.ChatOpenAI(messages)

        # 6. 返回结果
        return response.content

