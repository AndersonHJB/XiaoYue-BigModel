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
        question_vector = self.util.EmbeddingM3E.encode(message)

        # 2. 向量去数据量检索
        documents = self.util.VDBMilvus.search_document("mobot", [question_vector], 3)

        # 3. 抑制置信度比较低的答案
        retrival = ""
        if len(documents) == 0:
            retrival = "没有找到相关的数据"

        for doc in documents:
            if float(doc["distance"]) < 600:
                retrival += f'问题:{doc["question"]} 答案:{doc["answer"]}\n'

        # 4. 组装提示，历史对话，用户当前问题
        system_prompt = prompt.GENERIC_SYSTEM_PROMPT
        user_prompt = f"历史对话: {history}\n\n知识库: {retrival}\n根据知识进行回答用户问题: {message}"

        # 5.  去调用 LLM 的接口进行汇总推理
        response = self.util.ChatLlama2.completion(system_prompt, user_prompt)

        # 6. 返回结果
        return response

