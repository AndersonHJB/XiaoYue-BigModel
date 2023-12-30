import config
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from dashscope import Generation
from http import HTTPStatus


class Llama2:
    def __init__(self):
        super(Llama2, self).__init__()

        self.configs = config.ConfigParser()

        # 设置模型的名称（可在线下载，但是网络需要可以访问 HuggingFace ）
        # 或者本地的权重路径（提前离线下载）
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.configs.get(key='llama2')['model_name_or_path'],
            device_map='auto', torch_dtype=torch.float16)
        self.model = self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.configs.get(key='llama2')['model_name_or_path'],
            use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def completion(self, system_prompt, user_prompt):

        prompt = f"""
        <s>System: 
        {system_prompt}
        </s>
        <s>Human: 
        {user_prompt}
        </s>
        <s>Assistant:
        """

        input_ids = self.tokenizer([prompt], return_tensors="pt", add_special_tokens=False).input_ids.to('cuda')

        generate_input = {
            "input_ids": input_ids,
            "max_new_tokens": 512,  # max is 4096
            "do_sample": True,
            "top_k": 50,
            "top_p": 1,
            "temperature": 0.1,
            "repetition_penalty": 1,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        generate_ids = self.model.generate(**generate_input)

        response = self.tokenizer.decode(generate_ids[0])
        response = response.split("Assistant:")[-1]
        return response



class QwenLocal:
    def __init__(self):
        super(QwenLocal, self).__init__()

        self.configs = config.ConfigParser()

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.configs.get(key='qwen')['model_name_or_path'],
            device_map='auto',
            trust_remote_code=True
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.configs.get(key='qwen')['model_name_or_path'],
            trust_remote_code=True
        )

    def completion(self, system_prompt, user_prompt):
        response, history = self.model.chat(
            self.tokenizer, system=system_prompt, query=user_prompt, history=None)
        return response


class QwenOnline:
    def __init__(self):
        super(QwenOnline, self).__init__()

        self.configs = config.ConfigParser()
        self.model = Generation()

    def completion(self, system_prompt, user_prompt):
        # More information please see
        # https://help.aliyun.com/zh/dashscope/developer-reference/api-details
        messages = [{'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}]
        response = self.model.call(
            model='qwen-plus',
            messages=messages,
            result_format='message',
            api_key=self.configs.get(key='qwen')['api_key']
        )
        if response.status_code == HTTPStatus.OK:
            return response.output['choices'][0]['message']['content']
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
            return "I am so sorry, request failed, please retry"


class M3E:
    def __init__(self):
        super(M3E, self).__init__()

        self.configs = config.ConfigParser()

        # 设置模型的名称（可在线下载，但是网络需要可以访问 HuggingFace ）
        # 或者本地的权重路径（提前离线下载）
        self.model = SentenceTransformer(model_name_or_path=self.configs.get(key='m3e')['model_name_or_path'])

    def encode(self, text):
        return self.model.encode(text)
