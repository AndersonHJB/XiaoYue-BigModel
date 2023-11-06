import openai
# openai.api_key = 'Raplace to your API Key'
openai.api_key = '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'

prompt = ["问题：介绍一下 AI悦创·编程一对一是什么类型的公司\n回答："]
response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, temperature=0, max_tokens=1024)
print(response['choices'][0]['text'])