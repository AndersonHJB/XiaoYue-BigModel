### Llama2
#### 由于Llama2本身的中文对齐较弱，我们采用中文数据集微调后的模型，使其具备较强的中文对话能力。
```
# 相关链接
# 性能 13B > 7B

https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat
https://huggingface.co/FlagAlpha/Llama2-Chinese-7b-Chat

https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat-LoRA
https://huggingface.co/FlagAlpha/Llama2-Chinese-7b-Chat-LoRA
``` 

#### Meta 官方版本
```
# 相关链接
# 性能 70B > 13B > 7B
https://huggingface.co/meta-llama/Llama-2-70b-chat-hf
https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

#### 在算力允许的情况下，请使用选择更大的模型。
```
cd ./model/llama2/
python download.py
```