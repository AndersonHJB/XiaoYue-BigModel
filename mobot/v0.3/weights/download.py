import os
from huggingface_hub import snapshot_download


allow_patterns = ["*.*"]

# Hugging Face 对应的模型路径
repo_id = 'moka-ai/m3e-large'
# 本地保存模型相关文件的路径
local_dir = "./embedding/m3e-large"

if not os.path.exists(local_dir):
    os.makedirs(local_dir)


snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns, local_dir=local_dir)
