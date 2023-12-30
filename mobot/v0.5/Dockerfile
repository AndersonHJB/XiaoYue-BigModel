FROM python:3.10-slim

MAINTAINER <Carry Yip> carryyip1996@gmail.com

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/

# 如果是 API 则需要改成
# EXPOSE 5280
EXPOSE 7860

# 如果是 API 则需要改成
# CMD ["python", "api.py"]
CMD ["python", "web.py"]
