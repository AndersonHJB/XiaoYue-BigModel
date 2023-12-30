FROM python:3.10-slim

MAINTAINER <Carry Yip> carryyip1996@gmail.com

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/

EXPOSE 7860

CMD ["python", "web.py"]
