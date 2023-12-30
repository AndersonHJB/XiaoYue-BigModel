from fastapi import FastAPI, Header, HTTPException, Depends
import uvicorn
from pydantic import BaseModel
import service
import config


app = FastAPI(title="Mobot API")

svc = service.Service()

configs = config.ConfigParser()
self_api_key = configs.get(key='self')['api_key']


# 校验 HTTP Header 中 X-API-KEY 是否正确
async def verify_api_key(X_API_KEY: str = Header(None)):
    if X_API_KEY != self_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


# 规范 HTTP Body 中的 Schema
class MobotChatRequest(BaseModel):
    message: str
    chat_history: list


class MobotChatResponse(BaseModel):
    response: str


# 具体的 API 实现，规范输入输出参数，定义调用逻辑
@app.post('/api/mobot/chat')
def MobotChat(body: MobotChatRequest, X_API_KEY: str = Depends(verify_api_key)):
    # 获取用户的输入
    message = body.message
    chat_history = body.chat_history

    # 和 Web 一样的业务处理逻辑，拿到结果放进去接口规范的对象
    response = MobotChatResponse(
        response=svc.retrival_inference_answer(message, chat_history)
    )

    return response


# 进行 API 端口监听服务
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5280)
