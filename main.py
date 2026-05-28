import logging

import uvicorn
from fastapi import FastAPI

from api.router import router
from config import settings

logging.basicConfig(level=settings.log_level)

app = FastAPI(title="chat-with-my-notes-v2", version="0.2.0")
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.app_port, reload=False)
