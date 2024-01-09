from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/get_emoji")
async def get_emoji(text: str):
    return {"emoji": text}