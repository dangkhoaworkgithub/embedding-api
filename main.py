from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()

# Load model
model = SentenceTransformer("intfloat/multilingual-e5-small")

class EmbedRequest(BaseModel):
    input: str

class EmbedResponse(BaseModel):
    embedding: list[float]

@app.post("/embed/retrieval", response_model=EmbedResponse)
async def get_embedding(request: EmbedRequest):
    try:
        # Định dạng văn bản đúng chuẩn E5
        text = f"passage: {request.input}"
        embedding = model.encode(text).tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Embedding API is running!"}
