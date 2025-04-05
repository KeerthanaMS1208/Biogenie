from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from bio_llm import ask_bio_question

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(question: Question):
    try:
        print("Q:", question.question)
        answer = ask_bio_question(question.question)
        print("A:", answer)
        return {"answer": answer}
    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "Hello from BioGenie"}

