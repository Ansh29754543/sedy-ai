from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
import json, os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are Sedy, an intelligent student learning assistant made by Ansh Verma, a school student.
You explain concepts clearly and simply, generate flashcards and quizzes,
solve math and science problems step by step, summarize topics, and help
students understand programming. Always be encouraging, clear and educational.
Only reveal your identity when asked."""

MODEL = "llama-3.3-70b-versatile"

class ChatRequest(BaseModel):
    message: str

class FlashcardRequest(BaseModel):
    topic: str
    count: int = 6

class QuizRequest(BaseModel):
    topic: str
    difficulty: str = "medium"
    count: int = 5

def generate(system, user, max_tokens=1024):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/chat")
async def chat(req: ChatRequest):
    reply = generate(SYSTEM_PROMPT, req.message, 1024)
    return {"reply": reply}

@app.post("/flashcards")
async def flashcards(req: FlashcardRequest):
    user_prompt = f"""Generate {req.count} flashcards about "{req.topic}".
Return ONLY a JSON array, no extra text, no markdown:
[{{"question": "...", "answer": "..."}}, ...]"""
    raw = generate(SYSTEM_PROMPT, user_prompt, 1200)
    try:
        start = raw.find('[')
        end = raw.rfind(']') + 1
        cards = json.loads(raw[start:end])
    except:
        cards = [{"question": f"What is {req.topic}?", "answer": raw[:300]}]
    return {"cards": cards}

@app.post("/quiz")
async def quiz(req: QuizRequest):
    user_prompt = f"""Generate {req.count} {req.difficulty} multiple choice questions about "{req.topic}".
Return ONLY a JSON array, no extra text, no markdown:
[{{"question": "...", "options": ["A", "B", "C", "D"], "answer": 0, "explanation": "..."}}]
"answer" is the index (0-3) of the correct option."""
    raw = generate(SYSTEM_PROMPT, user_prompt, 1500)
    try:
        start = raw.find('[')
        end = raw.rfind(']') + 1
        questions = json.loads(raw[start:end])
    except:
        questions = [{"question": f"What is {req.topic}?", "options": ["A", "B", "C", "D"], "answer": 0, "explanation": raw[:300]}]
    return {"questions": questions}

# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
