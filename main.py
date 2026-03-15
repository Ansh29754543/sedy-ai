from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

# ── Request models ─────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list = []  # conversation history

class FlashcardRequest(BaseModel):
    topic: str
    count: int = 6

class QuizRequest(BaseModel):
    topic: str
    difficulty: str = "medium"
    count: int = 5

# ── Health check ───────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "Sedy API is live 🚀", "model": MODEL}

# ── Chat with memory ───────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    # Build messages with full history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add previous messages (last 10 only to save tokens)
    for h in req.history[-10:]:
        messages.append({
            "role": h.get("role", "user"),
            "content": h.get("content", "")
        })

    # Add current message
    messages.append({"role": "user", "content": req.message})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.7
    )
    return {"reply": response.choices[0].message.content.strip()}

# ── Flashcards ─────────────────────────────────────────────────────────
@app.post("/flashcards")
async def flashcards(req: FlashcardRequest):
    user_prompt = f"""Generate {req.count} flashcards about "{req.topic}".
Return ONLY a JSON array, no extra text, no markdown:
[{{"question": "...", "answer": "..."}}, ...]"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1200,
        temperature=0.7
    )
    raw = response.choices[0].message.content.strip()

    try:
        start = raw.find('[')
        end = raw.rfind(']') + 1
        cards = json.loads(raw[start:end])
    except:
        cards = [{"question": f"What is {req.topic}?", "answer": raw[:300]}]

    return {"cards": cards}

# ── Quiz ───────────────────────────────────────────────────────────────
@app.post("/quiz")
async def quiz(req: QuizRequest):
    user_prompt = f"""Generate {req.count} {req.difficulty} multiple choice questions about "{req.topic}".
Return ONLY a JSON array, no extra text, no markdown:
[{{"question": "...", "options": ["A", "B", "C", "D"], "answer": 0, "explanation": "..."}}]
"answer" is the index (0-3) of the correct option."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1500,
        temperature=0.7
    )
    raw = response.choices[0].message.content.strip()

    try:
        start = raw.find('[')
        end = raw.rfind(']') + 1
        questions = json.loads(raw[start:end])
    except:
        questions = [{
            "question": f"What is a key concept in {req.topic}?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": 0,
            "explanation": raw[:300]
        }]

    return {"questions": questions}
