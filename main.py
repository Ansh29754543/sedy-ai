from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import json
import os
import logging

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("sedy")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sedy API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Groq client ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set — API calls will fail!")

client = Groq(api_key=GROQ_API_KEY)

MODEL = "llama-3.3-70b-versatile"

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Sedy, an intelligent student learning assistant made by Ansh Verma, a school student.
You explain concepts clearly and simply, solve math and science problems step by step,
summarize topics, and help students understand programming.
Always be encouraging, clear and educational.
Only reveal your identity when asked.

IMPORTANT RULES:
- When explaining topics, use markdown formatting: headers (##), bold (**text**), bullet points (- item), code blocks (```language), math formulas ($formula$)
- For math problems, always show step-by-step working with formulas
- NEVER describe flashcards or quizzes in plain text — the frontend handles those separately
- Keep responses focused and well structured
- When the user refers to a previous topic (e.g. "it", "that", "the same topic", "at this basis"), use the conversation history to understand what they mean"""

# ── Request / Response models ──────────────────────────────────────────────────

class HistoryEntry(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[HistoryEntry] = []


class FlashcardRequest(BaseModel):
    topic: str
    count: int = 6
    history: list[HistoryEntry] = []   # ← NEW: pass history so backend can resolve vague topics


class QuizRequest(BaseModel):
    topic: str
    difficulty: str = "medium"
    count: int = 5
    history: list[HistoryEntry] = []   # ← NEW: same


class Flashcard(BaseModel):
    question: str
    answer: str


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    answer: int
    explanation: str


class ChatResponse(BaseModel):
    reply: str


class FlashcardResponse(BaseModel):
    cards: list[Flashcard]
    topic: str          # ← echoes resolved topic back to frontend


class QuizResponse(BaseModel):
    questions: list[QuizQuestion]
    topic: str          # ← echoes resolved topic back to frontend
    difficulty: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_messages(system: str, history: list[HistoryEntry], user_message: str) -> list[dict]:
    """
    Build the messages array for the Groq API.
    Includes last 20 history entries and sanitises for alternating roles.
    """
    messages = [{"role": "system", "content": system}]

    trimmed = history[-20:] if len(history) > 20 else history
    sanitised: list[dict] = []
    for entry in trimmed:
        role = entry.role if entry.role in ("user", "assistant") else "user"
        if sanitised and sanitised[-1]["role"] == role:
            sanitised[-1]["content"] += "\n" + entry.content
        else:
            sanitised.append({"role": role, "content": entry.content})

    messages.extend(sanitised)
    messages.append({"role": "user", "content": user_message})
    return messages


def extract_json_array(raw: str) -> list:
    """Robustly extract a JSON array from a string that may contain markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]

    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found in response")
    return json.loads(raw[start:end])


# Phrases that mean "use the topic from our conversation context"
VAGUE_TOPIC_SIGNALS = [
    "it", "this", "that", "them", "same", "the same",
    "the topic", "same topic", "this topic", "that topic",
    "above", "the above", "this one", "that one",
    "at this basis", "based on this", "based on that",
    "on this", "on that", "from this", "from above",
    "as discussed", "as mentioned", "as explained",
    "what we discussed", "what i said", "previous", "last topic",
]

def is_vague_topic(topic: str) -> bool:
    t = topic.strip().lower()
    return any(t == v or t.startswith(v) for v in VAGUE_TOPIC_SIGNALS)


def resolve_topic_from_history(raw_topic: str, history: list[HistoryEntry]) -> str:
    """
    If raw_topic is vague (e.g. "at this basis", "it", "this"),
    scan the conversation history to find the last concrete subject discussed.

    Resolution priority:
      1. Last assistant message that mentions "flashcards/quiz about X"
      2. Last user message with a non-vague topic
      3. Subject heading in last assistant reply (e.g. "## Introduction to Python")
      4. Fall back to raw_topic unchanged
    """
    if not is_vague_topic(raw_topic):
        return raw_topic

    if not history:
        return raw_topic

    reversed_history = list(reversed(history))

    # 1. Assistant entries that record "Generated N flashcards about X" or "quiz about X"
    import re
    for entry in reversed_history:
        if entry.role == "assistant":
            m = re.search(r'(?:flashcards?|quiz)\s+(?:about|on)\s+([^\n.!?]{3,60})', entry.content, re.I)
            if m:
                logger.info(f"resolve_topic: found in assistant history → {m.group(1).strip()!r}")
                return m.group(1).strip()

    # 2. User messages — strip action words, check what's left isn't vague
    action_words = r'flashcards?|flash\s*cards?|quiz(zes)?|make|create|generate|about|on|me|test|cards?|and\b|a\b|the\b'
    for entry in reversed_history:
        if entry.role == "user":
            cleaned = re.sub(action_words, '', entry.content, flags=re.I).strip()
            if cleaned and not is_vague_topic(cleaned) and len(cleaned) > 2:
                logger.info(f"resolve_topic: found in user history → {cleaned!r}")
                return cleaned

    # 3. Heading in assistant reply: ## Python  or  # Introduction to Python
    for entry in reversed_history:
        if entry.role == "assistant":
            m = re.search(r'(?:^|\n)#{1,3}\s+(?:introduction to\s+)?([A-Za-z][^\n]{2,50})', entry.content, re.I)
            if m:
                logger.info(f"resolve_topic: found heading → {m.group(1).strip()!r}")
                return m.group(1).strip()

    logger.warning(f"resolve_topic: could not resolve {raw_topic!r}, returning as-is")
    return raw_topic


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "Sedy API is live 🚀",
        "model": MODEL,
        "version": "2.1.0",
        "endpoints": ["/chat", "/flashcards", "/quiz"],
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint.
    Sends full history so Sedy maintains memory across the session.
    """
    logger.info(f"/chat  history_len={len(req.history)}  msg={req.message[:80]!r}")

    messages = build_messages(SYSTEM_PROMPT, req.history, req.message)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Groq API error in /chat: {e}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")

    reply = response.choices[0].message.content.strip()
    logger.info(f"/chat  reply_len={len(reply)}")
    return ChatResponse(reply=reply)


@app.post("/flashcards", response_model=FlashcardResponse)
async def flashcards(req: FlashcardRequest):
    """
    Generate flashcards for a topic.
    If topic is vague (e.g. "at this basis", "it"), resolves it from history.
    """
    # ── FIX: resolve vague topic before calling the model ─────────────
    topic = resolve_topic_from_history(req.topic.strip(), req.history)
    if not topic:
        raise HTTPException(status_code=400, detail="topic must not be empty")

    count = max(1, min(req.count, 20))
    logger.info(f"/flashcards  raw_topic={req.topic!r}  resolved={topic!r}  count={count}")

    user_prompt = (
        f'Generate exactly {count} flashcards about "{topic}".\n'
        f"Return ONLY a JSON array with no extra text, no markdown fences:\n"
        f'[{{"question": "...", "answer": "..."}}, ...]'
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=1500,
            temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Groq API error in /flashcards: {e}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")

    raw = response.choices[0].message.content.strip()

    try:
        data = extract_json_array(raw)
        cards = [
            Flashcard(
                question=str(c.get("question") or c.get("front") or ""),
                answer=str(c.get("answer")   or c.get("back")  or ""),
            )
            for c in data if isinstance(c, dict)
        ]
        cards = [c for c in cards if c.question and c.answer]
    except Exception as e:
        logger.warning(f"/flashcards  JSON parse failed: {e}  raw={raw[:200]!r}")
        cards = [Flashcard(question=f"What is {topic}?", answer=raw[:300])]

    logger.info(f"/flashcards  generated {len(cards)} cards for topic={topic!r}")
    return FlashcardResponse(cards=cards, topic=topic)   # ← return resolved topic


@app.post("/quiz", response_model=QuizResponse)
async def quiz(req: QuizRequest):
    """
    Generate a multiple-choice quiz for a topic.
    If topic is vague (e.g. "at this basis"), resolves it from history.
    """
    # ── FIX: resolve vague topic before calling the model ─────────────
    topic = resolve_topic_from_history(req.topic.strip(), req.history)
    if not topic:
        raise HTTPException(status_code=400, detail="topic must not be empty")

    difficulty = req.difficulty if req.difficulty in ("easy", "medium", "hard") else "medium"
    count = max(1, min(req.count, 20))
    logger.info(f"/quiz  raw_topic={req.topic!r}  resolved={topic!r}  difficulty={difficulty}  count={count}")

    user_prompt = (
        f'Generate exactly {count} {difficulty} multiple-choice questions about "{topic}".\n'
        f"Return ONLY a JSON array with no extra text, no markdown fences:\n"
        f'[{{"question": "...", "options": ["A", "B", "C", "D"], "answer": 0, "explanation": "..."}}]\n'
        f'"answer" is the zero-based index (0–3) of the correct option.'
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=2000,
            temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Groq API error in /quiz: {e}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")

    raw = response.choices[0].message.content.strip()

    try:
        data = extract_json_array(raw)
        questions = []
        for q in data:
            if not isinstance(q, dict):
                continue
            options = q.get("options", [])
            while len(options) < 4:
                options.append("N/A")
            options = options[:4]
            answer_idx = max(0, min(int(q.get("answer", 0)), 3))
            questions.append(QuizQuestion(
                question=str(q.get("question", "")),
                options=options,
                answer=answer_idx,
                explanation=str(q.get("explanation", "")),
            ))
        questions = [q for q in questions if q.question]
    except Exception as e:
        logger.warning(f"/quiz  JSON parse failed: {e}  raw={raw[:200]!r}")
        questions = [QuizQuestion(
            question=f"What is a key concept in {topic}?",
            options=["Option A", "Option B", "Option C", "Option D"],
            answer=0,
            explanation=raw[:300],
        )]

    logger.info(f"/quiz  generated {len(questions)} questions for topic={topic!r}")
    return QuizResponse(questions=questions, topic=topic, difficulty=difficulty)  # ← resolved topic


# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("sedy")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sedy API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Groq client ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set — API calls will fail!")

client = Groq(api_key=GROQ_API_KEY)

MODEL = "llama-3.3-70b-versatile"

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Sedy, an intelligent student learning assistant made by Ansh Verma, a school student.
You explain concepts clearly and simply, solve math and science problems step by step,
summarize topics, and help students understand programming.
Always be encouraging, clear and educational.
Only reveal your identity when asked.

IMPORTANT RULES:
- When explaining topics, use markdown formatting: headers (##), bold (**text**), bullet points (- item), code blocks (```language), math formulas ($formula$)
- For math problems, always show step-by-step working with formulas
- NEVER describe flashcards or quizzes in plain text — the frontend handles those separately
- Keep responses focused and well structured
- When the user refers to a previous topic (e.g. "it", "that", "the same topic"), use the conversation history to understand what they mean"""

# ── Request / Response models ──────────────────────────────────────────────────

class HistoryEntry(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[HistoryEntry] = []   # full conversation history from frontend


class FlashcardRequest(BaseModel):
    topic: str
    count: int = 6


class QuizRequest(BaseModel):
    topic: str
    difficulty: str = "medium"   # "easy" | "medium" | "hard"
    count: int = 5


class Flashcard(BaseModel):
    question: str
    answer: str


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    answer: int          # index 0-3 of correct option
    explanation: str


class ChatResponse(BaseModel):
    reply: str


class FlashcardResponse(BaseModel):
    cards: list[Flashcard]
    topic: str


class QuizResponse(BaseModel):
    questions: list[QuizQuestion]
    topic: str
    difficulty: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_messages(system: str, history: list[HistoryEntry], user_message: str) -> list[dict]:
    """
    Build the messages array for the Groq API.

    Rules:
    - System prompt always first.
    - Include the last 20 history entries (10 turns) to respect token limits
      while still giving meaningful context.
    - Append the current user message last.
    - Groq requires alternating user/assistant turns; we sanitise the history
      to guarantee that before sending.
    """
    messages = [{"role": "system", "content": system}]

    # Take last 20 history entries and sanitise roles
    trimmed = history[-20:] if len(history) > 20 else history
    sanitised: list[dict] = []
    for entry in trimmed:
        role = entry.role if entry.role in ("user", "assistant") else "user"
        # Avoid consecutive same-role messages (Groq rejects them)
        if sanitised and sanitised[-1]["role"] == role:
            # Merge into the previous entry to keep alternation
            sanitised[-1]["content"] += "\n" + entry.content
        else:
            sanitised.append({"role": role, "content": entry.content})

    messages.extend(sanitised)
    messages.append({"role": "user", "content": user_message})
    return messages


def extract_json_array(raw: str) -> list:
    """
    Robustly extract a JSON array from a string that may contain
    surrounding prose or markdown fences.
    """
    # Strip common markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]          # drop opening fence
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]          # drop closing fence

    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found in response")
    return json.loads(raw[start:end])


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "Sedy API is live 🚀",
        "model": MODEL,
        "version": "2.0.0",
        "endpoints": ["/chat", "/flashcards", "/quiz"],
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint.
    Accepts the full conversation history so Sedy can maintain memory
    across the entire session (history is persisted on the frontend
    in localStorage and sent with every request).
    """
    logger.info(f"/chat  history_len={len(req.history)}  msg={req.message[:80]!r}")

    messages = build_messages(SYSTEM_PROMPT, req.history, req.message)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Groq API error in /chat: {e}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")

    reply = response.choices[0].message.content.strip()
    logger.info(f"/chat  reply_len={len(reply)}")
    return ChatResponse(reply=reply)


@app.post("/flashcards", response_model=FlashcardResponse)
async def flashcards(req: FlashcardRequest):
    """
    Generate flip-card style flashcards for a given topic.
    Returns a clean JSON array — no markdown, no prose.
    """
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic must not be empty")

    count = max(1, min(req.count, 20))   # clamp between 1 and 20
    logger.info(f"/flashcards  topic={topic!r}  count={count}")

    user_prompt = (
        f'Generate exactly {count} flashcards about "{topic}".\n'
        f"Return ONLY a JSON array with no extra text, no markdown fences:\n"
        f'[{{"question": "...", "answer": "..."}}, ...]'
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=1500,
            temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Groq API error in /flashcards: {e}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")

    raw = response.choices[0].message.content.strip()

    try:
        data = extract_json_array(raw)
        cards = [
            Flashcard(
                question=str(c.get("question") or c.get("front") or ""),
                answer=str(c.get("answer")   or c.get("back")  or ""),
            )
            for c in data
            if isinstance(c, dict)
        ]
        # Drop empty cards
        cards = [c for c in cards if c.question and c.answer]
    except Exception as e:
        logger.warning(f"/flashcards  JSON parse failed: {e}  raw={raw[:200]!r}")
        # Graceful fallback: wrap whatever text we got into a single card
        cards = [Flashcard(question=f"What is {topic}?", answer=raw[:300])]

    logger.info(f"/flashcards  generated {len(cards)} cards")
    return FlashcardResponse(cards=cards, topic=topic)


@app.post("/quiz", response_model=QuizResponse)
async def quiz(req: QuizRequest):
    """
    Generate a multiple-choice quiz for a given topic.
    Each question has 4 options, the index of the correct one, and an explanation.
    """
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic must not be empty")

    difficulty = req.difficulty if req.difficulty in ("easy", "medium", "hard") else "medium"
    count = max(1, min(req.count, 20))
    logger.info(f"/quiz  topic={topic!r}  difficulty={difficulty}  count={count}")

    user_prompt = (
        f'Generate exactly {count} {difficulty} multiple-choice questions about "{topic}".\n'
        f"Return ONLY a JSON array with no extra text, no markdown fences:\n"
        f'[{{"question": "...", "options": ["A", "B", "C", "D"], "answer": 0, "explanation": "..."}}]\n'
        f'"answer" is the zero-based index (0–3) of the correct option.'
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=2000,
            temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Groq API error in /quiz: {e}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")

    raw = response.choices[0].message.content.strip()

    try:
        data = extract_json_array(raw)
        questions = []
        for q in data:
            if not isinstance(q, dict):
                continue
            options = q.get("options", [])
            # Ensure exactly 4 options
            while len(options) < 4:
                options.append("N/A")
            options = options[:4]

            answer_idx = int(q.get("answer", 0))
            answer_idx = max(0, min(answer_idx, 3))  # clamp to valid range

            questions.append(QuizQuestion(
                question=str(q.get("question", "")),
                options=options,
                answer=answer_idx,
                explanation=str(q.get("explanation", "")),
            ))
        questions = [q for q in questions if q.question]
    except Exception as e:
        logger.warning(f"/quiz  JSON parse failed: {e}  raw={raw[:200]!r}")
        # Graceful fallback
        questions = [QuizQuestion(
            question=f"What is a key concept in {topic}?",
            options=["Option A", "Option B", "Option C", "Option D"],
            answer=0,
            explanation=raw[:300],
        )]

    logger.info(f"/quiz  generated {len(questions)} questions")
    return QuizResponse(questions=questions, topic=topic, difficulty=difficulty)
