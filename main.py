from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import json
import os
import logging
import base64
import re

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("sedy")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sedy API", version="2.2.0")

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
- When explaining topics, use markdown formatting: headers (##), bold (**text**), bullet points (- item), code blocks (```language)
- NEVER describe flashcards or quizzes in plain text — the frontend handles those separately
- Keep responses focused and well structured
- When the user refers to a previous topic (e.g. "it", "that", "the same topic", "at this basis"), use the conversation history to understand what they mean

STRICT MATH FORMATTING RULES (critical — the frontend uses KaTeX to render math):
- ALL mathematical expressions, variables, fractions, and formulas MUST be wrapped in LaTeX delimiters
- Use $...$ for inline math. Examples: $w$, $h$, $2w^2$, $V = 36$, $P(A|X)$
- Use $$...$$ on its own line for display/block equations. Example: $$C = 200w^2 + \\frac{6480}{w}$$
- NEVER write bare math like: w^2, dC/dw, A_base, x^2+3 — always wrap in $...$
- For fractions always use \\frac{numerator}{denominator} inside $...$. Example: $\\frac{18}{w^2}$
- For subscripts use _{...}: $A_{\\text{base}}$, $A_{\\text{sides}}$, $P(X|A)$
- For derivatives: $\\frac{dC}{dw} = 0$
- For cube roots: $w = \\sqrt[3]{16.2}$
- For each step in a solution, place the equation on its own line using $$...$$
- Never write "w squared" in plain text — always $w^2$"""

# ── Prompt Refinement System Prompt ───────────────────────────────────────────
REFINE_SYSTEM_PROMPT = """You are a silent prompt refinement engine. Your ONLY job is to clean up a student's message before it reaches an AI tutor.

You must perform these three tasks in order:
1. SPELLING FIX — Correct obvious typos and misspellings (e.g. "helllodof" → "hello", "photosinthesis" → "photosynthesis", "qiuz" → "quiz"). Do not change technical terms you are unsure about.
2. GRAMMAR FIX — Fix grammar and sentence structure while keeping the original meaning and tone. Do not make it formal if it was casual.
3. CONTEXT RESOLUTION — If the message is vague or uses references like "it", "this", "that", "the same topic", "on it", "at this basis", "as before", look at the conversation history and rewrite the message with the actual topic filled in. For example:
   - History: user asked for "flashcards on photosynthesis", now says "and a quiz on it" → output: "Generate a quiz on photosynthesis"
   - History: user asked about Python, now says "explain that again" → output: "Explain Python again"
   - If nothing in history is relevant, leave the message as-is after spelling/grammar fixes.

CRITICAL RULES:
- Output ONLY the refined message. No explanation, no preamble, no quotes around it.
- Never add extra information or change the intent of the message.
- Never refuse or comment. Just output the refined text.
- If the message is already perfect, output it exactly as-is.
- Keep it concise — do not expand the message unnecessarily."""


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
    history: list[HistoryEntry] = []


class QuizRequest(BaseModel):
    topic: str
    difficulty: str = "medium"
    count: int = 5
    history: list[HistoryEntry] = []


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
    topic: str


class QuizResponse(BaseModel):
    questions: list[QuizQuestion]
    topic: str
    difficulty: str


# ── Prompt Refinement ──────────────────────────────────────────────────────────

async def refine_prompt(message: str, history: list[HistoryEntry]) -> str:
    """
    Silently refine the user's message before passing it to the main AI:
      1. Fix spelling/typos
      2. Fix grammar
      3. Resolve vague references using conversation history

    The user never sees this — the frontend displays their original message.
    Returns the refined message string, falling back to the original on any error.
    """
    if not message or not message.strip():
        return message

    # Build a compact history snippet (last 10 turns) for context
    history_snippet = ""
    if history:
        trimmed = history[-10:]
        lines = []
        for entry in trimmed:
            role_label = "Student" if entry.role == "user" else "Tutor"
            # Truncate long entries to keep the refine call cheap
            content_preview = entry.content[:200] + ("..." if len(entry.content) > 200 else "")
            lines.append(f"{role_label}: {content_preview}")
        history_snippet = "\n".join(lines)

    # Compose the refine request
    if history_snippet:
        user_content = (
            f"Conversation history so far:\n{history_snippet}\n\n"
            f"New message to refine: {message}"
        )
    else:
        user_content = f"Message to refine: {message}"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": REFINE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=256,
            temperature=0.2,   # low temperature for deterministic, conservative fixes
        )
        refined = response.choices[0].message.content.strip()

        # Safety check: if the model returned something wildly different in length, fall back
        if not refined or len(refined) > len(message) * 4:
            logger.warning(f"refine_prompt: suspicious output length, falling back. refined={refined[:80]!r}")
            return message

        logger.info(f"refine_prompt: '{message[:60]}' → '{refined[:60]}'")
        return refined

    except Exception as e:
        # Never let refinement failures block the main request
        logger.warning(f"refine_prompt: failed ({e}), using original message")
        return message


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
    If raw_topic is vague, scan conversation history to find the last concrete subject.
    """
    if not is_vague_topic(raw_topic):
        return raw_topic

    if not history:
        return raw_topic

    reversed_history = list(reversed(history))

    # 1. Assistant entries that record "Generated N flashcards about X" or "quiz about X"
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

    # 3. Heading in assistant reply
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
        "version": "2.2.0",
        "endpoints": ["/chat", "/flashcards", "/quiz", "/pdf-chat"],
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint.
    Silently refines the user's message (spelling, grammar, context) before
    sending to the model. The frontend always shows the original message.
    """
    logger.info(f"/chat  history_len={len(req.history)}  msg={req.message[:80]!r}")

    # ── Silent prompt refinement ───────────────────────────────────────────────
    refined_message = await refine_prompt(req.message, req.history)

    messages = build_messages(SYSTEM_PROMPT, req.history, refined_message)

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
    Refines the topic string silently before generation.
    """
    # ── Silent prompt refinement on the topic ──────────────────────────────────
    # Wrap in a natural sentence so the refiner has context to work with
    topic_as_prompt = f"Generate flashcards about {req.topic.strip()}"
    refined_prompt  = await refine_prompt(topic_as_prompt, req.history)

    # Extract topic back from the refined prompt
    topic_match = re.search(
        r'(?:flashcards?\s+(?:about|on|for)\s+|flashcards?\s+)(.+)',
        refined_prompt, re.I
    )
    topic = topic_match.group(1).strip() if topic_match else refined_prompt.strip()

    # Also run the original vague-topic resolver as a safety net
    topic = resolve_topic_from_history(topic, req.history)

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
    return FlashcardResponse(cards=cards, topic=topic)


@app.post("/quiz", response_model=QuizResponse)
async def quiz(req: QuizRequest):
    """
    Generate a multiple-choice quiz for a topic.
    Refines the topic string silently before generation.
    """
    # ── Silent prompt refinement on the topic ──────────────────────────────────
    topic_as_prompt = f"Quiz me on {req.topic.strip()}"
    refined_prompt  = await refine_prompt(topic_as_prompt, req.history)

    # Extract topic back
    topic_match = re.search(
        r'(?:quiz\s+(?:me\s+)?(?:on|about)\s+|quiz\s+on\s+)(.+)',
        refined_prompt, re.I
    )
    topic = topic_match.group(1).strip() if topic_match else refined_prompt.strip()

    # Also run the vague-topic resolver
    topic = resolve_topic_from_history(topic, req.history)

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
    return QuizResponse(questions=questions, topic=topic, difficulty=difficulty)


# ── PDF Chat ───────────────────────────────────────────────────────────────────

class PdfChatRequest(BaseModel):
    message: str
    pdf_base64: str
    pdf_name: str = "document.pdf"
    history: list[HistoryEntry] = []


class PdfChatResponse(BaseModel):
    reply: str
    pdf_name: str


PDF_SYSTEM_PROMPT = """You are Sedy, an intelligent student learning assistant made by Ansh Verma.
The user has uploaded a PDF document. Your job is to help them understand it.

You can:
- Answer questions about the document's content
- Summarize sections or the whole document
- Explain concepts mentioned in the document
- Generate study notes from the document
- When asked for flashcards or a quiz, describe the key topics clearly so the frontend can generate them

STRICT MATH FORMATTING RULES:
- ALL mathematical expressions MUST use LaTeX delimiters: $...$ for inline, $$...$$ for display
- NEVER write bare math like: w^2, dC/dw — always wrap in $...$
- Use \\frac{}{} for fractions, _{} for subscripts

Always be encouraging, clear, and educational.
Only answer based on the document content. If something is not in the document, say so."""


@app.post("/pdf-chat", response_model=PdfChatResponse)
async def pdf_chat(req: PdfChatRequest):
    """
    Accept a base64-encoded PDF and a user question.
    Silently refines the user's message before processing.
    """
    pdf_name = req.pdf_name.strip() or "document.pdf"
    logger.info(f"/pdf-chat  file={pdf_name!r}  msg={req.message[:80]!r}  history={len(req.history)}")

    # ── Silent prompt refinement ───────────────────────────────────────────────
    refined_message = await refine_prompt(req.message, req.history)

    # ── Decode and extract text from PDF ──────────────────────────────────────
    try:
        pdf_bytes = base64.b64decode(req.pdf_base64)
    except Exception as e:
        logger.error(f"/pdf-chat  base64 decode failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid base64 PDF data")

    pdf_text = ""
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_count = len(reader.pages)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            pdf_text += f"\n--- Page {i+1} ---\n{page_text}"
        logger.info(f"/pdf-chat  extracted {len(pdf_text)} chars from {page_count} pages")
    except Exception as e:
        logger.error(f"/pdf-chat  PDF text extraction failed: {e}")
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {str(e)}")

    if not pdf_text.strip():
        raise HTTPException(
            status_code=422,
            detail="The PDF appears to be empty or contains only scanned images (no extractable text). "
                   "Please try a text-based PDF."
        )

    MAX_PDF_CHARS = 60_000
    if len(pdf_text) > MAX_PDF_CHARS:
        pdf_text = pdf_text[:MAX_PDF_CHARS] + "\n\n[... document truncated to fit context ...]"
        logger.warning(f"/pdf-chat  PDF text truncated to {MAX_PDF_CHARS} chars")

    doc_context = (
        f"The user has uploaded a PDF named \"{pdf_name}\".\n"
        f"Here is the full text content of the document:\n"
        f"=== START OF DOCUMENT ===\n{pdf_text}\n=== END OF DOCUMENT ===\n"
    )

    messages = [{"role": "system", "content": PDF_SYSTEM_PROMPT + "\n\n" + doc_context}]

    trimmed = req.history[-10:] if len(req.history) > 10 else req.history
    sanitised: list[dict] = []
    for entry in trimmed:
        role = entry.role if entry.role in ("user", "assistant") else "user"
        if sanitised and sanitised[-1]["role"] == role:
            sanitised[-1]["content"] += "\n" + entry.content
        else:
            sanitised.append({"role": role, "content": entry.content})
    messages.extend(sanitised)

    # Use the refined message — user sees original on frontend
    messages.append({"role": "user", "content": refined_message})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1500,
            temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Groq API error in /pdf-chat: {e}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")

    reply = response.choices[0].message.content.strip()
    logger.info(f"/pdf-chat  reply_len={len(reply)}")
    return PdfChatResponse(reply=reply, pdf_name=pdf_name)
