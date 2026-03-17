from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import json
import os
import logging
import base64
import re
import httpx
import asyncio

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("sedy")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sedy API", version="3.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Groq client ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set — API calls will fail!")

client = Groq(api_key=GROQ_API_KEY)

MODEL = "llama-3.3-70b-versatile"

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

# ── Replicate API key ──────────────────────────────────────────────────────────
# Replace the value below with your real token from replicate.com
REPLICATE_API_TOKEN = "r8_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


# ══════════════════════════════════════════════════════════════════════════════
# ── SYSTEM PROMPTS ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are Sedy, an intelligent student learning assistant made by Ansh Verma, a school student.
You explain concepts clearly and simply, solve math and science problems step by step,
summarize topics, and help students understand programming.
Always be encouraging, clear and educational.
Only reveal your identity when asked.

IMPORTANT RULES:
- Use markdown formatting: headers (##), bold (**text**), bullet points (- item), code blocks (```language)
- NEVER describe flashcards or quizzes in plain text — the frontend handles those separately
- Keep responses focused and well structured
- Use conversation history for context when user refers to "it", "that", "same topic"

STRICT MATH FORMATTING (frontend uses KaTeX):
- ALL math MUST be wrapped in LaTeX: $...$ inline, $$...$$ display
- NEVER write bare math like w^2 — always wrap in $w^2$
- Fractions: $\\frac{a}{b}$  Subscripts: $A_{\\text{base}}$"""

# ──────────────────────────────────────────────────────────────────────────────

REFINE_SYSTEM_PROMPT = """You are a silent prompt refinement engine. Your ONLY job is to clean up a student's message.
1. Fix spelling typos
2. Fix grammar (keep casual tone)
3. Resolve vague references ("it", "this", "that") using conversation history

Output ONLY the refined message. No explanation, no preamble. If already perfect, output as-is."""

# ──────────────────────────────────────────────────────────────────────────────

GRAPH_REFINE_SYSTEM_PROMPT = """You are a silent prompt refinement engine for data visualisation requests.
1. Fix spelling/grammar
2. Resolve vague references using conversation history
3. Append chart type hint if strongly implied:
   - "breakdown/share/proportion" → append "(pie chart)"
   - "compare/vs/ranking/top N" → append "(bar chart)"
   - "over time/trend/history" → append "(line chart)"
Output ONLY the refined message."""

# ──────────────────────────────────────────────────────────────────────────────

GRAPH_SYSTEM_PROMPT = """You are a data assistant. Output ONLY a single valid JSON object — no prose, no markdown fences.

SCHEMA:
{
  "title": "Short chart title (max 60 chars)",
  "chart_type": "line",
  "unit": "Y axis unit e.g. Rs/kg or USD billions or %",
  "x_label": "X axis label",
  "caption": "One optional context sentence",
  "series": [{"label": "name", "data": [{"x": "2000", "y": 8.5}]}]
}

CHART TYPE RULES:
- "line": time series, continuous trends
- "bar": discrete categories, snapshots, rankings
- "pie": parts of a whole (ONE series; x = slice labels)

DATA RULES:
- x values = strings, y values = numbers
- Time series: 8-20 points. Pie: 3-8 slices. Bar: 3-10 groups.
- Use realistic historically plausible data."""

# ──────────────────────────────────────────────────────────────────────────────

PDF_SYSTEM_PROMPT = """You are Sedy, an intelligent student learning assistant made by Ansh Verma.
The user has uploaded a PDF. Help them understand it — answer questions, summarize, explain concepts.
Use LaTeX math: $...$ inline, $$...$$ display.
Only answer based on document content."""

# ──────────────────────────────────────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """You are an intent classifier for a student learning app.
Output EXACTLY one word from: graph, flashcard, quiz, both, chat, image

graph     = wants a chart/graph/visualisation
flashcard = wants flip study cards
quiz      = wants MCQ quiz
both      = wants flashcards AND a quiz
image     = wants an AI-generated image/picture/illustration/artwork
chat      = everything else

No punctuation, no explanation. One word only."""

# ──────────────────────────────────────────────────────────────────────────────

CODE_QUESTIONS_PROMPT = """You are helping clarify a coding request before writing code.
The user has asked: "{request}"

Generate 2-3 SHORT, RELEVANT multiple-choice questions to clarify missing details.

STRICT RULES:
- NEVER ask about something the user already mentioned (e.g. if they said "Python", never ask language)
- NEVER ask generic or obvious questions
- Ask only about things that will genuinely change how the code is written
- Good topics: complexity level, extra features, UI style, error handling, data storage
- Each question must have 3-4 short options (1-5 words each)
- If the request is already very detailed and nothing is missing, return an empty array []
- Return ONLY a valid JSON array, no markdown fences, no explanation

FORMAT:
[
  {{"question": "Question text?", "options": ["Option A", "Option B", "Option C"]}},
  {{"question": "Another question?", "options": ["Option A", "Option B", "Option C", "Option D"]}}
]"""

# ──────────────────────────────────────────────────────────────────────────────

IMAGE_PROMPT_REFINE_SYSTEM = """You are a prompt engineer for AI image generation.
The user has requested an image. Extract and expand their request into a clean,
detailed image generation prompt (max 120 words). Be descriptive about style,
lighting, composition, and mood. Output ONLY the refined prompt, nothing else."""


# ══════════════════════════════════════════════════════════════════════════════
# ── PYDANTIC MODELS ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class HistoryEntry(BaseModel):
    role: str
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

class GraphRequest(BaseModel):
    message: str
    chart_type: str = "auto"
    history: list[HistoryEntry] = []

class PdfChatRequest(BaseModel):
    message: str
    pdf_base64: str
    pdf_name: str = "document.pdf"
    history: list[HistoryEntry] = []

class ImageGenRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    reply: str

class Flashcard(BaseModel):
    question: str
    answer: str

class FlashcardResponse(BaseModel):
    cards: list[Flashcard]
    topic: str

class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    answer: int
    explanation: str

class QuizResponse(BaseModel):
    questions: list[QuizQuestion]
    topic: str
    difficulty: str

class GraphPoint(BaseModel):
    x: str
    y: float

class GraphSeries(BaseModel):
    label: str
    data: list[GraphPoint]

class GraphResponse(BaseModel):
    title: str
    chart_type: str = "line"
    unit: str = ""
    x_label: str = ""
    caption: str = ""
    data_source: str = ""
    series: list[GraphSeries]

class PdfChatResponse(BaseModel):
    reply: str
    pdf_name: str

class IntentRequest(BaseModel):
    message: str
    history: list[HistoryEntry] = []

class IntentResponse(BaseModel):
    intent: str

class CodeQuestionsRequest(BaseModel):
    message: str

class CodeQuestion(BaseModel):
    question: str
    options: list[str]

class CodeQuestionsResponse(BaseModel):
    questions: list[CodeQuestion]

class ImageGenResponse(BaseModel):
    image_url: str
    prompt_used: str


# ══════════════════════════════════════════════════════════════════════════════
# ── HELPERS ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def build_messages(system: str, history: list[HistoryEntry], user_message: str) -> list[dict]:
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


def strip_json_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    return raw


VAGUE_TOPIC_SIGNALS = [
    "it","this","that","them","same","the same","the topic","same topic",
    "this topic","that topic","above","the above","this one","that one",
    "at this basis","based on this","based on that","on this","on that",
    "from this","from above","as discussed","as mentioned","as explained",
    "what we discussed","what i said","previous","last topic",
]

def is_vague_topic(topic: str) -> bool:
    t = topic.strip().lower()
    return any(t == v or t.startswith(v) for v in VAGUE_TOPIC_SIGNALS)


def resolve_topic_from_history(raw_topic: str, history: list[HistoryEntry]) -> str:
    if not is_vague_topic(raw_topic):
        return raw_topic
    if not history:
        return raw_topic
    rev = list(reversed(history))
    for entry in rev:
        if entry.role == "assistant":
            m = re.search(r'(?:flashcards?|quiz)\s+(?:about|on)\s+([^\n.!?]{3,60})', entry.content, re.I)
            if m:
                return m.group(1).strip()
    action_words = r'flashcards?|flash\s*cards?|quiz(zes)?|make|create|generate|about|on|me|test|cards?|and\b|a\b|the\b'
    for entry in rev:
        if entry.role == "user":
            cleaned = re.sub(action_words, '', entry.content, flags=re.I).strip()
            if cleaned and not is_vague_topic(cleaned) and len(cleaned) > 2:
                return cleaned
    for entry in rev:
        if entry.role == "assistant":
            m = re.search(r'(?:^|\n)#{1,3}\s+(?:introduction to\s+)?([A-Za-z][^\n]{2,50})', entry.content, re.I)
            if m:
                return m.group(1).strip()
    return raw_topic


async def refine_prompt(message: str, history: list[HistoryEntry],
                        system_prompt: str = REFINE_SYSTEM_PROMPT) -> str:
    if not message or not message.strip():
        return message
    history_snippet = ""
    if history:
        lines = [f"{'Student' if e.role=='user' else 'Tutor'}: {e.content[:200]}" for e in history[-10:]]
        history_snippet = "\n".join(lines)
    user_content = (
        f"Conversation history:\n{history_snippet}\n\nMessage to refine: {message}"
        if history_snippet else f"Message to refine: {message}"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
            max_tokens=256, temperature=0.2,
        )
        refined = response.choices[0].message.content.strip()
        if not refined or len(refined) > len(message) * 5:
            return message
        logger.info(f"refine: '{message[:50]}' → '{refined[:50]}'")
        return refined
    except Exception as e:
        logger.warning(f"refine_prompt failed ({e}), using original")
        return message


async def refine_graph_prompt(message: str, history: list[HistoryEntry]) -> str:
    return await refine_prompt(message, history, system_prompt=GRAPH_REFINE_SYSTEM_PROMPT)


async def refine_image_prompt(raw_prompt: str) -> str:
    """Expand a short user image request into a rich generation prompt."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": IMAGE_PROMPT_REFINE_SYSTEM},
                {"role": "user",   "content": raw_prompt},
            ],
            max_tokens=180,
            temperature=0.7,
        )
        refined = response.choices[0].message.content.strip()
        if not refined:
            return raw_prompt
        logger.info(f"image prompt refined: '{raw_prompt[:50]}' → '{refined[:60]}'")
        return refined
    except Exception as e:
        logger.warning(f"refine_image_prompt failed ({e}), using original")
        return raw_prompt


async def fetch_live_data(query: str) -> str:
    if not SERPER_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=8.0) as c:
            resp = await c.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                json={"q": query, "num": 5},
            )
            if resp.status_code != 200:
                return ""
            snippets = []
            for r in resp.json().get("organic", [])[:5]:
                t, s = r.get("title", ""), r.get("snippet", "")
                if t or s:
                    snippets.append(f"• {t}: {s}")
            return "\n".join(snippets)
    except Exception as e:
        logger.warning(f"live_data failed: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# ── ROUTES ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "status": "Sedy API is live 🚀",
        "version": "3.1.0",
        "model": MODEL,
        "live_data": bool(SERPER_API_KEY),
        "image_gen": bool(REPLICATE_API_TOKEN and not REPLICATE_API_TOKEN.endswith("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")),
        "endpoints": ["/chat", "/flashcards", "/quiz", "/graph", "/pdf-chat", "/intent", "/code-questions", "/generate-image"],
    }


# ── /intent ────────────────────────────────────────────────────────────────────

@app.post("/intent", response_model=IntentResponse)
async def detect_intent(req: IntentRequest):
    logger.info(f"/intent  msg={req.message[:80]!r}")
    history_snippet = ""
    if req.history:
        lines = [f"{'Student' if e.role=='user' else 'Sedy'}: {e.content[:150]}" for e in req.history[-6:]]
        history_snippet = "\n".join(lines)
    user_content = (
        f"Recent conversation:\n{history_snippet}\n\nNew message: {req.message}"
        if history_snippet else f"Message: {req.message}"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": INTENT_SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            max_tokens=5, temperature=0.0,
        )
        raw = response.choices[0].message.content.strip().lower()
        intent = raw if raw in ("graph", "flashcard", "quiz", "both", "image", "chat") else "chat"
        logger.info(f"/intent  result={intent!r}")
        return IntentResponse(intent=intent)
    except Exception as e:
        logger.warning(f"/intent  failed ({e}), defaulting to chat")
        return IntentResponse(intent="chat")


# ── /chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    logger.info(f"/chat  history={len(req.history)}  msg={req.message[:80]!r}")
    refined = await refine_prompt(req.message, req.history)
    messages = build_messages(SYSTEM_PROMPT, req.history, refined)
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, max_tokens=1024, temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Groq error /chat: {e}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    reply = response.choices[0].message.content.strip()
    logger.info(f"/chat  reply_len={len(reply)}")
    return ChatResponse(reply=reply)


# ── /flashcards ────────────────────────────────────────────────────────────────

@app.post("/flashcards", response_model=FlashcardResponse)
async def flashcards(req: FlashcardRequest):
    logger.info(f"/flashcards  topic={req.topic!r}")
    topic_as_prompt = f"Generate flashcards about {req.topic.strip()}"
    refined = await refine_prompt(topic_as_prompt, req.history)
    m = re.search(r'(?:flashcards?\s+(?:about|on|for)\s+|flashcards?\s+)(.+)', refined, re.I)
    topic = m.group(1).strip() if m else refined.strip()
    topic = resolve_topic_from_history(topic, req.history)
    if not topic:
        raise HTTPException(status_code=400, detail="topic must not be empty")
    count = max(1, min(req.count, 20))
    user_prompt = (
        f'Generate exactly {count} flashcards about "{topic}".\n'
        f'Return ONLY a JSON array, no markdown:\n[{{"question":"...","answer":"..."}}]'
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            max_tokens=1500, temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Groq error /flashcards: {e}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    raw = response.choices[0].message.content.strip()
    try:
        data = extract_json_array(raw)
        cards = [
            Flashcard(
                question=str(c.get("question") or c.get("front", "")),
                answer=str(c.get("answer") or c.get("back", "")),
            )
            for c in data if isinstance(c, dict)
        ]
        cards = [c for c in cards if c.question and c.answer]
    except Exception:
        cards = [Flashcard(question=f"What is {topic}?", answer=raw[:300])]
    logger.info(f"/flashcards  generated {len(cards)} cards")
    return FlashcardResponse(cards=cards, topic=topic)


# ── /quiz ──────────────────────────────────────────────────────────────────────

@app.post("/quiz", response_model=QuizResponse)
async def quiz(req: QuizRequest):
    logger.info(f"/quiz  topic={req.topic!r}")
    topic_as_prompt = f"Quiz me on {req.topic.strip()}"
    refined = await refine_prompt(topic_as_prompt, req.history)
    m = re.search(r'(?:quiz\s+(?:me\s+)?(?:on|about)\s+)(.+)', refined, re.I)
    topic = m.group(1).strip() if m else refined.strip()
    topic = resolve_topic_from_history(topic, req.history)
    if not topic:
        raise HTTPException(status_code=400, detail="topic must not be empty")
    difficulty = req.difficulty if req.difficulty in ("easy", "medium", "hard") else "medium"
    count = max(1, min(req.count, 20))
    user_prompt = (
        f'Generate exactly {count} {difficulty} MCQs about "{topic}".\n'
        f'Return ONLY JSON array, no markdown:\n'
        f'[{{"question":"...","options":["A","B","C","D"],"answer":0,"explanation":"..."}}]'
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            max_tokens=2000, temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Groq error /quiz: {e}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    raw = response.choices[0].message.content.strip()
    try:
        data = extract_json_array(raw)
        questions = []
        for q in data:
            if not isinstance(q, dict):
                continue
            opts = q.get("options", [])
            while len(opts) < 4:
                opts.append("N/A")
            questions.append(QuizQuestion(
                question=str(q.get("question", "")),
                options=opts[:4],
                answer=max(0, min(int(q.get("answer", 0)), 3)),
                explanation=str(q.get("explanation", "")),
            ))
        questions = [q for q in questions if q.question]
    except Exception:
        questions = [QuizQuestion(
            question=f"Key concept in {topic}?",
            options=["Option A", "Option B", "Option C", "Option D"],
            answer=0,
            explanation=raw[:300],
        )]
    logger.info(f"/quiz  generated {len(questions)} questions")
    return QuizResponse(questions=questions, topic=topic, difficulty=difficulty)


# ── /graph ─────────────────────────────────────────────────────────────────────

@app.post("/graph", response_model=GraphResponse)
async def graph(req: GraphRequest):
    logger.info(f"/graph  msg={req.message[:80]!r}")
    refined = await refine_graph_prompt(req.message, req.history)
    logger.info(f"/graph  refined='{refined[:80]}'")
    live_snippets = ""
    data_source = "estimated"
    if SERPER_API_KEY:
        live_snippets = await fetch_live_data(refined + " data statistics numbers")
        if live_snippets:
            data_source = "live"
    user_content = (
        f"User request: {refined}\n\nReal-time data (use where possible):\n{live_snippets}\n\nGenerate chart JSON."
        if live_snippets else refined
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": GRAPH_SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            max_tokens=2500, temperature=0.3,
        )
    except Exception as e:
        logger.error(f"Groq error /graph: {e}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    raw = strip_json_fences(response.choices[0].message.content.strip())
    try:
        obj = json.loads(raw)
        series_list = []
        for i, s in enumerate(obj.get("series", [])):
            pts = []
            for pt in s.get("data", []):
                try:
                    pts.append(GraphPoint(x=str(pt["x"]), y=float(pt["y"])))
                except (KeyError, ValueError, TypeError):
                    continue
            if pts:
                series_list.append(GraphSeries(label=str(s.get("label", f"Series {i+1}")), data=pts))
        if not series_list:
            raise ValueError("No valid series found")
        ct = str(obj.get("chart_type", "line")).lower().strip()
        if ct not in ("line", "bar", "pie"):
            ct = "line"
        result = GraphResponse(
            title=str(obj.get("title", refined[:60])),
            chart_type=ct,
            unit=str(obj.get("unit", "")),
            x_label=str(obj.get("x_label", "")),
            caption=str(obj.get("caption", "")),
            data_source=data_source,
            series=series_list,
        )
        logger.info(f"/graph  title={result.title!r}  type={ct}  series={len(series_list)}")
        return result
    except Exception as e:
        logger.warning(f"/graph  parse failed: {e}  raw={raw[:300]!r}")
        raise HTTPException(status_code=422, detail=f"Could not parse graph data: {e}")


# ── /pdf-chat ──────────────────────────────────────────────────────────────────

@app.post("/pdf-chat", response_model=PdfChatResponse)
async def pdf_chat(req: PdfChatRequest):
    pdf_name = req.pdf_name.strip() or "document.pdf"
    logger.info(f"/pdf-chat  file={pdf_name!r}  msg={req.message[:80]!r}")
    refined_message = await refine_prompt(req.message, req.history)
    try:
        pdf_bytes = base64.b64decode(req.pdf_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 PDF data")
    pdf_text = ""
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for i, page in enumerate(reader.pages):
            pdf_text += f"\n--- Page {i+1} ---\n{page.extract_text() or ''}"
        logger.info(f"/pdf-chat  {len(pdf_text)} chars from {len(reader.pages)} pages")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {e}")
    if not pdf_text.strip():
        raise HTTPException(status_code=422, detail="PDF appears empty or image-only. Please try a text-based PDF.")
    MAX_PDF_CHARS = 60_000
    if len(pdf_text) > MAX_PDF_CHARS:
        pdf_text = pdf_text[:MAX_PDF_CHARS] + "\n\n[... document truncated to fit context ...]"
    doc_context = (
        f'The user has uploaded a PDF named "{pdf_name}".\n'
        f"=== DOCUMENT START ===\n{pdf_text}\n=== DOCUMENT END ==="
    )
    messages = [{"role": "system", "content": PDF_SYSTEM_PROMPT + "\n\n" + doc_context}]
    sanitised: list[dict] = []
    for entry in req.history[-10:]:
        role = entry.role if entry.role in ("user", "assistant") else "user"
        if sanitised and sanitised[-1]["role"] == role:
            sanitised[-1]["content"] += "\n" + entry.content
        else:
            sanitised.append({"role": role, "content": entry.content})
    messages.extend(sanitised)
    messages.append({"role": "user", "content": refined_message})
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, max_tokens=1500, temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Groq error /pdf-chat: {e}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    reply = response.choices[0].message.content.strip()
    logger.info(f"/pdf-chat  reply_len={len(reply)}")
    return PdfChatResponse(reply=reply, pdf_name=pdf_name)


# ── /code-questions ────────────────────────────────────────────────────────────

@app.post("/code-questions", response_model=CodeQuestionsResponse)
async def code_questions(req: CodeQuestionsRequest):
    logger.info(f"/code-questions  msg={req.message[:80]!r}")
    prompt = CODE_QUESTIONS_PROMPT.format(request=req.message.strip())
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": "Generate the clarifying questions now."},
            ],
            max_tokens=400,
            temperature=0.4,
        )
    except Exception as e:
        logger.error(f"Groq error /code-questions: {e}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    raw = response.choices[0].message.content.strip()
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean)
        if not isinstance(data, list):
            data = []
        questions = [
            CodeQuestion(
                question=str(q.get("question", "")),
                options=[str(o) for o in q.get("options", [])][:4]
            )
            for q in data
            if isinstance(q, dict) and q.get("question") and q.get("options")
        ]
        logger.info(f"/code-questions  generated {len(questions)} questions")
        return CodeQuestionsResponse(questions=questions)
    except Exception as e:
        logger.warning(f"/code-questions  parse failed: {e}  raw={raw[:200]!r}")
        return CodeQuestionsResponse(questions=[])


# ── /generate-image ────────────────────────────────────────────────────────────

@app.post("/generate-image", response_model=ImageGenResponse)
async def generate_image(req: ImageGenRequest):
    logger.info(f"/generate-image  prompt={req.prompt[:80]!r}")

    if not REPLICATE_API_TOKEN or REPLICATE_API_TOKEN.endswith("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"):
        raise HTTPException(status_code=500, detail="REPLICATE_API_TOKEN is not configured.")

    # Refine the user's short prompt into a rich generation prompt
    refined_prompt = await refine_image_prompt(req.prompt)

    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            # Step 1: create prediction
            resp = await c.post(
                "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions",
                headers={
                    "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
                    "Content-Type": "application/json",
                    "Prefer": "wait",
                },
                json={
                    "input": {
                        "prompt": refined_prompt,
                        "num_outputs": 1,
                        "aspect_ratio": "1:1",
                        "output_format": "webp",
                        "output_quality": 90,
                    }
                },
            )
            if resp.status_code not in (200, 201):
                raise HTTPException(status_code=502, detail=f"Replicate rejected request: {resp.text[:200]}")

            prediction = resp.json()

            # If Prefer: wait returned a finished result immediately
            if prediction.get("status") == "succeeded":
                image_url = prediction["output"][0]
                logger.info(f"/generate-image  url={image_url[:80]}")
                return ImageGenResponse(image_url=image_url, prompt_used=refined_prompt)

            prediction_id = prediction.get("id")
            if not prediction_id:
                raise HTTPException(status_code=502, detail="No prediction ID returned from Replicate.")

            # Step 2: poll until done (max 30 attempts × 2s = 60s)
            for _ in range(30):
                await asyncio.sleep(2)
                poll = await c.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers={"Authorization": f"Bearer {REPLICATE_API_TOKEN}"},
                )
                result = poll.json()
                status = result.get("status")
                if status == "succeeded":
                    image_url = result["output"][0]
                    logger.info(f"/generate-image  url={image_url[:80]}")
                    return ImageGenResponse(image_url=image_url, prompt_used=refined_prompt)
                elif status == "failed":
                    raise HTTPException(status_code=502, detail=f"Replicate generation failed: {result.get('error','unknown')}")

            raise HTTPException(status_code=504, detail="Image generation timed out after 60 seconds.")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Replicate error /generate-image: {e}")
        raise HTTPException(status_code=502, detail=f"Image generation failed: {e}")
