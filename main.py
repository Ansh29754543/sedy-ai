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

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("sedy")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sedy API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Groq client ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set — API calls will fail!")

client = Groq(api_key=GROQ_API_KEY)

MODEL        = "llama-3.3-70b-versatile"
MODEL_VISION = "meta-llama/llama-4-scout-17b-16e-instruct"

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")


# ══════════════════════════════════════════════════════════════════════════════
# ── SYSTEM PROMPTS ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are Sedy, an intelligent student learning assistant made by Ansh Verma, a school student.
You explain concepts clearly and simply, solve math and science problems step by step,
summarize topics, and help students understand programming.
Always be encouraging, clear and educational.
Only reveal your identity when asked.

IMPORTANT RULES:
- Use markdown formatting: headers (##), bold (**text**), bullet points (- item), code blocks
- NEVER describe flashcards or quizzes in plain text — the frontend handles those separately
- Keep responses focused and well structured
- Use conversation history for context when user refers to "it", "that", "same topic"

STRICT MATH FORMATTING (frontend uses KaTeX):
- ALL math MUST be wrapped in LaTeX: $...$ inline, $$...$$ display
- NEVER write bare math like w^2 — always wrap in $w^2$
- Fractions: $\\frac{a}{b}$  Subscripts: $A_{\\text{base}}$

FOLLOW-UP SUGGESTIONS — REQUIRED FOR EVERY RESPONSE:
After your reply, on a new line output EXACTLY:
SUGGESTIONS: <suggestion 1> | <suggestion 2> | <suggestion 3>
- Each suggestion: 4-8 words, a natural follow-up question
- Example: SUGGESTIONS: How does mitosis differ from meiosis? | What triggers cell division? | Quiz me on this topic
- The frontend strips this line — users never see the raw format"""

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

IMAGE_SYSTEM_PROMPT = """You are Sedy, an intelligent student learning assistant made by Ansh Verma.
The user has uploaded an image — a textbook page, handwritten notes, diagram, or math problem.

Your job:
1. Carefully read everything visible in the image
2. If the user asks a specific question, answer it
3. If no question, provide helpful analysis:
   - Math problems → solve step by step
   - Diagrams → explain what they show
   - Notes/text → summarize key points
   - Handwritten work → read and explain

Use markdown formatting. Use LaTeX for math: $...$ inline, $$...$$ display.

FOLLOW-UP SUGGESTIONS — REQUIRED:
After your reply output EXACTLY:
SUGGESTIONS: <suggestion 1> | <suggestion 2> | <suggestion 3>"""

# ──────────────────────────────────────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """You are an intent classifier for a student learning app.
Output EXACTLY one word from: graph, flashcard, quiz, both, chat

graph     = wants a chart/graph/visualisation
flashcard = wants flip study cards
quiz      = wants MCQ quiz
both      = wants flashcards AND a quiz
chat      = everything else

No punctuation, no explanation. One word only."""


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

class ImageChatRequest(BaseModel):
    message: str
    image_base64: str
    image_type: str = "image/jpeg"

class ChatResponse(BaseModel):
    reply: str
    suggestions: list[str] = []

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

class ImageChatResponse(BaseModel):
    reply: str
    suggestions: list[str] = []

class IntentRequest(BaseModel):
    message: str
    history: list[HistoryEntry] = []

class IntentResponse(BaseModel):
    intent: str


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


def parse_suggestions(reply: str) -> tuple[str, list[str]]:
    """Extract SUGGESTIONS: a | b | c from reply, return (clean_reply, [a, b, c])"""
    suggestions = []
    clean_lines = []
    for line in reply.strip().split('\n'):
        if line.strip().startswith("SUGGESTIONS:"):
            raw = line.strip()[len("SUGGESTIONS:"):].strip()
            suggestions = [s.strip() for s in raw.split('|') if s.strip()][:3]
        else:
            clean_lines.append(line)
    return '\n'.join(clean_lines).strip(), suggestions


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
        return refined
    except Exception as e:
        logger.warning(f"refine_prompt failed ({e}), using original")
        return message


async def refine_graph_prompt(message: str, history: list[HistoryEntry]) -> str:
    return await refine_prompt(message, history, system_prompt=GRAPH_REFINE_SYSTEM_PROMPT)


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
                t, s = r.get("title",""), r.get("snippet","")
                if t or s: snippets.append(f"• {t}: {s}")
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
        "version": "3.0.0",
        "model": MODEL,
        "model_vision": MODEL_VISION,
        "features": ["voice_input", "text_to_speech", "follow_up_suggestions", "image_upload"],
        "endpoints": ["/chat", "/flashcards", "/quiz", "/graph", "/pdf-chat", "/image-chat", "/intent"],
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
        intent = raw if raw in ("graph", "flashcard", "quiz", "both", "chat") else "chat"
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
    raw_reply = response.choices[0].message.content.strip()
    reply, suggestions = parse_suggestions(raw_reply)
    logger.info(f"/chat  reply_len={len(reply)}  suggestions={len(suggestions)}")
    return ChatResponse(reply=reply, suggestions=suggestions)


# ── /image-chat ────────────────────────────────────────────────────────────────

@app.post("/image-chat", response_model=ImageChatResponse)
async def image_chat(req: ImageChatRequest):
    """Vision endpoint — analyses images using Groq's llama-4-scout vision model."""
    logger.info(f"/image-chat  type={req.image_type}  msg={req.message[:80]!r}")

    allowed = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif"}
    img_type = req.image_type.lower()
    if img_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {img_type}")

    user_message_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{img_type};base64,{req.image_base64}"}
        },
        {
            "type": "text",
            "text": req.message.strip() or "Please analyse this image and help me understand it."
        }
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_VISION,
            messages=[
                {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message_content},
            ],
            max_tokens=1024,
            temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Groq error /image-chat: {e}")
        raise HTTPException(status_code=502, detail=f"Groq vision API error: {e}")

    raw_reply = response.choices[0].message.content.strip()
    reply, suggestions = parse_suggestions(raw_reply)
    logger.info(f"/image-chat  reply_len={len(reply)}  suggestions={len(suggestions)}")
    return ImageChatResponse(reply=reply, suggestions=suggestions)


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
        f'Return ONLY a JSON array:\n[{{"question":"...","answer":"..."}}]'
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            max_tokens=1500, temperature=0.7,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    raw = response.choices[0].message.content.strip()
    try:
        data = extract_json_array(raw)
        cards = [Flashcard(question=str(c.get("question") or c.get("front","")), answer=str(c.get("answer") or c.get("back",""))) for c in data if isinstance(c, dict)]
        cards = [c for c in cards if c.question and c.answer]
    except Exception:
        cards = [Flashcard(question=f"What is {topic}?", answer=raw[:300])]
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
        f'Return ONLY JSON array:\n[{{"question":"...","options":["A","B","C","D"],"answer":0,"explanation":"..."}}]'
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            max_tokens=2000, temperature=0.7,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    raw = response.choices[0].message.content.strip()
    try:
        data = extract_json_array(raw)
        questions = []
        for q in data:
            if not isinstance(q, dict): continue
            opts = q.get("options", [])
            while len(opts) < 4: opts.append("N/A")
            questions.append(QuizQuestion(
                question=str(q.get("question","")), options=opts[:4],
                answer=max(0, min(int(q.get("answer",0)), 3)),
                explanation=str(q.get("explanation","")),
            ))
        questions = [q for q in questions if q.question]
    except Exception:
        questions = [QuizQuestion(question=f"Key concept in {topic}?", options=["A","B","C","D"], answer=0, explanation=raw[:300])]
    return QuizResponse(questions=questions, topic=topic, difficulty=difficulty)


# ── /graph ─────────────────────────────────────────────────────────────────────

@app.post("/graph", response_model=GraphResponse)
async def graph(req: GraphRequest):
    logger.info(f"/graph  msg={req.message[:80]!r}")
    refined = await refine_graph_prompt(req.message, req.history)
    live_snippets = ""
    data_source = "estimated"
    if SERPER_API_KEY:
        live_snippets = await fetch_live_data(refined + " data statistics numbers")
        if live_snippets: data_source = "live"
    user_content = (
        f"User request: {refined}\n\nReal-time data:\n{live_snippets}\n\nGenerate chart JSON."
        if live_snippets else refined
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": GRAPH_SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            max_tokens=2500, temperature=0.3,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    raw = strip_json_fences(response.choices[0].message.content.strip())
    try:
        obj = json.loads(raw)
        series_list = []
        for i, s in enumerate(obj.get("series", [])):
            pts = []
            for pt in s.get("data", []):
                try: pts.append(GraphPoint(x=str(pt["x"]), y=float(pt["y"])))
                except: continue
            if pts: series_list.append(GraphSeries(label=str(s.get("label", f"Series {i+1}")), data=pts))
        if not series_list: raise ValueError("No valid series")
        ct = str(obj.get("chart_type","line")).lower().strip()
        if ct not in ("line","bar","pie"): ct = "line"
        return GraphResponse(
            title=str(obj.get("title", refined[:60])), chart_type=ct,
            unit=str(obj.get("unit","")), x_label=str(obj.get("x_label","")),
            caption=str(obj.get("caption","")), data_source=data_source, series=series_list,
        )
    except Exception as e:
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
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {e}")
    if not pdf_text.strip():
        raise HTTPException(status_code=422, detail="PDF appears empty or image-only.")
    if len(pdf_text) > 60_000:
        pdf_text = pdf_text[:60_000] + "\n\n[... truncated ...]"
    doc_context = f'PDF: "{pdf_name}"\n=== DOCUMENT ===\n{pdf_text}\n=== END ==='
    messages = [{"role": "system", "content": PDF_SYSTEM_PROMPT + "\n\n" + doc_context}]
    sanitised: list[dict] = []
    for entry in req.history[-10:]:
        role = entry.role if entry.role in ("user","assistant") else "user"
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
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    return PdfChatResponse(reply=response.choices[0].message.content.strip(), pdf_name=pdf_name)


# ── /code-questions ────────────────────────────────────────────────────────────

CODE_QUESTIONS_PROMPT = """You are helping clarify a coding request before writing code.
The user has asked: "{request}"

Generate 2-3 SHORT, RELEVANT multiple-choice questions to clarify missing details.

STRICT RULES:
- NEVER ask about something the user already mentioned (e.g. language, framework, topic)
- NEVER ask generic or obvious questions
- Ask about things that will genuinely change how the code is written
- Good question topics: complexity level, extra features, UI style, error handling, data storage, target audience
- Each question must have 3-4 short options (1-5 words each)
- If the request is already very detailed and nothing is missing, return an empty array []
- Return ONLY a valid JSON array, no markdown fences, no explanation

FORMAT:
[
  {{"question": "Question text?", "options": ["Option A", "Option B", "Option C"]}},
  {{"question": "Another question?", "options": ["Option A", "Option B", "Option C", "Option D"]}}
]"""


class CodeQuestionsRequest(BaseModel):
    message: str


class CodeQuestion(BaseModel):
    question: str
    options: list[str]


class CodeQuestionsResponse(BaseModel):
    questions: list[CodeQuestion]


@app.post("/code-questions", response_model=CodeQuestionsResponse)
async def code_questions(req: CodeQuestionsRequest):
    logger.info(f"/code-questions  msg={req.message[:80]!r}")
    prompt = CODE_QUESTIONS_PROMPT.format(request=req.message.strip())
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": "Generate the questions now."},
            ],
            max_tokens=400,
            temperature=0.4,
        )
    except Exception as e:
        logger.error(f"Groq error /code-questions: {e}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
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
