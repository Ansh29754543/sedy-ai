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
app = FastAPI(title="Sedy API", version="2.4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Groq client ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set — API calls will fail!")

client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"

# ── Optional: SerpAPI / Serper key for real-time data ─────────────────────────
# Set SERPER_API_KEY env var to enable live web search for graph data.
# If not set, the AI falls back to its own knowledge (still good for historical data).
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
- When explaining topics, use markdown formatting: headers (##), bold (**text**), bullet points (- item), code blocks (```language)
- NEVER describe flashcards or quizzes in plain text — the frontend handles those separately
- Keep responses focused and well structured
- When the user refers to a previous topic (e.g. "it", "that", "the same topic"), use the conversation history

STRICT MATH FORMATTING RULES (frontend uses KaTeX):
- ALL mathematical expressions MUST be wrapped in LaTeX delimiters
- Use $...$ for inline math, $$...$$ for display equations
- NEVER write bare math like: w^2, dC/dw — always wrap in $...$
- For fractions: $\\frac{numerator}{denominator}$
- For subscripts: $A_{\\text{base}}$"""

REFINE_SYSTEM_PROMPT = """You are a silent prompt refinement engine. Your ONLY job is to clean up a student's message.

Perform these three tasks:
1. SPELLING FIX — Correct obvious typos (e.g. "photosinthesis" → "photosynthesis")
2. GRAMMAR FIX — Fix grammar while keeping original tone
3. CONTEXT RESOLUTION — If vague ("it", "this", "on it", "at this basis"), look at history and fill in the actual topic

CRITICAL: Output ONLY the refined message. No explanation, no preamble, no quotes.
If already perfect, output exactly as-is. Keep it concise."""

GRAPH_SYSTEM_PROMPT = """You are a data assistant generating chart data. Return ONLY a valid JSON object — no prose, no markdown, no explanation.

Schema:
{
  "title": "Short descriptive chart title",
  "chart_type": "line",
  "unit": "unit of Y axis e.g. Rs/kg or USD billions or %",
  "x_label": "X axis label e.g. Year",
  "caption": "One optional sentence of context (or empty string)",
  "series": [
    {
      "label": "Series name",
      "data": [{"x": "2000", "y": 8.5}, ...]
    }
  ]
}

chart_type must be exactly one of: "line", "bar", "pie"
- Use "pie" only when the request is clearly about proportions/shares (market share, composition, breakdown)
- Use "bar" when comparing discrete categories or when user says "bar chart"
- Use "line" for trends over time (default)

For "pie" charts: use only ONE series, and x values become the slice labels
For "bar" charts: can have multiple series (grouped bars)
For "line" charts: can have multiple series

CRITICAL RULES:
- Return ONLY valid JSON. No backticks, no markdown, no preamble whatsoever.
- x values MUST be strings. y values MUST be numbers.
- Use realistic, historically accurate data. Use well-known approximations.
- For Indian commodity prices use Rs/kg. For GDP use USD billions.
- Include 5-20 data points for time series. For pie, 3-8 slices.
- Multiple items requested → multiple series objects (except pie which uses one series).
- Keep series labels short (1-3 words)."""

PDF_SYSTEM_PROMPT = """You are Sedy, an intelligent student learning assistant made by Ansh Verma.
The user has uploaded a PDF document. Help them understand it.
Answer questions, summarize, explain concepts, generate study notes.
Use LaTeX math formatting: $...$ inline, $$...$$ display.
Only answer based on document content. If not in the document, say so."""


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
    chart_type: str = "auto"   # "auto" | "line" | "bar" | "pie"
    history: list[HistoryEntry] = []

class PdfChatRequest(BaseModel):
    message: str
    pdf_base64: str
    pdf_name: str = "document.pdf"
    history: list[HistoryEntry] = []

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
    chart_type: str = "line"   # "line" | "bar" | "pie"
    unit: str = ""
    x_label: str = ""
    caption: str = ""
    data_source: str = ""      # "live" | "estimated" — shown as badge on frontend
    series: list[GraphSeries]

class PdfChatResponse(BaseModel):
    reply: str
    pdf_name: str


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
    reversed_history = list(reversed(history))
    for entry in reversed_history:
        if entry.role == "assistant":
            m = re.search(r'(?:flashcards?|quiz)\s+(?:about|on)\s+([^\n.!?]{3,60})', entry.content, re.I)
            if m:
                return m.group(1).strip()
    action_words = r'flashcards?|flash\s*cards?|quiz(zes)?|make|create|generate|about|on|me|test|cards?|and\b|a\b|the\b'
    for entry in reversed_history:
        if entry.role == "user":
            cleaned = re.sub(action_words, '', entry.content, flags=re.I).strip()
            if cleaned and not is_vague_topic(cleaned) and len(cleaned) > 2:
                return cleaned
    for entry in reversed_history:
        if entry.role == "assistant":
            m = re.search(r'(?:^|\n)#{1,3}\s+(?:introduction to\s+)?([A-Za-z][^\n]{2,50})', entry.content, re.I)
            if m:
                return m.group(1).strip()
    return raw_topic

async def refine_prompt(message: str, history: list[HistoryEntry]) -> str:
    if not message or not message.strip():
        return message
    history_snippet = ""
    if history:
        lines = []
        for entry in history[-10:]:
            role_label = "Student" if entry.role == "user" else "Tutor"
            lines.append(f"{role_label}: {entry.content[:200]}")
        history_snippet = "\n".join(lines)
    user_content = (
        f"Conversation history so far:\n{history_snippet}\n\nNew message to refine: {message}"
        if history_snippet else f"Message to refine: {message}"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": REFINE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=256,
            temperature=0.2,
        )
        refined = response.choices[0].message.content.strip()
        if not refined or len(refined) > len(message) * 4:
            return message
        logger.info(f"refine: '{message[:50]}' → '{refined[:50]}'")
        return refined
    except Exception as e:
        logger.warning(f"refine_prompt failed: {e}")
        return message


# ── Live web search for real-time data ────────────────────────────────────────

async def fetch_live_data(query: str) -> str:
    """
    Fetch real-time search snippets using Serper.dev API.
    Returns a string of search result snippets to feed to the graph AI,
    or empty string if SERPER_API_KEY is not configured.
    """
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
                logger.warning(f"Serper returned {resp.status_code}")
                return ""
            data = resp.json()
            snippets = []
            for r in data.get("organic", [])[:5]:
                title   = r.get("title", "")
                snippet = r.get("snippet", "")
                if title or snippet:
                    snippets.append(f"• {title}: {snippet}")
            result = "\n".join(snippets)
            logger.info(f"live_data: fetched {len(snippets)} snippets for query={query!r}")
            return result
    except Exception as e:
        logger.warning(f"live_data fetch failed: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# ── ROUTES ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "status": "Sedy API is live 🚀",
        "model": MODEL,
        "version": "2.4.0",
        "live_data": bool(SERPER_API_KEY),
        "endpoints": ["/chat", "/flashcards", "/quiz", "/graph", "/pdf-chat"],
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    logger.info(f"/chat  history={len(req.history)}  msg={req.message[:80]!r}")
    refined = await refine_prompt(req.message, req.history)
    messages = build_messages(SYSTEM_PROMPT, req.history, refined)
    try:
        response = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=1024, temperature=0.7)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    reply = response.choices[0].message.content.strip()
    logger.info(f"/chat  reply_len={len(reply)}")
    return ChatResponse(reply=reply)


@app.post("/flashcards", response_model=FlashcardResponse)
async def flashcards(req: FlashcardRequest):
    topic_as_prompt = f"Generate flashcards about {req.topic.strip()}"
    refined = await refine_prompt(topic_as_prompt, req.history)
    m = re.search(r'(?:flashcards?\s+(?:about|on|for)\s+|flashcards?\s+)(.+)', refined, re.I)
    topic = m.group(1).strip() if m else refined.strip()
    topic = resolve_topic_from_history(topic, req.history)
    if not topic:
        raise HTTPException(status_code=400, detail="topic must not be empty")
    count = max(1, min(req.count, 20))
    logger.info(f"/flashcards  topic={topic!r}  count={count}")
    user_prompt = (
        f'Generate exactly {count} flashcards about "{topic}".\n'
        f'Return ONLY a JSON array, no markdown:\n'
        f'[{{"question":"...","answer":"..."}}]'
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_prompt}],
            max_tokens=1500, temperature=0.7
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    raw = response.choices[0].message.content.strip()
    try:
        data = extract_json_array(raw)
        cards = [Flashcard(question=str(c.get("question") or c.get("front","")), answer=str(c.get("answer") or c.get("back",""))) for c in data if isinstance(c,dict)]
        cards = [c for c in cards if c.question and c.answer]
    except Exception:
        cards = [Flashcard(question=f"What is {topic}?", answer=raw[:300])]
    return FlashcardResponse(cards=cards, topic=topic)


@app.post("/quiz", response_model=QuizResponse)
async def quiz(req: QuizRequest):
    topic_as_prompt = f"Quiz me on {req.topic.strip()}"
    refined = await refine_prompt(topic_as_prompt, req.history)
    m = re.search(r'(?:quiz\s+(?:me\s+)?(?:on|about)\s+)(.+)', refined, re.I)
    topic = m.group(1).strip() if m else refined.strip()
    topic = resolve_topic_from_history(topic, req.history)
    if not topic:
        raise HTTPException(status_code=400, detail="topic must not be empty")
    difficulty = req.difficulty if req.difficulty in ("easy","medium","hard") else "medium"
    count = max(1, min(req.count, 20))
    logger.info(f"/quiz  topic={topic!r}  difficulty={difficulty}  count={count}")
    user_prompt = (
        f'Generate exactly {count} {difficulty} MCQs about "{topic}".\n'
        f'Return ONLY JSON array, no markdown:\n'
        f'[{{"question":"...","options":["A","B","C","D"],"answer":0,"explanation":"..."}}]'
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_prompt}],
            max_tokens=2000, temperature=0.7
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    raw = response.choices[0].message.content.strip()
    try:
        data = extract_json_array(raw)
        questions = []
        for q in data:
            if not isinstance(q, dict): continue
            opts = q.get("options",[])
            while len(opts) < 4: opts.append("N/A")
            questions.append(QuizQuestion(
                question=str(q.get("question","")), options=opts[:4],
                answer=max(0,min(int(q.get("answer",0)),3)),
                explanation=str(q.get("explanation",""))
            ))
        questions = [q for q in questions if q.question]
    except Exception:
        questions = [QuizQuestion(question=f"Key concept in {topic}?", options=["A","B","C","D"], answer=0, explanation=raw[:300])]
    return QuizResponse(questions=questions, topic=topic, difficulty=difficulty)


@app.post("/graph", response_model=GraphResponse)
async def graph(req: GraphRequest):
    """
    Generate structured chart data. Supports line, bar, and pie charts.
    Optionally fetches live web search data if SERPER_API_KEY is configured.
    """
    logger.info(f"/graph  msg={req.message[:80]!r}  chart_type={req.chart_type!r}")

    refined = await refine_prompt(req.message, req.history)

    # ── Attempt live data fetch ────────────────────────────────────────────────
    live_snippets = ""
    data_source = "estimated"
    if SERPER_API_KEY:
        # Build a targeted search query from the user message
        search_query = refined + " data statistics"
        live_snippets = await fetch_live_data(search_query)
        if live_snippets:
            data_source = "live"
            logger.info(f"/graph  live data fetched ({len(live_snippets)} chars)")

    # ── Build prompt ───────────────────────────────────────────────────────────
    chart_type_hint = ""
    if req.chart_type != "auto":
        chart_type_hint = f'\nIMPORTANT: The user specifically requested a "{req.chart_type}" chart. Set chart_type to "{req.chart_type}".'

    if live_snippets:
        user_content = (
            f"User request: {refined}\n\n"
            f"Here is real-time data from the web to use:\n{live_snippets}\n\n"
            f"Use the web data above to populate the chart values as accurately as possible."
            f"{chart_type_hint}"
        )
    else:
        user_content = refined + chart_type_hint

    messages = [
        {"role": "system", "content": GRAPH_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    try:
        response = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=2500, temperature=0.3)
    except Exception as e:
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
            raise ValueError("No valid series in response")

        ct = str(obj.get("chart_type","line")).lower()
        if ct not in ("line","bar","pie"):
            ct = "line"

        result = GraphResponse(
            title=str(obj.get("title", refined[:60])),
            chart_type=ct,
            unit=str(obj.get("unit","")),
            x_label=str(obj.get("x_label","")),
            caption=str(obj.get("caption","")),
            data_source=data_source,
            series=series_list,
        )
        logger.info(f"/graph  title={result.title!r}  type={ct}  series={len(series_list)}  source={data_source}")
        return result

    except Exception as e:
        logger.warning(f"/graph  parse failed: {e}  raw={raw[:300]!r}")
        raise HTTPException(status_code=422, detail=f"Could not parse graph data: {e}")


@app.post("/pdf-chat", response_model=PdfChatResponse)
async def pdf_chat(req: PdfChatRequest):
    pdf_name = req.pdf_name.strip() or "document.pdf"
    logger.info(f"/pdf-chat  file={pdf_name!r}  msg={req.message[:80]!r}")

    refined_message = await refine_prompt(req.message, req.history)

    try:
        pdf_bytes = base64.b64decode(req.pdf_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 PDF data")

    pdf_text = ""
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for i, page in enumerate(reader.pages):
            pdf_text += f"\n--- Page {i+1} ---\n{page.extract_text() or ''}"
        logger.info(f"/pdf-chat  extracted {len(pdf_text)} chars from {len(reader.pages)} pages")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {e}")

    if not pdf_text.strip():
        raise HTTPException(status_code=422, detail="PDF appears empty or image-only. Try a text-based PDF.")

    MAX_PDF_CHARS = 60_000
    if len(pdf_text) > MAX_PDF_CHARS:
        pdf_text = pdf_text[:MAX_PDF_CHARS] + "\n\n[... truncated ...]"

    doc_context = f'PDF: "{pdf_name}"\n=== START ===\n{pdf_text}\n=== END ==='
    messages = [{"role":"system","content": PDF_SYSTEM_PROMPT + "\n\n" + doc_context}]

    trimmed = req.history[-10:]
    sanitised: list[dict] = []
    for entry in trimmed:
        role = entry.role if entry.role in ("user","assistant") else "user"
        if sanitised and sanitised[-1]["role"] == role:
            sanitised[-1]["content"] += "\n" + entry.content
        else:
            sanitised.append({"role": role, "content": entry.content})
    messages.extend(sanitised)
    messages.append({"role":"user","content": refined_message})

    try:
        response = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=1500, temperature=0.7)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

    reply = response.choices[0].message.content.strip()
    return PdfChatResponse(reply=reply, pdf_name=pdf_name)
