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
app = FastAPI(title="Sedy API", version="2.5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Groq clients ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set — API calls will fail!")

client = Groq(api_key=GROQ_API_KEY)

# Smart model routing — use 70B for chat/graphs, 8B for flashcards/quiz/PDF (saves ~10x tokens)
MODEL_SMART = "llama-3.3-70b-versatile"   # chat, graphs, refinement
MODEL_FAST  = "llama-3.1-8b-instant"      # flashcards, quiz, PDF summarisation

# Optional: Serper.dev key for live web data on graphs
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

# ──────────────────────────────────────────────────────────────────────────────

REFINE_SYSTEM_PROMPT = """You are a silent prompt refinement engine. Your ONLY job is to clean up a student's message before it reaches an AI tutor.

Perform these three tasks in order:
1. SPELLING FIX — Correct obvious typos (e.g. "photosinthesis" → "photosynthesis", "grpah" → "graph")
2. GRAMMAR FIX — Fix grammar while keeping the original casual tone intact
3. CONTEXT RESOLUTION — If the message is vague ("it", "this", "that", "on it", "at this basis", "same topic"), look at the conversation history and rewrite with the actual topic filled in.
   Examples:
   - History has "flashcards about mitosis", user says "quiz me on it" → "Quiz me on mitosis"
   - History has "onion prices graph", user says "now show tomato too" → "Show onion and tomato prices graph"
   - If nothing relevant in history, leave as-is after spelling/grammar fix

CRITICAL RULES:
- Output ONLY the refined message. No explanation, no preamble, no quotes, no commentary.
- Never change the intent or add information not implied by the original message.
- Never refuse. Just output the cleaned text.
- If already perfect, output exactly as-is."""

# ──────────────────────────────────────────────────────────────────────────────

GRAPH_REFINE_SYSTEM_PROMPT = """You are a silent prompt refinement engine specialised for data visualisation requests.

Your job is to clean up and clarify a user's graph/chart request before it's sent to a data assistant.

Perform these tasks:
1. SPELLING FIX — Fix typos (e.g. "onoin" → "onion", "prise" → "price")
2. GRAMMAR FIX — Fix grammar while keeping casual tone
3. CONTEXT RESOLUTION — If vague references appear ("it", "same", "that topic", "add that too"), use the conversation history to resolve them.
   Examples:
   - History: "show onion prices", user says "show tomato too" → "Show onion and tomato prices from 2000 to 2020"
   - History: "GDP of India graph", user says "add China" → "Show GDP of India and China"
   - History: "plot wheat prices", user says "make it a bar chart" → "Bar chart of wheat prices"
4. CHART TYPE HINT — If the user's phrasing strongly implies a specific chart type, append a hint at the end:
   - Words like "breakdown", "composition", "share", "proportion", "percentage of" → append "(pie chart)"
   - Words like "compare", "vs", "side by side", "ranking", "top N" with discrete categories → append "(bar chart)"
   - Time-based trends, "over time", "from X to Y", historical data → append "(line chart)"
   - If unclear, do not append anything

CRITICAL: Output ONLY the refined message. No explanation, no quotes, no preamble."""

# ──────────────────────────────────────────────────────────────────────────────

GRAPH_SYSTEM_PROMPT = """You are a data assistant. Your ONLY output must be a single valid JSON object — no prose, no markdown fences, no explanation before or after.

SCHEMA:
{
  "title": "Short descriptive chart title (max 60 chars)",
  "chart_type": "line",
  "unit": "unit of Y axis e.g. Rs/kg or USD billions or % (empty string if not applicable)",
  "x_label": "X axis label e.g. Year or Month (empty string if obvious)",
  "caption": "One optional sentence of context, source note, or data caveat (empty string if none)",
  "series": [
    {
      "label": "Series name (1-3 words)",
      "data": [{"x": "2000", "y": 8.5}, ...]
    }
  ]
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHART TYPE DECISION RULES — you must pick the best type autonomously:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use "line" when:
  - Data shows change over continuous time (years, months, decades)
  - User says "trend", "over time", "from X to Y", "history", "growth"
  - Multiple things compared over time (e.g. India vs China GDP 2000-2023)
  Examples: "onion prices 2000-2020", "GDP growth over 20 years", "temperature by month"

Use "bar" when:
  - Comparing discrete, named categories (not a time series)
  - User says "compare", "vs", "ranking", "top N", "by country/region/category"
  - Data is a snapshot in time across multiple items
  Examples: "top 5 languages by speakers", "GDP of India China USA in 2023", "marks by subject"

Use "pie" when:
  - Data represents parts of a whole (shares, proportions, composition, breakdown)
  - Values add up to 100% or a meaningful total
  - User says "share", "breakdown", "composition", "proportion", "percentage of", "distribution"
  Examples: "energy sources breakdown", "market share of smartphones", "religions by followers %"
  IMPORTANT for pie: use exactly ONE series; x values are the slice labels

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Return ONLY valid JSON. Absolutely nothing outside the JSON object.
- x values MUST be strings ("2005", "Jan", "India"). y values MUST be numbers (12.5 not "12.5").
- Use realistic, historically plausible data. If estimating, use well-known approximations.
- Indian commodity prices: use Rs/kg. GDP: use USD billions. Percentages: use % unit.
- Time series: include 8-20 data points. Pie: 3-8 slices. Bar: 3-10 groups.
- Multiple items requested → multiple series (except pie, which always has 1 series).
- Series labels: 1-3 words max."""

# ──────────────────────────────────────────────────────────────────────────────

PDF_SYSTEM_PROMPT = """You are Sedy, an intelligent student learning assistant made by Ansh Verma.
The user has uploaded a PDF document. Help them understand it.
Answer questions, summarize sections, explain concepts, generate study notes.
Use LaTeX math formatting: $...$ inline, $$...$$ display.
Only answer based on the document content. If something isn't in the document, say so clearly."""


# ══════════════════════════════════════════════════════════════════════════════
# ── PYDANTIC MODELS ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class HistoryEntry(BaseModel):
    role: str        # "user" or "assistant"
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
    # chart_type kept for backwards compat but ignored — AI decides autonomously
    chart_type: str = "auto"
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
    chart_type: str = "line"      # "line" | "bar" | "pie" — decided by AI
    unit: str = ""
    x_label: str = ""
    caption: str = ""
    data_source: str = ""         # "live" | "estimated"
    series: list[GraphSeries]

class PdfChatResponse(BaseModel):
    reply: str
    pdf_name: str


# ══════════════════════════════════════════════════════════════════════════════
# ── HELPERS ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def build_messages(system: str, history: list[HistoryEntry], user_message: str) -> list[dict]:
    """Build the messages array with sanitised alternating roles."""
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
    """Extract a JSON array from a string that may contain markdown fences."""
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
    """Strip markdown code fences from a JSON object string."""
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
    """
    If raw_topic is vague, scan history to find the last concrete subject.
    Priority: assistant summary → user message → heading in assistant reply.
    """
    if not is_vague_topic(raw_topic):
        return raw_topic
    if not history:
        return raw_topic

    rev = list(reversed(history))

    # 1. Assistant messages that record "flashcards/quiz about X"
    for entry in rev:
        if entry.role == "assistant":
            m = re.search(r'(?:flashcards?|quiz)\s+(?:about|on)\s+([^\n.!?]{3,60})', entry.content, re.I)
            if m:
                logger.info(f"resolve_topic → assistant log: {m.group(1).strip()!r}")
                return m.group(1).strip()

    # 2. User messages — strip action words
    action_words = r'flashcards?|flash\s*cards?|quiz(zes)?|make|create|generate|about|on|me|test|cards?|and\b|a\b|the\b'
    for entry in rev:
        if entry.role == "user":
            cleaned = re.sub(action_words, '', entry.content, flags=re.I).strip()
            if cleaned and not is_vague_topic(cleaned) and len(cleaned) > 2:
                logger.info(f"resolve_topic → user message: {cleaned!r}")
                return cleaned

    # 3. Heading in last assistant reply
    for entry in rev:
        if entry.role == "assistant":
            m = re.search(r'(?:^|\n)#{1,3}\s+(?:introduction to\s+)?([A-Za-z][^\n]{2,50})', entry.content, re.I)
            if m:
                logger.info(f"resolve_topic → heading: {m.group(1).strip()!r}")
                return m.group(1).strip()

    logger.warning(f"resolve_topic: could not resolve {raw_topic!r}")
    return raw_topic


# ── Generic prompt refiner ────────────────────────────────────────────────────

async def refine_prompt(message: str, history: list[HistoryEntry],
                        system_prompt: str = REFINE_SYSTEM_PROMPT) -> str:
    """
    Silently refine the user's message:
      1. Fix spelling/typos
      2. Fix grammar
      3. Resolve vague context references from history

    The user NEVER sees this — the frontend always displays the original message.
    Falls back to the original on any error.
    Accepts an optional custom system_prompt so graph requests use GRAPH_REFINE_SYSTEM_PROMPT.
    """
    if not message or not message.strip():
        return message

    history_snippet = ""
    if history:
        lines = []
        for entry in history[-10:]:
            label = "Student" if entry.role == "user" else "Tutor"
            lines.append(f"{label}: {entry.content[:200]}")
        history_snippet = "\n".join(lines)

    user_content = (
        f"Conversation history:\n{history_snippet}\n\nMessage to refine: {message}"
        if history_snippet else f"Message to refine: {message}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_SMART,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=256,
            temperature=0.2,
        )
        refined = response.choices[0].message.content.strip()

        # Safety: if output is suspiciously long, fall back
        if not refined or len(refined) > len(message) * 5:
            logger.warning(f"refine_prompt: suspicious output, falling back")
            return message

        logger.info(f"refine: '{message[:50]}' → '{refined[:50]}'")
        return refined

    except Exception as e:
        logger.warning(f"refine_prompt failed ({e}), using original")
        return message


async def refine_graph_prompt(message: str, history: list[HistoryEntry]) -> str:
    """Graph-specific refinement using GRAPH_REFINE_SYSTEM_PROMPT."""
    return await refine_prompt(message, history, system_prompt=GRAPH_REFINE_SYSTEM_PROMPT)


# ── Live web search for real-time graph data ──────────────────────────────────

async def fetch_live_data(query: str) -> str:
    """
    Fetch real-time search snippets via Serper.dev.
    Returns a string of snippets to inject into the graph prompt,
    or empty string if SERPER_API_KEY is not set.
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
            logger.info(f"live_data: {len(snippets)} snippets for {query!r}")
            return result
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
        "version": "2.5.0",
        "model_smart": MODEL_SMART,
        "model_fast": MODEL_FAST,
        "live_data": bool(SERPER_API_KEY),
        "endpoints": ["/chat", "/flashcards", "/quiz", "/graph", "/pdf-chat"],
    }


# ── /chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint. Silently refines the message before sending to the model.
    Uses MODEL_SMART (70B) for best quality responses.
    """
    logger.info(f"/chat  history={len(req.history)}  msg={req.message[:80]!r}")

    refined = await refine_prompt(req.message, req.history)
    messages = build_messages(SYSTEM_PROMPT, req.history, refined)

    try:
        response = client.chat.completions.create(
            model=MODEL_SMART,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
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
    """
    Generate flashcards. Uses MODEL_FAST (8B) to save tokens.
    Silently refines topic and resolves vague references.
    """
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
            model=MODEL_FAST,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=1500,
            temperature=0.7,
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
    """
    Generate MCQ quiz. Uses MODEL_FAST (8B) to save tokens.
    Silently refines topic and resolves vague references.
    """
    topic_as_prompt = f"Quiz me on {req.topic.strip()}"
    refined = await refine_prompt(topic_as_prompt, req.history)

    m = re.search(r'(?:quiz\s+(?:me\s+)?(?:on|about)\s+)(.+)', refined, re.I)
    topic = m.group(1).strip() if m else refined.strip()
    topic = resolve_topic_from_history(topic, req.history)

    if not topic:
        raise HTTPException(status_code=400, detail="topic must not be empty")

    difficulty = req.difficulty if req.difficulty in ("easy", "medium", "hard") else "medium"
    count = max(1, min(req.count, 20))
    logger.info(f"/quiz  topic={topic!r}  difficulty={difficulty}  count={count}")

    user_prompt = (
        f'Generate exactly {count} {difficulty} MCQs about "{topic}".\n'
        f'Return ONLY JSON array, no markdown:\n'
        f'[{{"question":"...","options":["A","B","C","D"],"answer":0,"explanation":"..."}}]'
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_FAST,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=2000,
            temperature=0.7,
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
    """
    Generate structured chart data. The AI autonomously decides whether to use
    a line, bar, or pie chart based on the nature of the request.

    Pipeline:
      1. Graph-specific prompt refinement (spelling + grammar + context resolution + chart type hint)
      2. Optional live data fetch via Serper.dev (if SERPER_API_KEY is set)
      3. AI generates full JSON with chart_type, series, unit, title, caption
    """
    logger.info(f"/graph  msg={req.message[:80]!r}")

    # Step 1 — Graph-specific refinement (uses GRAPH_REFINE_SYSTEM_PROMPT)
    # This also appends a chart type hint like "(pie chart)" when the phrasing implies it
    refined = await refine_graph_prompt(req.message, req.history)
    logger.info(f"/graph  refined='{refined[:80]}'")

    # Step 2 — Optional live data fetch
    live_snippets = ""
    data_source = "estimated"
    if SERPER_API_KEY:
        search_query = refined + " data statistics numbers"
        live_snippets = await fetch_live_data(search_query)
        if live_snippets:
            data_source = "live"
            logger.info(f"/graph  live data: {len(live_snippets)} chars")

    # Step 3 — Build user content for the graph AI
    if live_snippets:
        user_content = (
            f"User request: {refined}\n\n"
            f"Real-time data from web search (use these values where possible):\n"
            f"{live_snippets}\n\n"
            f"Generate the chart JSON using the web data above for accuracy."
        )
    else:
        user_content = refined

    messages = [
        {"role": "system", "content": GRAPH_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_SMART,
            messages=messages,
            max_tokens=2500,
            temperature=0.3,   # low temp for consistent structured JSON
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
                series_list.append(GraphSeries(
                    label=str(s.get("label", f"Series {i+1}")),
                    data=pts,
                ))

        if not series_list:
            raise ValueError("No valid series found in model response")

        # Validate chart_type — AI picks it; we just sanitise
        ct = str(obj.get("chart_type", "line")).lower().strip()
        if ct not in ("line", "bar", "pie"):
            logger.warning(f"/graph  unknown chart_type={ct!r}, defaulting to line")
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
        logger.info(
            f"/graph  title={result.title!r}  type={ct}  "
            f"series={len(series_list)}  points={sum(len(s.data) for s in series_list)}  "
            f"source={data_source}"
        )
        return result

    except Exception as e:
        logger.warning(f"/graph  parse failed: {e}  raw={raw[:300]!r}")
        raise HTTPException(status_code=422, detail=f"Could not parse graph data: {e}")


# ── /pdf-chat ──────────────────────────────────────────────────────────────────

@app.post("/pdf-chat", response_model=PdfChatResponse)
async def pdf_chat(req: PdfChatRequest):
    """
    PDF Q&A. Extracts text with pypdf, sends to MODEL_FAST (8B) to save tokens.
    Silently refines the user's message before processing.
    """
    pdf_name = req.pdf_name.strip() or "document.pdf"
    logger.info(f"/pdf-chat  file={pdf_name!r}  msg={req.message[:80]!r}")

    refined_message = await refine_prompt(req.message, req.history)

    # Decode PDF
    try:
        pdf_bytes = base64.b64decode(req.pdf_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 PDF data")

    # Extract text
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
        raise HTTPException(
            status_code=422,
            detail="PDF appears empty or image-only. Please try a text-based PDF."
        )

    # Trim to fit context (~60k chars safe for llama 128k context)
    MAX_PDF_CHARS = 60_000
    if len(pdf_text) > MAX_PDF_CHARS:
        pdf_text = pdf_text[:MAX_PDF_CHARS] + "\n\n[... document truncated to fit context ...]"
        logger.warning(f"/pdf-chat  truncated to {MAX_PDF_CHARS} chars")

    doc_context = (
        f'The user has uploaded a PDF named "{pdf_name}".\n'
        f"=== DOCUMENT START ===\n{pdf_text}\n=== DOCUMENT END ==="
    )
    messages = [{"role": "system", "content": PDF_SYSTEM_PROMPT + "\n\n" + doc_context}]

    # Add last 10 turns of history
    trimmed = req.history[-10:]
    sanitised: list[dict] = []
    for entry in trimmed:
        role = entry.role if entry.role in ("user", "assistant") else "user"
        if sanitised and sanitised[-1]["role"] == role:
            sanitised[-1]["content"] += "\n" + entry.content
        else:
            sanitised.append({"role": role, "content": entry.content})
    messages.extend(sanitised)
    messages.append({"role": "user", "content": refined_message})

    try:
        response = client.chat.completions.create(
            model=MODEL_FAST,
            messages=messages,
            max_tokens=1500,
            temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Groq error /pdf-chat: {e}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

    reply = response.choices[0].message.content.strip()
    logger.info(f"/pdf-chat  reply_len={len(reply)}")
    return PdfChatResponse(reply=reply, pdf_name=pdf_name)
