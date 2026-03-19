from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq
import json
import os
import logging
import base64
import re
import httpx
import io

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("sedy")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sedy API", version="3.5.0")
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

# ── Available models ───────────────────────────────────────────────────────────
MODELS = {
    "pro":   "llama-3.3-70b-versatile",
    "flash": "llama-3.1-8b-instant",
    "smart": "qwen/qwen3-32b",
    "code":  "deepseek-r1-distill-qwen-32b",
}

NO_SYSTEM_ROLE_MODELS: set[str] = set()
DEFAULT_MODEL = MODELS["pro"]

AUTO_MODEL_MAP = {
    "chat":       MODELS["pro"],
    "pdf":        MODELS["pro"],
    "code":       MODELS["code"],
    "notes":      MODELS["pro"],
    "formula":    MODELS["smart"],
    "flashcard":  MODELS["smart"],
    "quiz":       MODELS["smart"],
    "graph":      MODELS["smart"],
    "flowchart":  MODELS["smart"],
    "intent":     MODELS["flash"],
    "refine":     MODELS["flash"],
}

def resolve_model(requested: str | None, task: str = "chat") -> str:
    if not requested or requested == "auto":
        return AUTO_MODEL_MAP.get(task, DEFAULT_MODEL)
    return MODELS.get(requested, DEFAULT_MODEL)

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

# ── PDF library ────────────────────────────────────────────────────────────────
PDF_ENGINE = None
try:
    from pypdf import PdfReader as _PyPdfReader
    PDF_ENGINE = "pypdf"
    logger.info("PDF engine: pypdf")
except ImportError:
    try:
        import pdfplumber as _pdfplumber
        PDF_ENGINE = "pdfplumber"
        logger.info("PDF engine: pdfplumber")
    except ImportError:
        try:
            import fitz
            PDF_ENGINE = "pymupdf"
            logger.info("PDF engine: pymupdf")
        except ImportError:
            logger.warning("No PDF library found! Install pypdf, pdfplumber, or PyMuPDF.")


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, int]:
    if PDF_ENGINE == "pypdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages  = reader.pages
        text   = "\n".join(
            f"\n--- Page {i+1} ---\n{p.extract_text() or ''}"
            for i, p in enumerate(pages)
        )
        return text, len(pages)

    elif PDF_ENGINE == "pdfplumber":
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = pdf.pages
            text  = "\n".join(
                f"\n--- Page {i+1} ---\n{p.extract_text() or ''}"
                for i, p in enumerate(pages)
            )
            return text, len(pages)

    elif PDF_ENGINE == "pymupdf":
        import fitz
        doc   = fitz.open(stream=pdf_bytes, filetype="pdf")
        text  = "\n".join(
            f"\n--- Page {i+1} ---\n{doc[i].get_text()}"
            for i in range(len(doc))
        )
        return text, len(doc)

    else:
        raise RuntimeError(
            "No PDF library is installed on the server. "
            "Please install 'pypdf' (pip install pypdf)."
        )


# ══════════════════════════════════════════════════════════════════════════════
# ── SYSTEM PROMPTS
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

REFINE_SYSTEM_PROMPT = """You are a silent prompt refinement engine for a student AI assistant.
Your job is to rewrite the student's message into a complete, self-contained, unambiguous request
by using the conversation history as context.

RULES:
1. Fix ALL spelling/typo errors (e.g. "grph" → "graph", "imnports" → "imports")
2. Fix grammar while keeping casual tone
3. CRITICAL — resolve ALL vague or incomplete references using history:
   - "in inr" after a GDP graph → "Show India GDP from 2000 to 2024 as a line graph in INR (Indian Rupees)"
   - "grph" after a table of data → "Show that data as a proper line graph"
   - "write answer only" after a math problem → "Give only the final answer to x² + 5x + 6 = 0"
   - "without any imports" after code → "Rewrite the snake game in Python without using any imports"
   - "explain more" → "Explain [the last topic discussed] in more detail"
   - "make it harder" after a quiz → "Make the quiz on [last topic] harder"
   - "same but for china" → "Show the same [graph/data] for China"
4. If the message is very short or a single word, ALWAYS expand it using context
5. If the message references "it", "this", "that", "same", "above" — replace with the actual subject
6. Preserve the student's intent — don't change what they're asking for, just make it complete

Output ONLY the refined message. No explanation, no preamble, no quotes around it."""

GRAPH_REFINE_SYSTEM_PROMPT = """You are a silent prompt refinement engine for data visualisation requests.
Your job is to rewrite the user's message into a complete, self-contained graph request using history.

RULES:
1. Fix ALL spelling/typo errors (e.g. "grph" → "graph", "gdp of india" is fine)
2. Resolve ALL vague references using conversation history:
   - "in inr" → expand to full request: "Show India GDP 2000-2024 line graph in INR"
   - "grph" or "graph" alone → "Show [last discussed data topic] as a line graph"
   - "same for china" → "Show [same metric/years] for China as a line graph"
   - "in dollars" → restate full request with currency changed
   - "make it a bar chart" → restate full request with chart type changed
3. Append chart type if implied:
   - "breakdown/share/proportion/percentage" → append "(pie chart)"
   - "compare/vs/ranking/top N/countries" → append "(bar chart)"
   - "over time/trend/history/years/growth" → append "(line chart)"
4. Always output a COMPLETE graph request with: what to show, time range if any, unit if any, chart type

Output ONLY the refined message. No explanation, no preamble."""

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

FLOWCHART_SYSTEM_PROMPT = """You are a flowchart layout engine for a student learning app. Output ONLY a single valid JSON object. No prose, no markdown fences, no explanation.

SCHEMA:
{
  "title": "Short title (max 50 chars)",
  "nodes": [
    {
      "id": "unique_snake_case_id",
      "label": "Short label (max 30 chars, use \\n for line break if needed)",
      "sub": "Optional subtitle max 25 chars",
      "shape": "oval | rect | diamond | para",
      "col": "gray | blue | teal | amber | green | coral | purple",
      "x": 280,
      "y": 40
    }
  ],
  "edges": [
    {
      "f": "from_node_id",
      "t": "to_node_id",
      "label": "Yes / No / short label (optional, max 10 chars)",
      "back": false,
      "bx": 500
    }
  ]
}

SHAPE RULES:
- oval   → start and end nodes ONLY
- rect   → regular process steps
- diamond → decision / condition nodes (always have Yes/No branches)
- para   → input / output operations (user input, data output)

COLOR RULES:
- gray   → start and end ovals
- blue   → input / data / lookup steps
- teal   → processing / transformation steps
- amber  → decisions (ALL diamond nodes must be amber)
- green  → success / positive outcome
- coral  → error / failure / rejection path
- purple → complex internal logic / sub-process

LAYOUT RULES (CRITICAL — the frontend renders nodes at exact x,y pixel positions):
- Canvas is 680px wide. Keep all node x values between 20 and 520 (node width = 140px, so right edge = x+140).
- Main flow goes top-to-bottom with ~100px vertical spacing between node centers.
- Branch left for No/error paths: use x around 60-100
- Branch right for Yes/alternate paths or parallel flows: use x around 420-480
- Center (main flow) x should be around 250-280
- Decision nodes branch: one child directly below (same x), one child to the left or right
- back edges (loop-backs): set "back": true and provide "bx" — the x coordinate of the waypoint for the return arrow. Use bx < 20 for left-side loops, bx > 540 for right-side loops.
- Nodes must NOT overlap: minimum 80px vertical gap between nodes at the same x column; minimum 160px horizontal gap between nodes at the same y row.
- Start with y=40 for the first node. Increment y by 90-110 for each level.
- For branching: branch children start at the same y as each other, positioned left/right of center.

LABEL RULES:
- Labels must be plain text — no markdown, no LaTeX, no special symbols except standard math (², ³, ÷, ≤, ≥, ≠, →)
- Use \\n to break long labels into 2 lines
- Keep labels SHORT — the box is only 140px wide
- edge labels: only "Yes", "No", or a very short phrase (max 10 chars)

OUTPUT ONLY THE JSON. Nothing else."""

PDF_SYSTEM_PROMPT = """You are Sedy, an intelligent student learning assistant made by Ansh Verma.
The user has uploaded a PDF whose FULL TEXT is embedded below between === DOCUMENT START === and === DOCUMENT END ===.

YOUR RULES:
1. Base ALL answers strictly on the document text provided. Do NOT say "I don't have access" — the text IS right there.
2. If asked to summarise: write ## headings for each major section, bullet-point the key ideas under each heading.
3. If asked a question: find the relevant passage and explain it clearly. Quote the page number when useful.
4. If the answer genuinely isn't in the document, say "This topic isn't covered in this PDF."
5. Use markdown formatting (##, **, bullets). Use LaTeX for math: $...$ inline, $$...$$ display.
6. Be thorough — do NOT give short or vague answers. Students need real detail."""

INTENT_SYSTEM_PROMPT = """You are an intent classifier for a student learning app.
Output EXACTLY one word from: graph, flashcard, quiz, both, notes, formula, flowchart, chat

graph     = user EXPLICITLY asks for a chart, graph, plot, bar chart, pie chart, line graph, or data visualisation.
flashcard = wants flip study cards
quiz      = wants MCQ quiz
both      = wants flashcards AND a quiz
notes     = wants structured study notes, revision notes, a summary in note form, key points as notes
formula   = wants a formula sheet, key terms, definitions list, cheat sheet, or terminology reference
flowchart = user asks for a flowchart, flow diagram, process diagram, diagram of a process, "diagram of X",
            "draw the steps of X", "flowchart for X", "how does X work as a diagram", "show me the process of X",
            "map out X", "chart the steps", "draw X process". Trigger even without the word "flowchart" if the
            user clearly wants a visual step-by-step process diagram.
chat      = everything else — explanations, questions, math problems, coding, "write answer", "explain", "solve",
            "what is", "how does", general conversation

CRITICAL:
- If the user asks a math question, wants an explanation, says "write answer", "solve this" — output: chat
- Only output "graph" if the user literally asks for a chart or visualisation of data
- Only output "flowchart" if the user wants a visual process/flow diagram
- Infer "notes" and "formula" from context and intent, not just keywords
- When unsure between notes/chat, prefer notes if it seems like a study/revision request

No punctuation, no explanation. One word only."""

NOTES_SYSTEM_PROMPT = """You are Sedy, an expert study notes writer for students made by Ansh Verma.
Generate comprehensive, well-structured study notes on the given topic.

FORMAT RULES:
1. Start with a one-line topic title as ## heading
2. Use ## for major sections, ### for sub-sections
3. Use bullet points (- ) for key facts under each section
4. Bold (**term**) all key terms when first introduced
5. Use LaTeX for ALL math: $...$ inline, $$...$$ display
6. End with a ## Key Takeaways section with 4-6 bullet points
7. Be comprehensive — cover ALL important aspects of the topic
8. Write for a student who needs to revise efficiently — clear, concise, complete

Do NOT add preamble like "Here are your notes". Start directly with the ## heading."""

FORMULA_SYSTEM_PROMPT = """You are Sedy, an expert at creating formula and key terms reference sheets for students made by Ansh Verma.
Generate a complete reference sheet for the given topic.

FORMAT RULES:
1. Start with ## [Topic] — Formula & Key Terms Sheet
2. Split into two sections:
   ### 📐 Formulas & Equations
   - Each formula on its own line: **Name**: $formula$ — brief explanation
   - Use LaTeX for ALL math: $...$ inline, $$...$$ display

   ### 📖 Key Terms & Definitions
   - Each term: **Term** — clear, concise definition (1-2 sentences)

3. Cover EVERY important formula and term — do not skip any
4. Order from basic to advanced within each section
5. If it is a non-math topic (history, biology etc), focus on key terms, dates, names, and concepts

Do NOT add preamble. Start directly with the ## heading."""

CODE_QUESTIONS_PROMPT = """You are helping clarify a coding request before writing code.
The user has asked: "{request}"

Generate 2-3 SHORT, RELEVANT multiple-choice questions to clarify missing details.

STRICT RULES:
- NEVER ask about something the user already mentioned
- NEVER ask generic or obvious questions
- Ask only about things that will genuinely change how the code is written
- Each question must have 3-4 short options (1-5 words each)
- If the request is already very detailed and nothing is missing, return an empty array []
- Return ONLY a valid JSON array, no markdown fences, no explanation

FORMAT:
[
  {{"question": "Question text?", "options": ["Option A", "Option B", "Option C"]}},
  {{"question": "Another question?", "options": ["Option A", "Option B", "Option C", "Option D"]}}
]"""


# ══════════════════════════════════════════════════════════════════════════════
# ── PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════

class HistoryEntry(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[HistoryEntry] = []
    model: str = "auto"

class FlashcardRequest(BaseModel):
    topic: str
    count: int = 6
    history: list[HistoryEntry] = []
    model: str = "auto"

class QuizRequest(BaseModel):
    topic: str
    difficulty: str = "medium"
    count: int = 5
    history: list[HistoryEntry] = []
    model: str = "auto"

class GraphRequest(BaseModel):
    message: str
    chart_type: str = "auto"
    history: list[HistoryEntry] = []
    model: str = "auto"

class PdfChatRequest(BaseModel):
    message: str
    pdf_base64: str
    pdf_name: str = "document.pdf"
    history: list[HistoryEntry] = []
    model: str = "auto"

class NotesRequest(BaseModel):
    topic: str
    history: list[HistoryEntry] = []
    model: str = "auto"
    pdf_base64: str = ""
    pdf_name: str = ""

class FormulaRequest(BaseModel):
    topic: str
    history: list[HistoryEntry] = []
    model: str = "auto"
    pdf_base64: str = ""
    pdf_name: str = ""

class IntentRequest(BaseModel):
    message: str
    history: list[HistoryEntry] = []

class CodeQuestionsRequest(BaseModel):
    message: str

# ── Flowchart models ───────────────────────────────────────────────────────────
class FlowchartRequest(BaseModel):
    message: str
    history: list[HistoryEntry] = []
    model: str = "auto"

class FlowchartNode(BaseModel):
    id: str
    label: str
    sub: str = ""
    shape: str = "rect"   # oval | rect | diamond | para
    col: str = "blue"     # gray | blue | teal | amber | green | coral | purple
    x: float = 0
    y: float = 0

class FlowchartEdge(BaseModel):
    f: str          # from node id
    t: str          # to node id
    label: str = ""
    back: bool = False
    bx: float = 0   # waypoint x for back-edges

class FlowchartResponse(BaseModel):
    title: str
    nodes: list[FlowchartNode]
    edges: list[FlowchartEdge]

# ── Response models ────────────────────────────────────────────────────────────
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

class NotesResponse(BaseModel):
    notes: str
    topic: str

class FormulaResponse(BaseModel):
    sheet: str
    topic: str

class IntentResponse(BaseModel):
    intent: str

class CodeQuestion(BaseModel):
    question: str
    options: list[str]

class CodeQuestionsResponse(BaseModel):
    questions: list[CodeQuestion]


# ══════════════════════════════════════════════════════════════════════════════
# ── HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_rate_limit_error(exc: Exception) -> dict | None:
    import re as _re
    msg = str(exc)
    if "429" not in msg and "rate_limit" not in msg.lower():
        return None

    wait_sec = 0
    wait_str = ""
    m = _re.search(r'try again in\s+([\dhms. ]+)', msg, _re.I)
    if m:
        raw = m.group(1).strip()
        wait_str = raw
        hours   = sum(float(x) for x in _re.findall(r'([\d.]+)h', raw))
        minutes = sum(float(x) for x in _re.findall(r'([\d.]+)m', raw))
        seconds = sum(float(x) for x in _re.findall(r'([\d.]+)s', raw))
        wait_sec = int(hours * 3600 + minutes * 60 + seconds)

    limit_type = "tokens"
    if "tokens per minute" in msg.lower() or "TPM" in msg:
        limit_type = "TPM"
    elif "tokens per day" in msg.lower() or "TPD" in msg:
        limit_type = "TPD"
    elif "requests per minute" in msg.lower() or "RPM" in msg:
        limit_type = "RPM"

    used  = 0
    limit = 0
    mu = _re.search(r'Used\s+([\d,]+)', msg)
    ml = _re.search(r'Limit\s+([\d,]+)', msg)
    if mu: used  = int(mu.group(1).replace(',',''))
    if ml: limit = int(ml.group(1).replace(',',''))

    return {
        "type":         "rate_limit",
        "wait_seconds": wait_sec,
        "wait_display": wait_str,
        "limit_type":   limit_type,
        "used":         used,
        "limit":        limit,
    }


def build_messages(system: str, history: list[HistoryEntry], user_message: str) -> list[dict]:
    messages = [{"role": "system", "content": system}]
    trimmed  = history[-20:] if len(history) > 20 else history
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


def strip_think_tags(raw: str) -> str:
    return re.sub(r'<think>[\s\S]*?</think>', '', raw).strip()


def extract_json_array(raw: str) -> list:
    raw = strip_think_tags(raw).strip()
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
    raw = strip_think_tags(raw).strip()
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

    msg = message.strip()
    if len(msg) > 200 and ' ' in msg and not any(
        vague in msg.lower() for vague in ['in inr', 'in usd', 'same for', 'make it', 'show it', 'as graph', 'grph']
    ):
        return message

    history_snippet = ""
    if history:
        lines = [
            f"{'Student' if e.role=='user' else 'Tutor'}: {e.content[:400]}"
            for e in history[-20:]
        ]
        history_snippet = "\n".join(lines)

    user_content = (
        f"=== Conversation History ===\n{history_snippet}\n\n=== Message to refine ===\n{msg}"
        if history_snippet else f"Message to refine: {msg}"
    )
    try:
        response = client.chat.completions.create(
            model=MODELS["flash"],
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
            max_tokens=512, temperature=0.2,
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
# ── ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "status": "Sedy API is live 🚀",
        "version": "3.5.0",
        "pdf_engine": PDF_ENGINE or "none — install pypdf!",
        "live_data": bool(SERPER_API_KEY),
        "endpoints": [
            "/chat", "/flashcards", "/quiz", "/graph", "/pdf-chat",
            "/intent", "/code-questions", "/notes", "/formula-sheet", "/flowchart"
        ],
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
            model=MODELS["flash"],
            messages=[{"role": "system", "content": INTENT_SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            max_tokens=5, temperature=0.0,
        )
        raw    = response.choices[0].message.content.strip().lower()
        valid  = ("graph", "flashcard", "quiz", "both", "notes", "formula", "flowchart", "chat")
        intent = raw if raw in valid else "chat"
        logger.info(f"/intent  result={intent!r}")
        return IntentResponse(intent=intent)
    except Exception as e:
        logger.warning(f"/intent  failed ({e}), defaulting to chat")
        return IntentResponse(intent="chat")


# ── /chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    model = resolve_model(req.model, "chat")
    logger.info(f"/chat  model={model}  history={len(req.history)}  msg={req.message[:80]!r}")
    refined  = await refine_prompt(req.message, req.history)
    messages = build_messages(SYSTEM_PROMPT, req.history, refined)
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=32768, temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Groq error /chat: {e}")
        rl = parse_rate_limit_error(e)
        if rl:
            raise HTTPException(status_code=429, detail=rl)
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    reply = strip_think_tags(response.choices[0].message.content.strip())
    logger.info(f"/chat  reply_len={len(reply)}")
    return ChatResponse(reply=reply)


# ── /flashcards ────────────────────────────────────────────────────────────────

@app.post("/flashcards", response_model=FlashcardResponse)
async def flashcards(req: FlashcardRequest):
    model = resolve_model(req.model, "flashcard")
    logger.info(f"/flashcards  model={model}  topic={req.topic!r}")
    topic_as_prompt = f"Generate flashcards about {req.topic.strip()}"
    refined = await refine_prompt(topic_as_prompt, req.history)
    m = re.search(r'(?:flashcards?\s+(?:about|on|for)\s+|flashcards?\s+)(.+)', refined, re.I)
    topic = m.group(1).strip() if m else refined.strip()
    topic = resolve_topic_from_history(topic, req.history)
    if not topic:
        raise HTTPException(status_code=400, detail="topic must not be empty")

    auto_count = req.count == 0
    count = max(1, min(req.count, 50)) if not auto_count else 0

    if not auto_count:
        count_instr = f"Generate exactly {count} flashcards"
    else:
        count_instr = (
            "Decide for yourself how many flashcards are needed to fully cover every "
            "important concept in this topic. "
            "Use your judgment: simple/narrow topics need 8-12 cards; "
            "broad, multi-concept, or multi-chapter topics need 15-30 cards. "
            "Generate ALL the cards needed — do NOT stop early."
        )

    user_prompt = (
        f'{count_instr} about "{topic}".\n'
        f'Each card must cover a distinct concept — no duplicates.\n'
        f'Return ONLY a valid JSON array, no markdown, no extra text:\n'
        f'[{{"question":"...","answer":"..."}}]'
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            max_tokens=32768, temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Groq error /flashcards: {e}")
        rl = parse_rate_limit_error(e)
        if rl:
            raise HTTPException(status_code=429, detail=rl)
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    raw = response.choices[0].message.content.strip()
    try:
        data  = extract_json_array(raw)
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
    model = resolve_model(req.model, "quiz")
    logger.info(f"/quiz  model={model}  topic={req.topic!r}")
    topic_as_prompt = f"Quiz me on {req.topic.strip()}"
    refined = await refine_prompt(topic_as_prompt, req.history)
    m = re.search(r'(?:quiz\s+(?:me\s+)?(?:on|about)\s+)(.+)', refined, re.I)
    topic = m.group(1).strip() if m else refined.strip()
    topic = resolve_topic_from_history(topic, req.history)
    if not topic:
        raise HTTPException(status_code=400, detail="topic must not be empty")
    difficulty = req.difficulty if req.difficulty in ("easy", "medium", "hard") else "medium"

    auto_count = req.count == 0
    count = max(1, min(req.count, 50)) if not auto_count else 0

    if not auto_count:
        count_instr = f"Generate exactly {count} {difficulty} MCQ questions"
    else:
        count_instr = (
            f"Decide for yourself how many {difficulty} MCQ questions are needed to "
            f"properly test every important concept in this topic. "
            f"Use your judgment: focused topics need 8-10 questions; "
            f"broad or multi-chapter topics need 12-20 questions. "
            f"Generate ALL necessary questions — do NOT cut short."
        )

    user_prompt = (
        f'{count_instr} about "{topic}".\n'
        f'Each question must test a different concept — no duplicates.\n'
        f'Return ONLY a valid JSON array, no markdown, no extra text:\n'
        f'[{{"question":"...","options":["A","B","C","D"],"answer":0,"explanation":"..."}}]'
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            max_tokens=32768, temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Groq error /quiz: {e}")
        rl = parse_rate_limit_error(e)
        if rl:
            raise HTTPException(status_code=429, detail=rl)
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    raw = response.choices[0].message.content.strip()
    try:
        data      = extract_json_array(raw)
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
    model = resolve_model(req.model, "graph")
    logger.info(f"/graph  model={model}  msg={req.message[:80]!r}")
    refined = await refine_graph_prompt(req.message, req.history)
    live_snippets = ""
    data_source   = "estimated"
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
            model=model,
            messages=[{"role": "system", "content": GRAPH_SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            max_tokens=32768, temperature=0.3,
        )
    except Exception as e:
        logger.error(f"Groq error /graph: {e}")
        rl = parse_rate_limit_error(e)
        if rl:
            raise HTTPException(status_code=429, detail=rl)
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    raw = strip_json_fences(response.choices[0].message.content.strip())
    try:
        obj         = json.loads(raw)
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
    model = resolve_model(req.model, "pdf")
    pdf_name = req.pdf_name.strip() or "document.pdf"
    logger.info(f"/pdf-chat  model={model}  file={pdf_name!r}  msg={req.message[:80]!r}")

    raw_b64 = req.pdf_base64
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]
    raw_b64 = raw_b64.strip().replace("\n", "").replace("\r", "").replace(" ", "")
    try:
        pdf_bytes = base64.b64decode(raw_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 PDF data: {exc}")

    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not appear to be a valid PDF (missing %PDF header).",
        )

    try:
        pdf_text, page_count = extract_pdf_text(pdf_bytes)
        logger.info(f"/pdf-chat  {len(pdf_text)} chars from {page_count} pages  engine={PDF_ENGINE}")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {exc}")

    if not pdf_text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "This PDF appears to be image-only (scanned) and contains no extractable text. "
                "Please try a text-based PDF, or copy-paste the text you'd like help with."
            ),
        )

    MAX_PDF_CHARS = 80_000
    truncation_note = ""
    original_len = len(pdf_text)
    if original_len > MAX_PDF_CHARS:
        pdf_text = pdf_text[:MAX_PDF_CHARS]
        truncation_note = (
            f"\n\n[NOTE: Document was {original_len} chars; showing first {MAX_PDF_CHARS}. "
            f"Later pages may not be shown.]"
        )

    refined_message = await refine_prompt(req.message, req.history)

    doc_block = (
        f'\n\nDOCUMENT: "{pdf_name}" ({page_count} pages)\n'
        f"=== DOCUMENT START ===\n{pdf_text}{truncation_note}\n=== DOCUMENT END ==="
    )
    messages: list[dict] = [{"role": "system", "content": PDF_SYSTEM_PROMPT + doc_block}]

    sanitised: list[dict] = []
    for entry in req.history[-6:]:
        role = entry.role if entry.role in ("user", "assistant") else "user"
        if sanitised and sanitised[-1]["role"] == role:
            sanitised[-1]["content"] += "\n" + entry.content
        else:
            sanitised.append({"role": role, "content": entry.content})
    messages.extend(sanitised)
    messages.append({"role": "user", "content": refined_message})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=32768,
            temperature=0.3,
        )
    except Exception as e:
        logger.error(f"Groq error /pdf-chat: {e}")
        rl = parse_rate_limit_error(e)
        if rl:
            raise HTTPException(status_code=429, detail=rl)
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

    reply = strip_think_tags(response.choices[0].message.content.strip())
    if not reply:
        reply = "I couldn't generate a response. Please try rephrasing your question."
    logger.info(f"/pdf-chat  reply_len={len(reply)}")
    return PdfChatResponse(reply=reply, pdf_name=pdf_name)


# ── /code-questions ────────────────────────────────────────────────────────────

@app.post("/code-questions", response_model=CodeQuestionsResponse)
async def code_questions(req: CodeQuestionsRequest):
    logger.info(f"/code-questions  msg={req.message[:80]!r}")
    prompt = CODE_QUESTIONS_PROMPT.format(request=req.message.strip())
    try:
        response = client.chat.completions.create(
            model=MODELS["smart"],
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": "Generate the clarifying questions now."},
            ],
            max_tokens=400, temperature=0.4,
        )
    except Exception as e:
        logger.error(f"Groq error /code-questions: {e}")
        rl = parse_rate_limit_error(e)
        if rl:
            raise HTTPException(status_code=429, detail=rl)
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")
    raw   = response.choices[0].message.content.strip()
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


# ── /notes ─────────────────────────────────────────────────────────────────────

@app.post("/notes", response_model=NotesResponse)
async def generate_notes(req: NotesRequest):
    model = resolve_model(req.model, "notes")
    logger.info(f"/notes  model={model}  topic={req.topic!r}")

    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic must not be empty")

    if req.pdf_base64:
        try:
            raw_b64   = req.pdf_base64.split(",", 1)[1] if "," in req.pdf_base64 else req.pdf_base64
            pdf_bytes = base64.b64decode(raw_b64.strip())
            pdf_text, page_count = extract_pdf_text(pdf_bytes)
            if len(pdf_text) > 60_000:
                pdf_text = pdf_text[:60_000] + "\n\n[truncated]"
            user_prompt = (
                f'Generate comprehensive study notes on "{topic}" '
                f'based on this document ({req.pdf_name}, {page_count} pages):\n\n{pdf_text}'
            )
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not read PDF: {e}")
    else:
        user_prompt = f'Generate comprehensive study notes on: "{topic}"'

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": NOTES_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=32768,
            temperature=0.5,
        )
    except Exception as e:
        logger.error(f"Groq error /notes: {e}")
        rl = parse_rate_limit_error(e)
        if rl:
            raise HTTPException(status_code=429, detail=rl)
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

    notes = strip_think_tags(response.choices[0].message.content.strip())
    logger.info(f"/notes  reply_len={len(notes)}")
    return NotesResponse(notes=notes, topic=topic)


# ── /formula-sheet ─────────────────────────────────────────────────────────────

@app.post("/formula-sheet", response_model=FormulaResponse)
async def formula_sheet(req: FormulaRequest):
    model = resolve_model(req.model, "formula")
    logger.info(f"/formula-sheet  model={model}  topic={req.topic!r}")

    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic must not be empty")

    if req.pdf_base64:
        try:
            raw_b64   = req.pdf_base64.split(",", 1)[1] if "," in req.pdf_base64 else req.pdf_base64
            pdf_bytes = base64.b64decode(raw_b64.strip())
            pdf_text, page_count = extract_pdf_text(pdf_bytes)
            if len(pdf_text) > 60_000:
                pdf_text = pdf_text[:60_000] + "\n\n[truncated]"
            user_prompt = (
                f'Generate a complete formula and key terms sheet for "{topic}" '
                f'based on this document ({req.pdf_name}, {page_count} pages):\n\n{pdf_text}'
            )
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not read PDF: {e}")
    else:
        user_prompt = f'Generate a complete formula and key terms reference sheet for: "{topic}"'

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FORMULA_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=32768,
            temperature=0.3,
        )
    except Exception as e:
        logger.error(f"Groq error /formula-sheet: {e}")
        rl = parse_rate_limit_error(e)
        if rl:
            raise HTTPException(status_code=429, detail=rl)
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

    sheet = strip_think_tags(response.choices[0].message.content.strip())
    logger.info(f"/formula-sheet  reply_len={len(sheet)}")
    return FormulaResponse(sheet=sheet, topic=topic)


# ── /flowchart ─────────────────────────────────────────────────────────────────

@app.post("/flowchart", response_model=FlowchartResponse)
async def flowchart(req: FlowchartRequest):
    model = resolve_model(req.model, "flowchart")
    logger.info(f"/flowchart  model={model}  msg={req.message[:80]!r}")

    # Refine the message using conversation history
    refined = await refine_prompt(req.message, req.history)

    user_prompt = f"Create a flowchart for: {refined}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FLOWCHART_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=4096,
            temperature=0.2,
        )
    except Exception as e:
        logger.error(f"Groq error /flowchart: {e}")
        rl = parse_rate_limit_error(e)
        if rl:
            raise HTTPException(status_code=429, detail=rl)
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

    raw = strip_json_fences(response.choices[0].message.content.strip())

    try:
        obj = json.loads(raw)

        nodes = []
        for n in obj.get("nodes", []):
            if not isinstance(n, dict):
                continue
            # Validate and clamp x/y so nodes always stay on canvas
            x = max(20, min(float(n.get("x", 280)), 520))
            y = max(20, float(n.get("y", 40)))
            nodes.append(FlowchartNode(
                id    = str(n.get("id", f"node_{len(nodes)}")),
                label = str(n.get("label", "Step"))[:50],
                sub   = str(n.get("sub", ""))[:30],
                shape = n.get("shape", "rect") if n.get("shape") in ("oval","rect","diamond","para") else "rect",
                col   = n.get("col", "blue") if n.get("col") in ("gray","blue","teal","amber","green","coral","purple") else "blue",
                x     = x,
                y     = y,
            ))

        edges = []
        node_ids = {n.id for n in nodes}
        for e in obj.get("edges", []):
            if not isinstance(e, dict):
                continue
            f_id = str(e.get("f", ""))
            t_id = str(e.get("t", ""))
            # Only include edges where both nodes exist
            if f_id not in node_ids or t_id not in node_ids:
                logger.warning(f"/flowchart  skipping edge {f_id}→{t_id} — node not found")
                continue
            edges.append(FlowchartEdge(
                f     = f_id,
                t     = t_id,
                label = str(e.get("label", ""))[:15],
                back  = bool(e.get("back", False)),
                bx    = float(e.get("bx", 0)),
            ))

        if not nodes:
            raise ValueError("No valid nodes in flowchart response")

        logger.info(f"/flowchart  title={obj.get('title','')!r}  nodes={len(nodes)}  edges={len(edges)}")
        return FlowchartResponse(
            title = str(obj.get("title", refined[:50])),
            nodes = nodes,
            edges = edges,
        )

    except Exception as e:
        logger.warning(f"/flowchart  parse failed: {e}  raw={raw[:400]!r}")
        raise HTTPException(status_code=422, detail=f"Could not parse flowchart data: {e}")
