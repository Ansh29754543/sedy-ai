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
app = FastAPI(title="Sedy API", version="3.7.0")
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
    "pro":    "llama-3.3-70b-versatile",
    "flash":  "llama-3.1-8b-instant",
    "smart":  "qwen/qwen3-32b",
    "code":   "deepseek-r1-distill-qwen-32b",
    "vision": "meta-llama/llama-4-scout-17b-16e-instruct",  # ← NEW: vision model
}

DEFAULT_MODEL = MODELS["pro"]

AUTO_MODEL_MAP = {
    "chat":      MODELS["pro"],
    "pdf":       MODELS["pro"],
    "code":      MODELS["code"],
    "notes":     MODELS["pro"],
    "formula":   MODELS["smart"],
    "flashcard": MODELS["smart"],
    "quiz":      MODELS["smart"],
    "graph":     MODELS["smart"],
    "flowchart": MODELS["smart"],
    "intent":    MODELS["flash"],
    "refine":    MODELS["flash"],
    "image":     MODELS["vision"],   # ← NEW: always use vision model for images
}

def resolve_model(requested: str | None, task: str = "chat") -> str:
    if task == "image":
        return MODELS["vision"]   # vision model is fixed for image tasks
    if not requested or requested == "auto":
        return AUTO_MODEL_MAP.get(task, DEFAULT_MODEL)
    return MODELS.get(requested, DEFAULT_MODEL)

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

# ── PDF Engine Detection ───────────────────────────────────────────────────────
PDF_ENGINE = None
try:
    import fitz
    PDF_ENGINE = "pymupdf"
    logger.info("PDF engine: pymupdf (with OCR support)")
except ImportError:
    try:
        from pypdf import PdfReader as _PyPdfReader
        PDF_ENGINE = "pypdf"
        logger.info("PDF engine: pypdf (no OCR — scanned PDFs will fail)")
    except ImportError:
        try:
            import pdfplumber as _pdfplumber
            PDF_ENGINE = "pdfplumber"
            logger.info("PDF engine: pdfplumber (no OCR — scanned PDFs will fail)")
        except ImportError:
            logger.warning("No PDF library found! Install pymupdf: pip install pymupdf")


def _ocr_page_fitz(page) -> str:
    try:
        tp = page.get_textpage_ocr(flags=0, language="eng")
        return page.get_text(textpage=tp)
    except Exception as e:
        logger.warning(f"OCR failed on page: {e}")
        return ""


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, int]:
    if PDF_ENGINE == "pymupdf":
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        extracted = []
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text().strip()
            if len(text) < 20:
                logger.info(f"Page {i+1} appears image-only — attempting OCR")
                text = _ocr_page_fitz(page)
            extracted.append(f"\n--- Page {i+1} ---\n{text}")
        full_text = "\n".join(extracted)
        return full_text, len(doc)
    elif PDF_ENGINE == "pypdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = reader.pages
        extracted = []
        for i, p in enumerate(pages):
            text = p.extract_text() or ""
            extracted.append(f"\n--- Page {i+1} ---\n{text}")
        full_text = "\n".join(extracted)
        meaningful = len(re.sub(r"[\s\-]", "", full_text))
        if meaningful < 100:
            logger.warning("pypdf extracted almost no text — PDF is likely scanned.")
        return full_text, len(pages)
    elif PDF_ENGINE == "pdfplumber":
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = pdf.pages
            text = "\n".join(
                f"\n--- Page {i+1} ---\n{p.extract_text() or ''}"
                for i, p in enumerate(pages)
            )
            return text, len(pages)
    else:
        raise RuntimeError(
            "No PDF library is installed on the server. "
            "Please install pymupdf: pip install pymupdf"
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

# ── Vision / Image system prompt ──────────────────────────────────────────────
IMAGE_SYSTEM_PROMPT = """You are Sedy, an intelligent AI study assistant with vision capabilities, made by Ansh Verma.
The student has shared an image. Look at it VERY carefully and help them understand it completely.

YOUR CAPABILITIES:
- Read printed text, handwritten text, equations from images
- Solve math problems shown in images (handwritten or printed)
- Explain diagrams: biology, chemistry, physics, geography, circuits, flowcharts
- Describe what is in the image clearly
- Answer questions about the image content
- Read question papers, textbook pages, notes, worksheets
- Identify graphs and explain data trends
- IDENTIFY LOGOS, EMBLEMS, SEALS, BADGES, SYMBOLS — read every text and visual clue in them

LOGO & EMBLEM IDENTIFICATION — VERY IMPORTANT:
You are an expert at identifying Indian school, college, government, and institutional logos. When you see a logo or emblem:
1. READ all text in the image, including Hindi/Sanskrit/regional language text at top, bottom, and sides
2. LOOK at the central image/symbol carefully — what animal, object, or scene is depicted?
3. LOOK at the shape: circular seal, shield, badge, etc.
4. MATCH against your knowledge of Indian institutions

COMMON INDIAN EDUCATIONAL LOGOS you must recognize:
- **KVS / Kendriya Vidyalaya Sangathan** — circular seal, Sanskrit motto "तत् त्वं पूषन् अपावृणु" (Tat Tvam Pushan Apavrinu), rising sun with rays, tricolor (orange, white, green) stripes at bottom, "केन्द्रीय विद्यालय संगठन" in Hindi, "KENDRIYA VIDYALAYA SANGATHAN" in English
- **CBSE** — Central Board of Secondary Education, blue logo, open book
- **NCERT** — National Council of Educational Research and Training, circular seal
- **NVS / Navodaya Vidyalaya Samiti** — "नवोदय विद्यालय समिति", rising sun
- **Ministry of Education, Govt of India** — Ashoka Pillar with lions, "शिक्षा मंत्रालय"
- **IIT logos** — each IIT has unique logos with their city name
- **University of Delhi** — circular seal with "विद्यया अमृतमश्नुते"
- **UPSC** — Union Public Service Commission
- **NEET / JEE logos** — NTA (National Testing Agency)
- **State board logos** — BSEH, BSEB, Maharashtra Board, UP Board, etc.
- **Army School / AWES** — Army Welfare Education Society
- **Sainik School** — "सैनिक स्कूल", Army-related symbols
- **DAV Schools** — "दयानन्द आर्य वैदिक", Om symbol
- **DPS / Delhi Public School** — circular logo with lamp

IDENTIFICATION STRATEGY:
1. First read ALL visible text in the image — this is your #1 clue
2. If you see "KVS" or "केन्द्रीय विद्यालय" or "Kendriya Vidyalaya" anywhere → it is KVS
3. If you see the rising sun + tricolor stripes + Sanskrit motto → very likely KVS
4. State the organization name confidently if you can identify it
5. Explain what the logo represents: colors, symbols, motto meaning
6. If you truly cannot identify it with certainty, describe what you see in detail and give your best guess

MULTILINGUAL SUPPORT — VERY IMPORTANT:
- If the student writes in Hindi, respond FULLY in Hindi (Devanagari script)
- If the student writes in Bengali (বাংলা), respond fully in Bengali
- If the student writes in Tamil (தமிழ்), respond fully in Tamil
- If the student writes in Telugu (తెలుగు), respond fully in Telugu
- If the student writes in Kannada (ಕನ್ನಡ), respond fully in Kannada
- If the student writes in Malayalam (മലയാളം), respond fully in Malayalam
- If the student writes in Marathi (मराठी), respond fully in Marathi
- If the student writes in Gujarati (ગુજરાતી), respond fully in Gujarati
- If the student writes in Punjabi (ਪੰਜਾਬੀ), respond fully in Punjabi
- If the student writes in Urdu, respond fully in Urdu
- If the student writes in Odia (ଓଡ଼ିଆ), respond fully in Odia
- If the student writes in Assamese (অসমীয়া), respond fully in Assamese
- If the student mixes Hindi + English (Hinglish), respond in the same friendly Hinglish style
- DEFAULT: Respond in the same language the student used in their question
- If no question is asked (student just sends the image), describe it fully and identify it in English, then ask what they need help with

FORMAT RULES:
- Use markdown formatting: ## headers, **bold**, bullet points
- For math: use LaTeX $...$ inline and $$...$$ display
- Be thorough and specific — students need real, accurate information
- Be encouraging and friendly"""

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

4. PDF & DOCUMENT CONTEXT (very important):
   - If the history contains a PDF explanation, summary, or Q&A, treat that document as the active topic.
   - "make notes on it" after a PDF explanation → "Make study notes on [PDF topic]"
   - "make a flowchart" after a PDF explanation → "Make a flowchart for [PDF topic]"
   - "make flashcards" after a PDF → "Make flashcards on [PDF topic]"
   - "quiz me" after a PDF → "Quiz me on [PDF topic]"
   - Always extract the actual subject/topic name from the PDF content in history

5. TOPIC MEMORY — always carry the last known topic forward:
   - "explain more" → "Explain [last topic] in more detail"
   - "give an example" → "Give an example of [last concept discussed]"
   - "make it simpler" → "Explain [last topic] in simpler terms"
   - Short follow-ups like "and?" / "more?" / "continue" → "Continue explaining [last topic]"

6. If the message is very short or a single word, ALWAYS expand it using context
7. If the message references "it", "this", "that", "same", "above" — ALWAYS replace with the actual subject from history
8. Preserve the student's intent — don't change what they're asking for, just make it complete and specific

Output ONLY the refined message. No explanation, no preamble, no quotes around it."""

GRAPH_REFINE_SYSTEM_PROMPT = """You are a silent prompt refinement engine for data visualisation requests.
Your job is to rewrite the user's message into a complete, self-contained graph request using history.

RULES:
1. Fix ALL spelling/typo errors
2. Resolve ALL vague references using conversation history
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
- para   → input / output operations

COLOR RULES:
- gray   → start and end ovals
- blue   → input / data / lookup steps
- teal   → processing / transformation steps
- amber  → decisions (ALL diamond nodes must be amber)
- green  → success / positive outcome
- coral  → error / failure / rejection path
- purple → complex internal logic / sub-process

LAYOUT RULES:
- Canvas is 680px wide. Keep all node x values between 20 and 520.
- Main flow goes top-to-bottom with ~100px vertical spacing.
- Branch left for No/error paths: x around 60-100
- Branch right for Yes/alternate paths: x around 420-480
- Center (main flow) x should be around 250-280
- back edges (loop-backs): set "back": true and provide "bx"
- Nodes must NOT overlap: minimum 80px vertical gap at same x
- Start with y=40 for the first node

OUTPUT ONLY THE JSON. Nothing else."""

PDF_SYSTEM_PROMPT = """You are Sedy, an intelligent student learning assistant made by Ansh Verma.
The user has uploaded a PDF whose FULL TEXT is embedded below between === DOCUMENT START === and === DOCUMENT END ===.

YOUR RULES:
1. Base ALL answers strictly on the document text provided.
2. If asked to summarise: write ## headings for each major section, bullet-point the key ideas.
3. If asked a question: find the relevant passage and explain it clearly.
4. If the answer genuinely isn't in the document, say "This topic isn't covered in this PDF."
5. Use markdown formatting. Use LaTeX for math: $...$ inline, $$...$$ display.
6. Be thorough — do NOT give short or vague answers."""

INTENT_SYSTEM_PROMPT = """You are an intent classifier for a student learning app.
Output EXACTLY one word from: graph, flashcard, quiz, both, notes, formula, flowchart, chat

graph     = user EXPLICITLY asks for a chart, graph, plot, bar chart, pie chart, line graph, or data visualisation.
flashcard = wants flip study cards
quiz      = wants MCQ quiz
both      = wants flashcards AND a quiz
notes     = wants structured study notes, revision notes, a summary in note form, key points as notes
formula   = wants a formula sheet, key terms, definitions list, cheat sheet, or terminology reference
flowchart = user asks for a flowchart, flow diagram, process diagram, diagram of a process
chat      = everything else

CONTEXT-AWARE FOLLOW-UP RULES:
- "make notes on it" after ANY topic or PDF → notes
- "make a flowchart" after ANY topic or PDF → flowchart
- "make flashcards" after ANY topic or PDF → flashcard
- "quiz me" after ANY topic or PDF → quiz
- "formula sheet" after ANY topic or PDF → formula

CRITICAL:
- Math questions, explanations, "solve this" → chat
- Only "graph" if user literally asks for a chart
- Only "flowchart" if user wants a visual process diagram

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
7. Be comprehensive — cover ALL important aspects

Do NOT add preamble. Start directly with the ## heading."""

FORMULA_SYSTEM_PROMPT = """You are Sedy, an expert at creating formula and key terms reference sheets for students made by Ansh Verma.
Generate a complete reference sheet for the given topic.

FORMAT RULES:
1. Start with ## [Topic] — Formula & Key Terms Sheet
2. Split into two sections:
   ### 📐 Formulas & Equations
   - Each formula: **Name**: $formula$ — brief explanation

   ### 📖 Key Terms & Definitions
   - Each term: **Term** — clear, concise definition

3. Cover EVERY important formula and term
4. Order from basic to advanced

Do NOT add preamble. Start directly with the ## heading."""

CODE_QUESTIONS_PROMPT = """You are helping clarify a coding request before writing code.
The user has asked: "{request}"

Generate 2-3 SHORT, RELEVANT multiple-choice questions to clarify missing details.

STRICT RULES:
- NEVER ask about something the user already mentioned
- Ask only about things that will genuinely change how the code is written
- Each question must have 3-4 short options
- If the request is already very detailed, return an empty array []
- Return ONLY a valid JSON array, no markdown fences

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

# ── NEW: Image chat request ────────────────────────────────────────────────────
class ImageChatRequest(BaseModel):
    message: str = ""           # optional — student's question about the image
    images: list[str]           # list of base64 image strings (with or without data URI prefix)
    image_names: list[str] = [] # optional filenames for display
    history: list[HistoryEntry] = []

# ── Flowchart models ───────────────────────────────────────────────────────────
class FlowchartRequest(BaseModel):
    message: str
    history: list[HistoryEntry] = []
    model: str = "auto"

class FlowchartNode(BaseModel):
    id: str
    label: str
    sub: str = ""
    shape: str = "rect"
    col: str = "blue"
    x: float = 0
    y: float = 0

class FlowchartEdge(BaseModel):
    f: str
    t: str
    label: str = ""
    back: bool = False
    bx: float = 0

class FlowchartResponse(BaseModel):
    title: str
    nodes: list[FlowchartNode]
    edges: list[FlowchartEdge]

# ── Response models ────────────────────────────────────────────────────────────
class ChatResponse(BaseModel):
    reply: str

class ImageChatResponse(BaseModel):
    reply: str
    image_count: int

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
        "type": "rate_limit", "wait_seconds": wait_sec, "wait_display": wait_str,
        "limit_type": limit_type, "used": used, "limit": limit,
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
            f"{'Student' if e.role=='user' else 'Tutor'}: {e.content[:600]}"
            for e in history[-30:]
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


def _decode_pdf_base64(raw_b64: str) -> bytes:
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]
    raw_b64 = raw_b64.strip().replace("\n", "").replace("\r", "").replace(" ", "")
    try:
        return base64.b64decode(raw_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 PDF data: {exc}")


def _validate_and_extract_pdf(pdf_bytes: bytes, pdf_name: str) -> tuple[str, int]:
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not appear to be a valid PDF (missing %PDF header).",
        )
    try:
        pdf_text, page_count = extract_pdf_text(pdf_bytes)
        logger.info(f"Extracted {len(pdf_text)} chars from {page_count} pages ({pdf_name}), engine={PDF_ENGINE}")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {exc}")
    if not pdf_text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "This PDF appears to be image-only (scanned) and no text could be extracted. "
                + ("Install pymupdf for OCR support." if PDF_ENGINE != "pymupdf" else
                   "OCR was attempted but produced no output — the PDF may be corrupted.")
            ),
        )
    return pdf_text, page_count


# ── NEW: Image base64 helper ───────────────────────────────────────────────────
def _prepare_image_url(raw_b64: str) -> str:
    """
    Ensure the image is a proper data URI for the vision API.
    Accepts: raw base64 string, or existing data:image/...;base64,... URI.
    Returns: data URI string ready to pass to Groq vision API.
    """
    raw_b64 = raw_b64.strip().replace("\n", "").replace("\r", "").replace(" ", "")

    # Already a data URI — return as-is
    if raw_b64.startswith("data:image/"):
        return raw_b64

    # Detect image type from magic bytes
    try:
        img_bytes = base64.b64decode(raw_b64[:64])  # just first bytes to detect type
        if img_bytes[:2] == b'\xff\xd8':
            mime = "image/jpeg"
        elif img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            mime = "image/png"
        elif img_bytes[:6] in (b'GIF87a', b'GIF89a'):
            mime = "image/gif"
        elif img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
            mime = "image/webp"
        else:
            mime = "image/jpeg"  # default fallback
    except Exception:
        mime = "image/jpeg"

    return f"data:{mime};base64,{raw_b64}"


# ══════════════════════════════════════════════════════════════════════════════
# ── ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "status": "Sedy API is live 🚀",
        "version": "3.7.0",
        "pdf_engine": PDF_ENGINE or "none — install pymupdf!",
        "ocr_support": PDF_ENGINE == "pymupdf",
        "vision_model": MODELS["vision"],
        "live_data": bool(SERPER_API_KEY),
        "endpoints": [
            "/chat", "/flashcards", "/quiz", "/graph", "/pdf-chat",
            "/intent", "/code-questions", "/notes", "/formula-sheet",
            "/flowchart", "/image-chat",
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


# ── NEW: /image-chat ───────────────────────────────────────────────────────────

@app.post("/image-chat", response_model=ImageChatResponse)
async def image_chat(req: ImageChatRequest):
    """
    Vision endpoint — accepts 1-5 images as base64 strings + an optional question.
    Uses Llama 4 Scout (meta-llama/llama-4-scout-17b-16e-instruct) on Groq.
    Responds in whatever Indian language the student used in their question.
    """
    if not req.images:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    # Cap at 5 images (model limit)
    images = req.images[:5]
    image_count = len(images)

    logger.info(f"/image-chat  images={image_count}  msg={req.message[:80]!r}")

    # Build the user content block with image(s) + text
    content: list[dict] = []

    for i, raw_b64 in enumerate(images):
        try:
            image_url = _prepare_image_url(raw_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image {i+1}: {e}")
        content.append({
            "type": "image_url",
            "image_url": {"url": image_url},
        })

    # Attach the student's question (or a default prompt if none)
    question = req.message.strip() if req.message.strip() else (
        "Please look at this image carefully and describe what you see. "
        "If it contains any text, equations, diagrams or problems, explain them clearly. "
        "Then ask me what I need help with."
    )
    content.append({"type": "text", "text": question})

    # Build message list — include recent history as text only
    messages: list[dict] = [{"role": "system", "content": IMAGE_SYSTEM_PROMPT}]

    # Add last 6 history turns as plain text (no images in history)
    sanitised: list[dict] = []
    for entry in req.history[-6:]:
        role = entry.role if entry.role in ("user", "assistant") else "user"
        if sanitised and sanitised[-1]["role"] == role:
            sanitised[-1]["content"] += "\n" + entry.content
        else:
            sanitised.append({"role": role, "content": entry.content})
    messages.extend(sanitised)

    # The actual image + question turn
    messages.append({"role": "user", "content": content})

    try:
        response = client.chat.completions.create(
            model=MODELS["vision"],
            messages=messages,
            max_tokens=4096,
            temperature=0.4,
        )
    except Exception as e:
        logger.error(f"Groq error /image-chat: {e}")
        rl = parse_rate_limit_error(e)
        if rl:
            raise HTTPException(status_code=429, detail=rl)
        raise HTTPException(status_code=502, detail=f"Groq vision API error: {e}")

    reply = strip_think_tags(response.choices[0].message.content.strip())
    if not reply:
        reply = "I could see the image but couldn't generate a response. Please try again."

    logger.info(f"/image-chat  reply_len={len(reply)}")
    return ImageChatResponse(reply=reply, image_count=image_count)


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
            "Simple/narrow topics need 8-12 cards; broad topics need 15-30 cards."
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
            f"properly test every important concept in this topic."
        )
    user_prompt = (
        f'{count_instr} about "{topic}".\n'
        f'Each question must test a different concept — no duplicates.\n'
        f'Return ONLY a valid JSON array:\n'
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
    pdf_bytes = _decode_pdf_base64(req.pdf_base64)
    pdf_text, page_count = _validate_and_extract_pdf(pdf_bytes, pdf_name)
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
            model=model, messages=messages, max_tokens=32768, temperature=0.3,
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
        pdf_bytes = _decode_pdf_base64(req.pdf_base64)
        pdf_text, page_count = _validate_and_extract_pdf(pdf_bytes, req.pdf_name or "document.pdf")
        if len(pdf_text) > 60_000:
            pdf_text = pdf_text[:60_000] + "\n\n[truncated]"
        user_prompt = (
            f'Generate comprehensive study notes on "{topic}" '
            f'based on this document ({req.pdf_name}, {page_count} pages):\n\n{pdf_text}'
        )
    else:
        user_prompt = f'Generate comprehensive study notes on: "{topic}"'
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": NOTES_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=32768, temperature=0.5,
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
        pdf_bytes = _decode_pdf_base64(req.pdf_base64)
        pdf_text, page_count = _validate_and_extract_pdf(pdf_bytes, req.pdf_name or "document.pdf")
        if len(pdf_text) > 60_000:
            pdf_text = pdf_text[:60_000] + "\n\n[truncated]"
        user_prompt = (
            f'Generate a complete formula and key terms sheet for "{topic}" '
            f'based on this document ({req.pdf_name}, {page_count} pages):\n\n{pdf_text}'
        )
    else:
        user_prompt = f'Generate a complete formula and key terms reference sheet for: "{topic}"'
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FORMULA_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=32768, temperature=0.3,
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
    refined = await refine_prompt(req.message, req.history)
    user_prompt = f"Create a flowchart for: {refined}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FLOWCHART_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=4096, temperature=0.2,
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
