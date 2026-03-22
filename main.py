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
import io
import asyncio

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("sedy")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sedy API", version="4.1.0")
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

# ── Serper (web search) ────────────────────────────────────────────────────────
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

# ── Available models ───────────────────────────────────────────────────────────
MODELS = {
    "pro":    "llama-3.3-70b-versatile",
    "flash":  "llama-3.1-8b-instant",
    "smart":  "qwen/qwen3-32b",
    "code":   "deepseek-r1-distill-qwen-32b",
    "vision": "meta-llama/llama-4-scout-17b-16e-instruct",
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
    "image":     MODELS["vision"],
    "voice":     MODELS["flash"],   # NEW — fast model for real-time voice
}

def resolve_model(requested: str | None, task: str = "chat") -> str:
    if task == "image":
        return MODELS["vision"]
    if not requested or requested == "auto":
        return AUTO_MODEL_MAP.get(task, DEFAULT_MODEL)
    return MODELS.get(requested, DEFAULT_MODEL)

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
        logger.info("PDF engine: pypdf")
    except ImportError:
        try:
            import pdfplumber as _pdfplumber
            PDF_ENGINE = "pdfplumber"
            logger.info("PDF engine: pdfplumber")
        except ImportError:
            logger.warning("No PDF library found! Install pymupdf: pip install pymupdf")


def _ocr_page_fitz(page) -> str:
    try:
        tp = page.get_textpage_ocr(flags=0, language="eng+hin+ben+tam+tel+kan+mal+mar+guj+pan+urd+ori")
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
        return "\n".join(extracted), len(doc)
    elif PDF_ENGINE == "pypdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages  = reader.pages
        extracted = []
        for i, p in enumerate(pages):
            text = p.extract_text() or ""
            extracted.append(f"\n--- Page {i+1} ---\n{text}")
        full_text = "\n".join(extracted)
        meaningful = len(re.sub(r"[\s\-]", "", full_text))
        if meaningful < 100:
            logger.warning("pypdf extracted almost no text — PDF may be scanned.")
        return full_text, len(pages)
    elif PDF_ENGINE == "pdfplumber":
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(
                f"\n--- Page {i+1} ---\n{p.extract_text() or ''}"
                for i, p in enumerate(pdf.pages)
            )
            return text, len(pdf.pages)
    else:
        raise RuntimeError(
            "No PDF library is installed. Please install pymupdf: pip install pymupdf"
        )


# ══════════════════════════════════════════════════════════════════════════════
# ── LANGUAGE SUPPORT CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SUPPORTED_LANGUAGES = """
SUPPORTED LANGUAGES — detect and respond in the student's language automatically:

Indian Languages:
- हिंदी (Hindi) — respond fully in Hindi using Devanagari script
- বাংলা (Bengali) — respond fully in Bengali using Bengali script
- தமிழ் (Tamil) — respond fully in Tamil using Tamil script
- తెలుగు (Telugu) — respond fully in Telugu using Telugu script
- ಕನ್ನಡ (Kannada) — respond fully in Kannada using Kannada script
- മലയാളം (Malayalam) — respond fully in Malayalam using Malayalam script
- मराठी (Marathi) — respond fully in Marathi using Devanagari script
- ગુજરાતી (Gujarati) — respond fully in Gujarati using Gujarati script
- ਪੰਜਾਬੀ (Punjabi) — respond fully in Punjabi using Gurmukhi script
- اردو (Urdu) — respond fully in Urdu using Nastaliq/Arabic script
- ଓଡ଼ିଆ (Odia) — respond fully in Odia using Odia script
- অসমীয়া (Assamese) — respond fully in Assamese using Bengali script
- संस्कृत (Sanskrit) — respond fully in Sanskrit using Devanagari script
- Hinglish — when student mixes Hindi + English, match that style

English:
- English (default when no other language detected)

LANGUAGE RULES:
1. ALWAYS detect the student's language from their message
2. Respond ENTIRELY in that language — do not switch mid-response
3. Use the correct native script (not romanised transliteration) unless student uses romanised
4. Math notation ($...$) stays as LaTeX regardless of language
5. Technical terms (like "algorithm", "photosynthesis") can stay in English within the response
6. If student code-switches (Hinglish, Tanglish etc.), match their style naturally
"""

# ══════════════════════════════════════════════════════════════════════════════
# ── SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = f"""You are Sedy, an intelligent student learning assistant made by Ansh Verma, a school student.
You explain concepts clearly and simply, solve math and science problems step by step,
summarize topics, and help students understand programming.
Always be encouraging, clear and educational.
Only reveal your identity when asked.

{SUPPORTED_LANGUAGES}

IMPORTANT RULES:
- Use markdown formatting: headers (##), bold (**text**), bullet points (- item), code blocks (```language)
- NEVER describe flashcards or quizzes in plain text — the frontend handles those separately
- Keep responses focused and well structured
- Use conversation history for context when user refers to "it", "that", "same topic"

STRICT MATH FORMATTING (frontend uses KaTeX):
- ALL math MUST be wrapped in LaTeX: $...$ inline, $$...$$ display
- NEVER write bare math like w^2 — always wrap in $w^2$
- Fractions: $\\frac{{a}}{{b}}$  Subscripts: $A_{{\\text{{base}}}}$"""

IMAGE_SYSTEM_PROMPT = f"""You are Sedy, an intelligent AI study assistant with vision capabilities, made by Ansh Verma.
The student has shared an image. Look at it VERY carefully and help them understand it completely.

{SUPPORTED_LANGUAGES}

YOUR CAPABILITIES:
- Read printed text, handwritten text, equations from images
- Read text in ANY Indian script — Devanagari, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Gurmukhi, Odia, Urdu
- Solve math problems shown in images (handwritten or printed)
- Explain diagrams: biology, chemistry, physics, geography, circuits, flowcharts
- Describe what is in the image clearly
- Answer questions about the image content
- Read question papers, textbook pages, notes, worksheets
- Identify graphs and explain data trends
- IDENTIFY LOGOS, EMBLEMS, SEALS, BADGES, SYMBOLS — read every text and visual clue in them
- Read Indian textbook pages (NCERT, state board) in any language

FORMAT RULES:
- Use markdown formatting: ## headers, **bold**, bullet points
- For math: use LaTeX $...$ inline and $$...$$ display
- Be thorough and specific
- Be encouraging and friendly"""

REFINE_SYSTEM_PROMPT = f"""You are a silent prompt refinement engine for a student AI assistant.
Rewrite the student's message into a complete, self-contained, unambiguous request
using the conversation history as context.

{SUPPORTED_LANGUAGES}

RULES:
1. Fix ALL spelling/typo errors — including errors in Indian language text
2. Fix grammar while keeping casual tone and the student's chosen language
3. Resolve ALL vague references using history
4. PDF & DOCUMENT CONTEXT: If history contains a PDF, treat that document as the active topic
5. TOPIC MEMORY — always carry the last known topic forward
6. If the message is very short or single word, ALWAYS expand it using context
7. Replace "it", "this", "that", "same", "above", "यह", "वो", "इसके" with the actual subject from history
8. Preserve the student's intent AND their chosen language

Output ONLY the refined message. No explanation, no preamble, no quotes."""

GRAPH_REFINE_SYSTEM_PROMPT = f"""You are a silent prompt refinement engine for data visualisation requests.
Rewrite the user's message into a complete, self-contained graph request using history.

{SUPPORTED_LANGUAGES}

RULES:
1. Fix ALL spelling/typo errors in any language
2. Resolve ALL vague references using conversation history
3. Append chart type if implied:
   - "breakdown/share/proportion/percentage/हिस्सा/भाग" → append "(pie chart)"
   - "compare/vs/ranking/top N/countries/तुलना" → append "(bar chart)"
   - "over time/trend/history/years/growth/समय के साथ" → append "(line chart)"
4. Always output a COMPLETE graph request with: what to show, time range if any, unit if any, chart type
5. Output the refined request in English (graph data labels will be in English regardless)

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

PDF_SYSTEM_PROMPT = f"""You are Sedy, an intelligent student learning assistant made by Ansh Verma.
The user has uploaded a PDF whose FULL TEXT is embedded below.

{SUPPORTED_LANGUAGES}

YOUR RULES:
1. Base ALL answers strictly on the document text provided.
2. If asked to summarise: write ## headings for each major section, bullet-point the key ideas.
3. If asked a question: find the relevant passage and explain it clearly.
4. If the answer genuinely isn't in the document, say "This topic isn't covered in this PDF."
5. Use markdown formatting. Use LaTeX for math: $...$ inline, $$...$$ display.
6. Be thorough — do NOT give short or vague answers.
7. If the PDF contains text in an Indian language, read and respond in that same language."""

INTENT_SYSTEM_PROMPT = """You are an intent classifier for a student learning app.
Output EXACTLY one word from: graph, flashcard, quiz, both, notes, formula, flowchart, practice, chat

graph     = user EXPLICITLY asks for a chart, graph, plot, bar chart, pie chart, line graph, or data visualisation.
            Also matches: ग्राफ, चार्ट, গ্রাফ, வரைபடம், గ్రాఫ్
flashcard = wants flip study cards
            Also matches: फ्लैशकार्ड, কার্ড, フラッシュカード
quiz      = wants MCQ quiz
            Also matches: क्विज़, কুইজ, வினாடி வினா, క్విజ్
both      = wants flashcards AND a quiz
notes     = wants structured study notes, revision notes, a summary in note form, key points as notes
            Also matches: नोट्स, টীকা, குறிப்புகள், నోట్స్
formula   = wants a formula sheet, key terms, definitions list, cheat sheet, or terminology reference
            Also matches: सूत्र, সূত্র, சூத்திரம், సూత్రాలు
flowchart = user asks for a flowchart, flow diagram, process diagram, diagram of a process
            Also matches: फ्लोचार्ट, প্রবাহচিত্র, ஓட்டப்படம்
practice  = user wants to PRACTICE, SOLVE problems, do EXERCISES, or get QUESTIONS to solve on whiteboard
            Key signals: "practice", "practise", "solve problems", "give me questions", "let me try",
            "I want to practice", "practice questions", "exercise problems", "solve on whiteboard",
            "whiteboard practice", "let me solve", "give problems to solve", "test me on solving"
            Also matches: अभ्यास, प्रैक्टिस, প্র্যাকটিস, பயிற்சி, సాధన, ಅಭ್ಯಾಸ
            IMPORTANT: If user says "practice [topic]" or "practice on [topic]" or "I want to practice [topic]" → practice
chat      = everything else

CRITICAL:
- "Solve this [specific problem]" → chat (they want the answer, not practice)
- "Practice [topic]" or "give me problems to practice" → practice
- Math questions, explanations, "what is X", "explain X" → chat
- Only "graph" if user literally asks for a chart
- Only "flowchart" if user wants a visual process diagram
- Works for ALL languages: English, Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi, Urdu, Odia

No punctuation, no explanation. One word only."""

NOTES_SYSTEM_PROMPT = f"""You are Sedy, an expert study notes writer for students made by Ansh Verma.
Generate comprehensive, well-structured study notes on the given topic.

{SUPPORTED_LANGUAGES}

FORMAT RULES:
1. Start with a one-line topic title as ## heading (in the student's language)
2. Use ## for major sections, ### for sub-sections
3. Use bullet points (- ) for key facts under each section
4. Bold (**term**) all key terms when first introduced
5. Use LaTeX for ALL math: $...$ inline, $$...$$ display
6. End with a ## Key Takeaways section (or equivalent in the student's language) with 4-6 bullet points
7. Be comprehensive — cover ALL important aspects
8. If topic is from Indian curriculum (NCERT, state board), align with that syllabus

Do NOT add preamble. Start directly with the ## heading."""

FORMULA_SYSTEM_PROMPT = f"""You are Sedy, an expert at creating formula and key terms reference sheets for students made by Ansh Verma.
Generate a complete reference sheet for the given topic.

{SUPPORTED_LANGUAGES}

FORMAT RULES:
1. Start with ## [Topic] — Formula & Key Terms Sheet (in the student's language)
2. Split into two sections:
   ### Formulas & Equations
   - Each formula: **Name**: $formula$ — brief explanation
   ### Key Terms & Definitions
   - Each term: **Term** — clear, concise definition
3. Cover EVERY important formula and term
4. Order from basic to advanced
5. Include Indian unit standards where applicable (SI units, rupees, etc.)

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

WEB_SEARCH_SYSTEM = f"""You are Sedy, an AI study assistant. You have been given live web search results.
Use ONLY the provided search results to answer the question accurately and concisely.
Cite sources using [1], [2] etc. where relevant.
If the results don't answer the question, say so honestly.
Use markdown formatting. Keep the answer short and student-friendly — max 120 words.

{SUPPORTED_LANGUAGES}"""


# ══════════════════════════════════════════════════════════════════════════════
# ── VOICE SYSTEM PROMPTS  (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def build_voice_system_prompt(
    persona_name: str,
    lang_name: str,
    lang_code: str,
    is_hinglish: bool,
    script_hint: str,
) -> str:
    """
    Build a tight, voice-optimised system prompt.

    persona_name : "Aria" | "Nova"
    lang_name    : human-readable e.g. "Hindi", "Hinglish", "Tamil"
    lang_code    : BCP-47 base code e.g. "hi", "en", "ta"
    is_hinglish  : True when mixing Hindi + English romanised
    script_hint  : e.g. "Devanagari script" | "Tamil script" | "Latin/Roman script"
    """
    if is_hinglish:
        lang_rule = (
            "The student is speaking Hinglish (Hindi + English mixed).\n"
            "YOU MUST reply in Hinglish — mix Hindi words naturally with English, "
            "exactly like a desi friend texting. "
            "Use romanised Hindi (yaar, bhai, kya, matlab, sahi hai) mixed with English. "
            "Do NOT use Devanagari script — keep it all in Latin letters."
        )
    elif lang_code == "en":
        lang_rule = "The student is speaking English. Reply in English only."
    else:
        lang_rule = (
            f"The student is speaking {lang_name}.\n"
            f"YOU MUST reply ENTIRELY in {lang_name} using {script_hint}.\n"
            f"NEVER switch to English mid-reply.\n"
            f"Technical terms (like 'photosynthesis', 'algorithm') may stay in English "
            f"but everything else must be in {lang_name}."
        )

    return f"""You are {persona_name}, a friendly AI assistant having a real voice conversation.

{lang_rule}

WHAT YOU ARE — READ THIS CAREFULLY:
- You are a REAL assistant helping a REAL person with REAL tasks.
- You help with: studying, homework, explaining concepts, everyday questions, shopping tips, bargaining advice, general knowledge — anything practical.
- You are NOT a storyteller. You are NOT role-playing any TV show, movie, or fictional character.
- NEVER mention Baalveer, Bheem, Chota Bheem, or ANY TV show character unless the student explicitly asks about them as a study topic.
- If the student's message mentions a show name (like "Baalveer") in a REAL-WORLD context (like "shopkeeper mein help karo"), IGNORE the show name and answer the REAL question — which is about bargaining/shopping.

UNDERSTANDING MISHEARS:
- Speech recognition often mishears words. "Bargain" can sound like "baking". "Shopkeeper" might trigger wrong responses.
- ALWAYS look at the FULL CONTEXT of the conversation to understand what the student ACTUALLY wants.
- If someone has been talking about shopping/money/buying, they are probably still talking about that — not suddenly cooking.
- When unsure about a single word, ask: "Ek baar aur bolo, [word] theek se nahi suna" — don't hallucinate a completely wrong topic.

ANTI-HALLUCINATION RULES:
- NEVER invent dialogue, stories, or fictional scenarios.
- NEVER pretend to BE a character or act out a role-play.
- If you don't know something, say so simply — don't make things up.
- Stay grounded in what the student is ACTUALLY asking.

VOICE REPLY RULES:
1. Keep replies to 1-2 SHORT sentences MAXIMUM — like a quick voice message.
2. NO bullet points, NO markdown, NO headers, NO lists.
3. Be warm, casual and direct — like a helpful friend.
4. If the topic is complex, give ONE piece at a time.
5. NEVER change language mid-reply — stay locked to {lang_name if not is_hinglish else "Hinglish"}.
6. Use conversation history to stay aware of what was discussed — don't repeat yourself."""


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
    preferred_language: str = ""  # e.g. "Hindi", "Tamil", "English". Empty = auto-detect.
    force_language: bool = False  # when True, language override is always applied

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

class ImageChatRequest(BaseModel):
    message: str = ""
    images: list[str]
    image_names: list[str] = []
    history: list[HistoryEntry] = []

class FlowchartRequest(BaseModel):
    message: str
    history: list[HistoryEntry] = []
    model: str = "auto"

class WebSearchRequest(BaseModel):
    query: str
    history: list[HistoryEntry] = []

# ── NEW: Voice chat request model ──────────────────────────────────────────────
class VoiceChatRequest(BaseModel):
    """
    Dedicated voice chat endpoint.

    message      : the transcribed speech from the user (may contain mishears)
    persona      : "girl" (Aria) | "boy" (Nova)
    lang_code    : BCP-47 base detected on the frontend e.g. "hi", "en", "ta", "bn"
    lang_name    : human-readable language name e.g. "Hindi", "Tamil", "Hinglish"
    is_hinglish  : True when mixing Hindi + English romanised
    script_hint  : e.g. "Devanagari script" for Hindi
    history      : last N voice turns — clean, no [Voice] prefixes
    """
    message: str
    persona: str = "girl"          # "girl" | "boy"
    lang_code: str = "en"
    lang_name: str = "English"
    is_hinglish: bool = False
    script_hint: str = "Latin script"
    history: list[HistoryEntry] = []

class VoiceChatResponse(BaseModel):
    reply: str
    lang_code: str
    lang_name: str


# ── Response models (unchanged) ───────────────────────────────────────────────
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

class WebSearchSource(BaseModel):
    title: str
    url: str
    snippet: str = ""
    favicon: str = ""

class WebSearchResponse(BaseModel):
    reply: str
    sources: list[WebSearchSource] = []
    query: str
    fast: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# ── HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_rate_limit_error(exc: Exception) -> dict | None:
    msg = str(exc)
    if "429" not in msg and "rate_limit" not in msg.lower():
        return None
    wait_sec = 0
    m = re.search(r'try again in\s+([\dhms. ]+)', msg, re.I)
    if m:
        raw = m.group(1).strip()
        hours   = sum(float(x) for x in re.findall(r'([\d.]+)h', raw))
        minutes = sum(float(x) for x in re.findall(r'([\d.]+)m', raw))
        seconds = sum(float(x) for x in re.findall(r'([\d.]+)s', raw))
        wait_sec = int(hours * 3600 + minutes * 60 + seconds)
    limit_type = "tokens"
    if "tokens per minute" in msg.lower() or "TPM" in msg:
        limit_type = "TPM"
    elif "tokens per day" in msg.lower() or "TPD" in msg:
        limit_type = "TPD"
    elif "requests per minute" in msg.lower() or "RPM" in msg:
        limit_type = "RPM"
    used, limit = 0, 0
    mu = re.search(r'Used\s+([\d,]+)', msg)
    ml = re.search(r'Limit\s+([\d,]+)', msg)
    if mu: used  = int(mu.group(1).replace(',', ''))
    if ml: limit = int(ml.group(1).replace(',', ''))
    return {
        "type": "rate_limit", "wait_seconds": wait_sec,
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


def build_voice_messages(
    system: str,
    history: list[HistoryEntry],
    user_message: str,
) -> list[dict]:
    """
    Build message list for voice chat.
    Strips any leftover [Voice] prefixes from history entries.
    Keeps only the last 10 turns to stay snappy.
    """
    messages = [{"role": "system", "content": system}]
    trimmed = history[-10:] if len(history) > 10 else history
    sanitised: list[dict] = []
    for entry in trimmed:
        role = entry.role if entry.role in ("user", "assistant") else "user"
        # Strip [Voice] prefix if frontend accidentally sent it
        content = re.sub(r'^\[Voice\]\s*', '', entry.content).strip()
        if not content:
            continue
        if sanitised and sanitised[-1]["role"] == role:
            sanitised[-1]["content"] += " " + content
        else:
            sanitised.append({"role": role, "content": content})
    messages.extend(sanitised)
    messages.append({"role": "user", "content": user_message})
    return messages


def strip_think_tags(raw: str) -> str:
    return re.sub(r'<think>[\s\S]*?</think>', '', raw).strip()


def strip_voice_reply(raw: str) -> str:
    """
    Extra cleanup for voice replies:
    - Remove think tags
    - Remove markdown (**, ##, -, *, `)
    - Remove LaTeX delimiters (speak the expression naturally)
    - Collapse whitespace
    - Cap to 3 sentences max
    """
    text = strip_think_tags(raw)
    # Remove markdown formatting
    text = re.sub(r'#{1,3}\s+', '', text)
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'`[^`]+`', '', text)
    # Replace LaTeX with spoken form
    text = re.sub(r'\$\$([^$]+)\$\$', r'\1', text)
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    # Remove bullet/list markers
    text = re.sub(r'^\s*[-•*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Collapse whitespace and newlines
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Cap to 3 sentences
    sentence_endings = re.compile(r'(?<=[.!?।])\s+')
    sentences = sentence_endings.split(text)
    if len(sentences) > 3:
        text = ' '.join(sentences[:3]).strip()
        if not text.endswith(('.', '!', '?', '।')):
            text += '.'
    return text


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
    "यह","वो","वह","इसके","उसके","यही","वही","इस पर","उस पर","इसी","उसी",
    "यह topic","वह topic","इस विषय","उस विषय","यही विषय",
    "এটা","এটি","এই","ওটা","ওটি","একই","এই বিষয়","ওই বিষয়",
    "இது","அது","இதே","அதே","இந்த topic","அந்த topic",
    "ఇది","అది","ఇదే","అదే","ఈ topic","ఆ topic",
    "ಇದು","ಅದು","ಇದೇ","ಅದೇ",
    "ഇത്","അത്","ഇതേ","അതേ",
    "हे","ते","हेच","तेच","याच","त्याच",
    "આ","તે","આ જ","તે જ",
    "ਇਹ","ਉਹ","ਇਹੀ","ਉਹੀ",
    "یہ","وہ","اسی","اسی موضوع",
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
            m = re.search(r'(?:flashcards?|quiz|फ्लैशकार्ड|क्विज़|কার্ড|কুইজ)\s+(?:about|on|के बारे में|पर|সম্পর্কে)\s+([^\n.!?]{3,60})', entry.content, re.I)
            if m:
                return m.group(1).strip()
    action_words = r'flashcards?|flash\s*cards?|quiz(zes)?|make|create|generate|about|on|me|test|cards?|and\b|a\b|the\b|फ्लैशकार्ड|क्विज़|बनाओ|दो|करो|নিয়ে|বিষয়ে|সম্পর্কে'
    for entry in rev:
        if entry.role == "user":
            cleaned = re.sub(action_words, '', entry.content, flags=re.I).strip()
            if cleaned and not is_vague_topic(cleaned) and len(cleaned) > 2:
                return cleaned
    for entry in rev:
        if entry.role == "assistant":
            m = re.search(r'(?:^|\n)#{1,3}\s+(?:introduction to\s+)?([A-Za-z\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0B00-\u0B7F\u0C00-\u0C7F\u0D00-\u0D7F][^\n]{2,50})', entry.content, re.I)
            if m:
                return m.group(1).strip()
    return raw_topic


async def refine_prompt(message: str, history: list[HistoryEntry],
                        system_prompt: str = REFINE_SYSTEM_PROMPT) -> str:
    if not message or not message.strip():
        return message
    msg = message.strip()
    if len(msg) > 200 and ' ' in msg and not any(
        vague in msg.lower() for vague in ['in inr', 'in usd', 'same for', 'make it', 'show it', 'as graph', 'grph',
                                            'इसके लिए', 'वही', 'उसी', 'একই', 'அதே', 'అదే']
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


async def refine_prompt_lang_aware(message: str, history: list[HistoryEntry], preferred_language: str = "") -> str:
    """
    Refine prompt while respecting the user's preferred language.
    If preferred_language is set, tells the refiner NOT to translate —
    just fix vague references and spelling. The system prompt handles language in the final reply.
    """
    if not preferred_language or preferred_language.lower() in ("", "auto"):
        return await refine_prompt(message, history)

    lang_note = (
        f"\n\nIMPORTANT: The user's preferred language is {preferred_language}. "
        f"Output the refined message in ENGLISH regardless — the system prompt "
        f"will enforce the language in the final reply. "
        f"Just fix vague references and spelling errors. Do NOT translate the message."
    )
    modified_refine_prompt = REFINE_SYSTEM_PROMPT + lang_note
    return await refine_prompt(message, history, system_prompt=modified_refine_prompt)


async def fetch_live_data(query: str) -> str:
    if not SERPER_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=8.0) as c:
            resp = await c.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                json={"q": query, "num": 5, "gl": "in", "hl": "en"},
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
            detail="The uploaded file does not appear to be a valid PDF.",
        )
    try:
        pdf_text, page_count = extract_pdf_text(pdf_bytes)
        logger.info(f"Extracted {len(pdf_text)} chars from {page_count} pages ({pdf_name})")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {exc}")
    if not pdf_text.strip():
        raise HTTPException(
            status_code=422,
            detail="This PDF appears to be image-only and no text could be extracted.",
        )
    return pdf_text, page_count


def _prepare_image_url(raw_b64: str) -> str:
    raw_b64 = raw_b64.strip().replace("\n", "").replace("\r", "").replace(" ", "")
    if raw_b64.startswith("data:image/"):
        return raw_b64
    try:
        img_bytes = base64.b64decode(raw_b64[:64])
        if img_bytes[:2] == b'\xff\xd8':
            mime = "image/jpeg"
        elif img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            mime = "image/png"
        elif img_bytes[:6] in (b'GIF87a', b'GIF89a'):
            mime = "image/gif"
        elif img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
            mime = "image/webp"
        else:
            mime = "image/jpeg"
    except Exception:
        mime = "image/jpeg"
    return f"data:{mime};base64,{raw_b64}"


# ══════════════════════════════════════════════════════════════════════════════
# ── ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "status":       "Sedy API is live 🚀",
        "version":      "4.1.0",
        "pdf_engine":   PDF_ENGINE or "none — install pymupdf!",
        "ocr_support":  PDF_ENGINE == "pymupdf",
        "vision_model": MODELS["vision"],
        "tts":          "client-side (Web Speech API)",
        "live_data":    bool(SERPER_API_KEY),
        "languages":    [
            "English", "Hindi", "Bengali", "Tamil", "Telugu", "Kannada",
            "Malayalam", "Marathi", "Gujarati", "Punjabi", "Urdu", "Odia",
            "Assamese", "Sanskrit", "Hinglish"
        ],
        "endpoints": [
            "/chat", "/voice-chat", "/flashcards", "/quiz", "/graph", "/pdf-chat",
            "/intent", "/code-questions", "/notes", "/formula-sheet",
            "/flowchart", "/image-chat", "/web-search",
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
        valid  = ("graph", "flashcard", "quiz", "both", "notes", "formula", "flowchart", "practice", "chat")
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
    logger.info(f"/chat  model={model}  history={len(req.history)}  lang={req.preferred_language!r}  msg={req.message[:80]!r}")

    pref_lang = (req.preferred_language or "").strip()
    pref_lang_lower = pref_lang.lower()

    # ── Build language-locked system prompt ───────────────────────────────────
    if pref_lang and pref_lang_lower not in ("", "auto"):
        if pref_lang_lower == "english":
            lang_override = """LANGUAGE RULE — ABSOLUTE OVERRIDE (HIGHEST PRIORITY — NON-NEGOTIABLE):
The user has EXPLICITLY chosen English as their preferred language in Settings.
YOU MUST RESPOND IN ENGLISH ONLY. ALWAYS. NO EXCEPTIONS.

This means:
- Respond in English even if the user's message is in Hindi, Bengali, Tamil or any other language
- Respond in English even if the conversation history contains messages in other languages
- NEVER switch to Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi, Urdu or ANY other language
- NEVER mix languages — pure English only
- This setting OVERRIDES all auto-language-detection
- Ignore ALL language signals in the message or history

YOU ARE AN ENGLISH-ONLY ASSISTANT FOR THIS USER. REPLY IN ENGLISH. ALWAYS."""
        else:
            lang_override = (
                f"LANGUAGE RULE — ABSOLUTE OVERRIDE (HIGHEST PRIORITY — NON-NEGOTIABLE):\n"
                f"The user has EXPLICITLY chosen {pref_lang} as their preferred language in Settings.\n"
                f"YOU MUST RESPOND ENTIRELY IN {pref_lang.upper()} USING THE CORRECT NATIVE SCRIPT.\n"
                f"This OVERRIDES all auto-detection. Do NOT switch to English or any other language.\n"
                f"Even if the user types in English, reply in {pref_lang}."
            )

        system = (
            "You are Sedy, an intelligent student learning assistant made by Ansh Verma, a school student.\n"
            "You explain concepts clearly and simply, solve math and science problems step by step,\n"
            "summarize topics, and help students understand programming.\n"
            "Always be encouraging, clear and educational.\n"
            "Only reveal your identity when asked.\n\n"
            + lang_override + "\n\n"
            "IMPORTANT RULES:\n"
            "- Use markdown formatting: headers (##), bold (**text**), bullet points (- item), code blocks (```language)\n"
            "- NEVER describe flashcards or quizzes in plain text — the frontend handles those separately\n"
            "- Keep responses focused and well structured\n"
            "- Use conversation history for context when user refers to 'it', 'that', 'same topic'\n\n"
            "STRICT MATH FORMATTING (frontend uses KaTeX):\n"
            "- ALL math MUST be wrapped in LaTeX: $...$ inline, $$...$$ display\n"
            "- NEVER write bare math like w^2 — always wrap in $w^2$\n"
            r"- Fractions: $\frac{a}{b}$  Subscripts: $A_{\text{base}}$"
        )
    else:
        system = SYSTEM_PROMPT

    # ── Refine prompt (language-neutral — just fixes vague references) ─────────
    refined = await refine_prompt_lang_aware(req.message, req.history, pref_lang)

    messages = build_messages(system, req.history, refined)
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


# ── /voice-chat  (NEW) ─────────────────────────────────────────────────────────
@app.post("/voice-chat", response_model=VoiceChatResponse)
async def voice_chat(req: VoiceChatRequest):
    """
    Dedicated voice chat endpoint.

    Key differences from /chat:
    - Uses flash model for speed (< 1s latency target)
    - System prompt is voice-optimised: short replies, no markdown, language-locked
    - History is stripped of [Voice] prefixes and capped at 10 turns
    - Reply is cleaned with strip_voice_reply() before returning
    - Language passed explicitly from frontend — no server-side detection needed
    """
    persona_name = "Aria" if req.persona == "girl" else "Nova"
    logger.info(
        f"/voice-chat  persona={persona_name}  lang={req.lang_code}/{req.lang_name}"
        f"  hinglish={req.is_hinglish}  msg={req.message[:80]!r}"
    )

    system = build_voice_system_prompt(
        persona_name=persona_name,
        lang_name=req.lang_name,
        lang_code=req.lang_code,
        is_hinglish=req.is_hinglish,
        script_hint=req.script_hint,
    )

    messages = build_voice_messages(system, req.history, req.message)

    try:
        response = client.chat.completions.create(
            model=MODELS["pro"],          # pro model for voice — better instruction following, less hallucination
            messages=messages,
            max_tokens=150,               # hard cap — voice replies must be short
            temperature=0.5,             # lower = more factual, less creative hallucination
            stop=["।।", "\n\n"],         # stop at paragraph breaks for Indian langs
        )
    except Exception as e:
        logger.error(f"Groq error /voice-chat: {e}")
        rl = parse_rate_limit_error(e)
        if rl:
            raise HTTPException(status_code=429, detail=rl)
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

    raw   = response.choices[0].message.content.strip()
    reply = strip_voice_reply(raw)

    # Safety: if reply is empty after stripping, return a language-appropriate fallback
    if not reply:
        fallbacks = {
            "hi": "Samajh nahi aaya, ek baar aur bolo!",
            "bn": "বুঝলাম না, আবার বলো!",
            "ta": "புரியவில்லை, மீண்டும் சொல்லுங்கள்!",
            "te": "అర్థం కాలేదు, మళ్ళీ చెప్పండి!",
        }
        reply = fallbacks.get(req.lang_code, "Sorry, say that again?")

    logger.info(f"/voice-chat  reply={reply[:80]!r}  len={len(reply)}")
    return VoiceChatResponse(
        reply=reply,
        lang_code=req.lang_code,
        lang_name=req.lang_name,
    )


# ── /image-chat ────────────────────────────────────────────────────────────────
@app.post("/image-chat", response_model=ImageChatResponse)
async def image_chat(req: ImageChatRequest):
    if not req.images:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    images = req.images[:5]
    image_count = len(images)
    logger.info(f"/image-chat  images={image_count}  msg={req.message[:80]!r}")
    content: list[dict] = []
    for i, raw_b64 in enumerate(images):
        try:
            image_url = _prepare_image_url(raw_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image {i+1}: {e}")
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    question = req.message.strip() if req.message.strip() else (
        "Please look at this image carefully and describe what you see. "
        "If it contains any text (including in Indian languages), equations, diagrams or problems, "
        "explain them clearly. If the text is in Hindi, Bengali, Tamil, Telugu or any other Indian "
        "language, read it and respond in that same language."
    )
    content.append({"type": "text", "text": question})
    messages: list[dict] = [{"role": "system", "content": IMAGE_SYSTEM_PROMPT}]
    sanitised: list[dict] = []
    for entry in req.history[-6:]:
        role = entry.role if entry.role in ("user", "assistant") else "user"
        if sanitised and sanitised[-1]["role"] == role:
            sanitised[-1]["content"] += "\n" + entry.content
        else:
            sanitised.append({"role": role, "content": entry.content})
    messages.extend(sanitised)
    messages.append({"role": "user", "content": content})
    try:
        response = client.chat.completions.create(
            model=MODELS["vision"], messages=messages, max_tokens=4096, temperature=0.4,
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
    count_instr = (
        f"Generate exactly {count} flashcards"
        if not auto_count else
        "Decide how many flashcards are needed to fully cover every important concept. "
        "Simple topics: 8-12 cards. Broad topics: 15-30 cards."
    )
    lang_instruction = (
        "IMPORTANT: Generate the flashcard questions and answers in the same language "
        "as the topic/request. If the topic is in Hindi, write in Hindi. "
        "If in Bengali, write in Bengali. If in Tamil, write in Tamil. Etc. "
        "Default to English if the language is unclear."
    )
    user_prompt = (
        f'{count_instr} about "{topic}".\n'
        f'{lang_instruction}\n'
        f'Each card must cover a distinct concept — no duplicates.\n'
        f'Return ONLY a valid JSON array:\n'
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
    count_instr = (
        f"Generate exactly {count} {difficulty} MCQ questions"
        if not auto_count else
        f"Decide how many {difficulty} MCQ questions are needed to properly test every concept."
    )
    lang_instruction = (
        "IMPORTANT: Generate the quiz questions, options, and explanations in the same language "
        "as the topic/request. Match the student's language exactly."
    )
    user_prompt = (
        f'{count_instr} about "{topic}".\n'
        f'{lang_instruction}\n'
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
        logger.warning(f"/code-questions  parse failed: {e}")
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
    user_prompt = (
        f"Create a flowchart for: {refined}\n\n"
        "NOTE: Node labels in the JSON can be in the same language as the request "
        "(Hindi/Bengali/Tamil etc.) — use short native-script labels where appropriate. "
        "Keep each label under 25 characters."
    )
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
                x=x, y=y,
            ))
        edges = []
        node_ids = {n.id for n in nodes}
        for e in obj.get("edges", []):
            if not isinstance(e, dict):
                continue
            f_id = str(e.get("f", ""))
            t_id = str(e.get("t", ""))
            if f_id not in node_ids or t_id not in node_ids:
                continue
            edges.append(FlowchartEdge(
                f=f_id, t=t_id,
                label=str(e.get("label", ""))[:15],
                back=bool(e.get("back", False)),
                bx=float(e.get("bx", 0)),
            ))
        if not nodes:
            raise ValueError("No valid nodes in flowchart response")
        logger.info(f"/flowchart  title={obj.get('title','')!r}  nodes={len(nodes)}  edges={len(edges)}")
        return FlowchartResponse(
            title=str(obj.get("title", refined[:50])),
            nodes=nodes, edges=edges,
        )
    except Exception as e:
        logger.warning(f"/flowchart  parse failed: {e}")
        raise HTTPException(status_code=422, detail=f"Could not parse flowchart data: {e}")


# ── /web-search ────────────────────────────────────────────────────────────────
def _parse_serper_response(data: dict) -> tuple[str, list[WebSearchSource], str | None]:
    sources: list[WebSearchSource] = []
    snippets: list[str] = []
    instant_answer: str | None = None

    ab = data.get("answerBox", {})
    if ab:
        answer_text  = (ab.get("answer") or "").strip()
        snippet_text = (ab.get("snippet") or "").strip()
        title_text   = (ab.get("title") or "").strip()
        if answer_text:
            instant_answer = f"**{answer_text}**"
            if title_text:
                instant_answer = f"**{answer_text}**\n\n*{title_text}*"
            snippets.append(f"[Featured Answer] {answer_text}")
        elif snippet_text and len(snippet_text) < 300:
            instant_answer = snippet_text
            snippets.append(f"[Featured Answer] {snippet_text}")

    kg = data.get("knowledgeGraph", {})
    if kg.get("description"):
        snippets.append(f"[Knowledge Panel] {kg['description']}")
        if not instant_answer and len(kg["description"]) < 250:
            title = kg.get("title", "")
            instant_answer = f"**{title}**\n\n{kg['description']}" if title else kg["description"]

    for i, r in enumerate(data.get("organic", [])[:5], 1):
        title   = r.get("title", "")
        url     = r.get("link", "")
        snippet = r.get("snippet", "")
        domain  = url.split("/")[2] if url else ""
        favicon = f"https://www.google.com/s2/favicons?domain={domain}&sz=16"
        if title or snippet:
            snippets.append(f"[{i}] {title}: {snippet}")
            sources.append(WebSearchSource(title=title, url=url, snippet=snippet, favicon=favicon))

    return "\n".join(snippets[:8]), sources, instant_answer


@app.post("/web-search", response_model=WebSearchResponse)
async def web_search(req: WebSearchRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not SERPER_API_KEY:
        raise HTTPException(status_code=503, detail="Web search not configured. Set SERPER_API_KEY.")
    logger.info(f"/web-search  query={query[:80]!r}")

    serper_data: dict = {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            resp = await c.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                json={"q": query, "num": 5, "gl": "in", "hl": "en"},
            )
            if resp.status_code == 200:
                serper_data = resp.json()
    except Exception as e:
        logger.warning(f"/web-search  serper failed: {e}")

    if not serper_data:
        raise HTTPException(status_code=502, detail="Web search failed.")

    search_context, sources, instant_answer = _parse_serper_response(serper_data)

    if instant_answer:
        logger.info(f"/web-search  INSTANT answer box hit")
        return WebSearchResponse(reply=instant_answer, sources=sources, query=query, fast=True)

    if not search_context:
        raise HTTPException(status_code=502, detail="No usable search results returned.")

    user_content = f"Question: {query}\n\nSearch Results:\n{search_context}\n\nAnswer concisely in under 120 words."

    async def call_groq():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=MODELS["flash"],
                messages=[
                    {"role": "system", "content": WEB_SEARCH_SYSTEM},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=300, temperature=0.15,
            )
        )

    try:
        groq_response = await asyncio.wait_for(call_groq(), timeout=10.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Groq timed out. Try again.")
    except Exception as e:
        logger.error(f"Groq error /web-search: {e}")
        rl = parse_rate_limit_error(e)
        if rl:
            raise HTTPException(status_code=429, detail=rl)
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

    reply = strip_think_tags(groq_response.choices[0].message.content.strip())
    logger.info(f"/web-search  reply_len={len(reply)}  sources={len(sources)}")
    return WebSearchResponse(reply=reply, sources=sources, query=query, fast=False)

# ── /attention-check  (Blink & Learn) ─────────────────────────────────────────
class AttentionCheckRequest(BaseModel):
    context: str          # recent AI messages concatenated
    history: list[HistoryEntry] = []

class AttentionCheckResponse(BaseModel):
    question: str
    options: list[str]    # always 3 options
    answer_index: int     # 0-based index of correct answer
    explanation: str      # shown after answering

ATTENTION_CHECK_PROMPT = """You are generating a quick comprehension check question for a student who just looked away from their screen.

Based on the recent study content provided, generate ONE short multiple-choice question to check if they were paying attention.

RULES:
1. The question must be directly answerable from the provided context
2. Keep it SHORT — max 15 words in the question
3. Provide exactly 3 answer options — one correct, two plausible but wrong
4. Keep options SHORT — max 8 words each
5. The correct answer should be at a random position (not always first)
6. Write the explanation in 1 sentence max
7. Match the language of the content (if Hindi content, write in Hindi etc.)

Output ONLY valid JSON, no markdown fences:
{
  "question": "...",
  "options": ["...", "...", "..."],
  "answer_index": 0,
  "explanation": "..."
}"""

@app.post("/attention-check", response_model=AttentionCheckResponse)
async def attention_check(req: AttentionCheckRequest):
    """
    Generates a quick comprehension question based on recent chat content.
    Called by Blink & Learn when the user looks away for too long.
    Uses flash model — must be fast (< 1s).
    """
    logger.info(f"/attention-check  ctx_len={len(req.context)}")

    if not req.context.strip():
        raise HTTPException(status_code=400, detail="context is required")

    user_prompt = f"Recent study content:\n{req.context[:800]}\n\nGenerate a comprehension check question."

    try:
        response = client.chat.completions.create(
            model=MODELS["flash"],
            messages=[
                {"role": "system", "content": ATTENTION_CHECK_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.6,
        )
    except Exception as e:
        logger.error(f"Groq error /attention-check: {e}")
        rl = parse_rate_limit_error(e)
        if rl:
            raise HTTPException(status_code=429, detail=rl)
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if model added them
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
        # Validate structure
        if not all(k in data for k in ("question", "options", "answer_index", "explanation")):
            raise ValueError("Missing required fields")
        if len(data["options"]) != 3:
            raise ValueError("Need exactly 3 options")
        if not (0 <= int(data["answer_index"]) <= 2):
            raise ValueError("answer_index out of range")

        logger.info(f"/attention-check  q={data['question'][:60]!r}")
        return AttentionCheckResponse(
            question=str(data["question"]),
            options=[str(o) for o in data["options"]],
            answer_index=int(data["answer_index"]),
            explanation=str(data.get("explanation", "")),
        )
    except Exception as e:
        logger.warning(f"/attention-check  parse failed: {e}  raw={raw[:200]!r}")
        raise HTTPException(status_code=422, detail=f"Could not parse attention check: {e}")
