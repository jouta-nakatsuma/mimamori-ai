import os
import re
import json
import time
import unicodedata
import logging
import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Any, Tuple
import httpx
import yaml
import smtplib, ssl, pathlib
from email.message import EmailMessage

from fastapi import FastAPI, HTTPException, Request, Query
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# -------------------------
# Bootstrapping
# -------------------------
ENV_PATH = pathlib.Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH, override=True, encoding="utf-8")
app = FastAPI()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# --- Env / Config ---
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
oa = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

RETURN_FRIENDLY    = os.getenv("RETURN_FRIENDLY", "1") == "1"   # flagged時も代替文をanswerに返す
RATE_LIMIT_ENABLE  = os.getenv("RATE_LIMIT_ENABLE", "0") == "1" # 任意: 連続flaggedのクールダウン

NG_WORDS_PATH      = os.getenv("NG_WORDS_PATH", "lexicons/ng_words.yaml")
def load_ng_words():
    pub = Path("lexicons/ng_words.yaml")
    loc = Path("lexicons/ng_words.local.yaml")
    def _load(p):
        return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    base = _load(pub) or {}
    override = _load(loc) or {}
    # 単純マージ（同名キーはローカル優先、値はリスト想定で連結・重複除去）
    merged = dict(base)
    for k, v in override.items():
        a = list(dict.fromkeys((base.get(k) or []) + (v or [])))
        merged[k] = a
    return merged

RL_WINDOW_SEC      = int(os.getenv("RL_WINDOW_SEC", "60"))
RL_MAX_ALERTS      = int(os.getenv("RL_MAX_ALERTS", "3"))
RL_COOLDOWN_SEC    = int(os.getenv("RL_COOLDOWN_SEC", "120"))

HTTPX_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=30.0)

TEACHER_PROMPT_PATH = os.getenv("TEACHER_PROMPT_PATH", "prompts/teacher.txt")
def _load_teacher_prompt():
    try:
        with open(TEACHER_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""
TEACHER_PROMPT = _load_teacher_prompt()
REVIEW_MODEL = os.getenv("REVIEW_MODEL", "gpt-5")

# --- logging (ADD/REPLACE) ---
LOG_DIR = "logs"
PROMPTS_JSONL = os.path.join(LOG_DIR, "prompts.jsonl")
TURNS_JSONL   = os.path.join(LOG_DIR, "turns.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

def append_turn(role: str, text: str, thread_id: str = "", actor: str | None = None, meta: dict | None = None):
    rec = {
        "ts": datetime.datetime.now().isoformat(),
        "thread_id": thread_id or "",
        "actor": actor or "",
        "role": role,     # "user" / "assistant"
        "text": text or "",
        "meta": meta or {},   # 例: {"kind":"raw"|"shown", "blocked":True|False}
    }
    with open(TURNS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_prompt_log(text: str, thread_id: str = ""):
    rec = {"ts": datetime.datetime.now().isoformat(), "thread_id": thread_id, "text": text}
    with open(PROMPTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

class ChatReq(BaseModel):
    message: str
    thread_id: str | None = None

NIGHTLY_TOKEN = os.getenv("NIGHTLY_TOKEN", "changeme")

# -------------------------
# Logging
# -------------------------
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("guardian")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler("logs/guardian.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def log_event(side: str, hits, cats, text: str, ip: str):
    preview = (text or "")[:120].replace("\n", " ")
    try:
        logger.info(json.dumps({
            "side": side, "hits": hits, "cats": cats, "ip": ip, "text_preview": preview
        }, ensure_ascii=False))
    except Exception:
        pass

# -------------------------
# Utilities
# -------------------------
THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
def sanitize(text: str) -> str:
    if not text:
        return text
    text = THINK_RE.sub("", text)
    return text.strip()

def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").lower()

# -------------------------
# Lexicon (YAML) Loader
# -------------------------
class Lexicon:
    def __init__(self):
        self.version: int = 0
        self.path: str = NG_WORDS_PATH
        self.categories: Dict[str, Dict[str, Any]] = {}
        self.patterns: Dict[str, List[re.Pattern]] = {}
        self.combos: List[Dict[str, Any]] = []
        self.exclusions: List[re.Pattern] = []
        self._mtime: float = 0.0

    def load(self) -> Tuple[bool, str]:
        """YAMLを読み込み・正規化・コンパイル。成功時 True, 失敗時 False と理由を返す。"""
        if not os.path.exists(self.path):
            return False, f"Lexicon file not found: {self.path}"
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            return False, f"YAML load error: {e}"

        # version
        self.version = int(data.get("version", 1))

        # categories
        self.categories = data.get("categories", {}) or {}

        # patterns -> {category: [regex...]}
        raw_patterns = data.get("patterns", {}) or {}
        compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for cat, regs in raw_patterns.items():
            lst: List[re.Pattern] = []
            for rx in regs or []:
                try:
                    lst.append(re.compile(rx))
                except re.error as e:
                    logger.warning(f"[lexicon] regex compile error in category '{cat}': {rx} ({e})")
            compiled_patterns[cat] = lst
        self.patterns = compiled_patterns

        # combos -> list of {category, term, intent, window}
        self.combos = []
        for c in data.get("combos", []) or []:
            try:
                cat = c["category"]
                term = re.compile(c["term"])
                intent = re.compile(c["intent"])
                window = int(c.get("window", 12))
                self.combos.append({"category": cat, "term": term, "intent": intent, "window": window})
            except Exception as e:
                logger.warning(f"[lexicon] combo compile error: {c} ({e})")

        # exclusions
        self.exclusions = []
        for ex in data.get("exclusions", []) or []:
            try:
                self.exclusions.append(re.compile(ex))
            except re.error as e:
                logger.warning(f"[lexicon] exclusion regex compile error: {ex} ({e})")

        # timestamp
        try:
            self._mtime = os.path.getmtime(self.path)
        except Exception:
            self._mtime = time.time()

        return True, "ok"

    def notify_enabled(self, cat: str) -> bool:
        meta = self.categories.get(cat, {})
        return bool(meta.get("notify", True))

    def friendly(self, hits: List[str]) -> str:
        # 最初にヒットしたカテゴリのテンプレを優先
        for h in hits:
            meta = self.categories.get(h, {})
            if "friendly" in meta:
                return str(meta["friendly"])
        # 既定
        return "ごめんね、そのお願いには応えられないよ。"

    def info(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "path": self.path,
            "mtime": self._mtime,
            "categories": list(self.categories.keys()),
            "patterns_counts": {k: len(v) for k, v in self.patterns.items()},
            "combos_counts": len(self.combos),
            "exclusions_counts": len(self.exclusions),
        }

lex = Lexicon()
ok, msg = lex.load()
if not ok:
    logger.warning(f"[lexicon] initial load failed: {msg}")

def excluded(text_norm: str) -> bool:
    for ex in lex.exclusions:
        if ex.search(text_norm):
            return True
    return False

def has_pair_compiled(term_re: re.Pattern, intent_re: re.Pattern, window: int, text_norm: str) -> bool:
    # term と intent のマッチ位置を列挙し、非重複かつ距離<=window ならヒット
    term_spans = [(m.start(), m.end()) for m in term_re.finditer(text_norm)]
    intent_spans = [(m.start(), m.end()) for m in intent_re.finditer(text_norm)]
    for ts, te in term_spans:
        for is_, ie in intent_spans:
            # 任意方向の近接（重なりは距離0とみなす）
            if is_ >= te:
                dist = is_ - te
            elif ts >= ie:
                dist = ts - ie
            else:
                dist = 0
            if dist <= window:
                return True
    return False

def local_high_risk(text: str) -> List[str]:
    t = _norm(text)
    hits: List[str] = []
    if excluded(t):
        return hits

    # 1) 単独パターン
    for cat, regs in lex.patterns.items():
        if any(r.search(t) for r in regs):
            hits.append(cat)

    # 2) 名詞×意図の近接
    for c in lex.combos:
        if has_pair_compiled(c["term"], c["intent"], c["window"], t):
            if c["category"] not in hits:
                hits.append(c["category"])

    return hits

# -------------------------
# OpenAI Moderation (v1)
# -------------------------
def mod_flag(text: str):
    """OpenAI Moderation。失敗や未設定時は安全側で継続する。"""
    if not oa:
        return False, {"moderation_error": "no_api_key"}
    try:
        m = oa.moderations.create(model="omni-moderation-latest", input=text)
        flagged = bool(m.results[0].flagged)
        cats = m.results[0].categories
        if hasattr(cats, "model_dump"):
            cats = cats.model_dump()
        return flagged, cats
    except Exception as e:
        return False, {"moderation_error": str(e)}

def friendly_reply(hits: List[str]) -> str:
    return lex.friendly(hits)

# -------------------------
# Optional: Simple rate limit (per IP)
# -------------------------
_rl_state = {}  # ip -> {"alerts": [(ts, side), ...], "cooldown_until": ts}

def _rate_limit_check(ip: str):
    """連続flaggedに対する簡易クールダウン。Trueなら通知送信をスキップ（返却は通常通りsafe:false）。"""
    if not RATE_LIMIT_ENABLE:
        return False
    now = time.time()
    st = _rl_state.get(ip, {"alerts": [], "cooldown_until": 0.0})
    if st.get("cooldown_until", 0.0) > now:
        _rl_state[ip] = st
        return True
    st["alerts"] = [(ts, side) for (ts, side) in st.get("alerts", []) if now - ts <= RL_WINDOW_SEC]
    if len(st["alerts"]) >= RL_MAX_ALERTS:
        st["cooldown_until"] = now + RL_COOLDOWN_SEC
        _rl_state[ip] = st
        return True
    _rl_state[ip] = st
    return False

def _rate_limit_record(ip: str, side: str):
    if not RATE_LIMIT_ENABLE:
        return
    now = time.time()
    st = _rl_state.get(ip, {"alerts": [], "cooldown_until": 0.0})
    st["alerts"] = [(ts, sd) for (ts, sd) in st.get("alerts", []) if now - ts <= RL_WINDOW_SEC]
    st["alerts"].append((now, side))
    _rl_state[ip] = st

def send_mail(subject: str, body: str, to: str | None = None, attachments: list | None = None) -> bool:
    """
    attachments: [(filename:str, content:bytes|str, mimetype:str), ...]
    依存: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM, SMTP_TO (.env)
    """
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASS")
    sender = os.getenv("SMTP_FROM") or user
    to = to or os.getenv("SMTP_TO")
    if not (host and port and sender and to):
        return False

    to_list = [addr.strip() for addr in str(to).split(",") if addr.strip()]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(to_list)
    msg.set_content(body)

    if attachments:
        for name, content, mimetype in attachments:
            if isinstance(content, str):
                content = content.encode("utf-8")
            maintype, subtype = (mimetype.split("/", 1) if "/" in mimetype else ("application","octet-stream"))
            msg.add_attachment(content, maintype=maintype, subtype=subtype, filename=name)

    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=30, context=ssl.create_default_context()) as s:
                if user and pwd: s.login(user, pwd)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=30) as s:
                s.ehlo(); s.starttls(context=ssl.create_default_context()); s.ehlo()
                if user and pwd: s.login(user, pwd)
                s.send_message(msg)
        return True
    except Exception as e:
        import logging
        logging.exception(f"mail send failed: {e}")
        return False
    
# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "ollama_host": OLLAMA_HOST,
        "model": OLLAMA_MODEL,
        "openai_v1": bool(oa is not None),
        "friendly": RETURN_FRIENDLY,
        "rate_limit": RATE_LIMIT_ENABLE,
        "lexicon": lex.info(),
    }

@app.get("/debug/mod")
def debug_mod(text: str):
    hits = local_high_risk(text)
    flagged, cats = mod_flag(text)
    return {"hits": hits, "flagged": flagged, "cats": cats}

@app.post("/debug/reload")
def debug_reload():
    ok, msg = lex.load()
    return {"ok": ok, "msg": msg, "lexicon": lex.info()}

@app.post("/cron/nightly")
def cron_nightly(token: str = Query(None)):
    if token != NIGHTLY_TOKEN:
        raise HTTPException(status_code=403, detail="forbidden")

    today = datetime.date.today().isoformat()

    # 1回だけ turns.jsonl を読み込む
    today_turns = []
    if os.path.exists(TURNS_JSONL):
        with open(TURNS_JSONL, encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    if j.get("ts", "").startswith(today):
                        today_turns.append(j)
                except:
                    pass

    # 統一して集計
    user_turns = [j for j in today_turns if j.get("role") == "user"]
    cnt_inputs = len(user_turns)          # 入力件数=ユーザー発話
    cnt_turns  = len(today_turns)         # 入出力レコード件数

    # 抜粋（先頭50文字）をユーザー発話から
    samples = [(j.get("text","") or "")[:50] for j in user_turns[:5]]

    # turnsがまだ無い（初日など）時だけ prompts.jsonl にフォールバック
    if cnt_turns == 0 and os.path.exists(PROMPTS_JSONL):
        cnt_inputs = 0
        samples = []
        with open(PROMPTS_JSONL, encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    if j.get("ts","").startswith(today):
                        cnt_inputs += 1
                        if len(samples) < 5:
                            samples.append((j.get("text","") or "")[:50])
                except:
                    pass

    # 整形テキスト（raw/shownを区別して出力）
    def fmt(j):
        t   = j.get("ts","")
        tid = j.get("thread_id","")
        role= j.get("role","")
        meta= j.get("meta",{}) or {}
        kind= meta.get("kind","")
        blocked = meta.get("blocked", False)

        if role == "assistant" and kind == "raw":
            who = "AI (raw)"
        elif role == "assistant" and kind == "shown":
            who = "AI (shown, blocked)" if blocked else "AI"
        elif role == "assistant":
            who = "AI"
        else:
            who = "You"
        return f"{t} [{who}] ({tid}):\n{j.get('text','')}\n"

    transcript_txt = "\n".join(fmt(j) for j in today_turns)

    body = "[日次ダイジェスト]\n"
    body += f"日付: {today}\n"
    body += f"入力件数: {cnt_inputs}\n"
    body += f"入出力レコード件数: {cnt_turns}\n"
    body += ("抜粋(先頭50文字):\n - " + "\n - ".join(samples)) if samples else "抜粋: なし"

    attachments = [(f"transcript-{today}.txt", transcript_txt or "(本日レコードなし)", "text/plain")]

    ok = send_mail(subject=f"[みまもりAI 日報] {today} 入力{cnt_inputs}/レコード{cnt_turns}",
                   body=body, attachments=attachments)
    return {"ok": bool(ok), "inputs": cnt_inputs, "turns": cnt_turns}

@app.post("/chat")
async def chat(req: ChatReq, request: Request):
    append_prompt_log(req.message, thread_id=req.thread_id or "")
    append_turn("user", req.message, thread_id=(req.thread_id or ""), actor=getattr(req, "actor", None))
    
    client_ip = request.client.host if request and request.client else "unknown"

    # --- A) 入力の先行チェック ---
    input_hits = local_high_risk(req.message)
    input_flagged, input_cats = mod_flag(req.message)
    should_alert_input = bool(input_hits) or input_flagged
    notify_input = any(lex.notify_enabled(h) for h in input_hits) or input_flagged

    if should_alert_input:
        rl_skip = _rate_limit_check(client_ip)

        reason = "入力内容が高リスクです。"
        if oa:
            try:
                review = oa.chat.completions.create(
                    model=REVIEW_MODEL,
                    messages=[{
                        "role": "user",
                        "content": f"次の入力が子ども向けに危険な理由を簡潔に述べてください: {req.message}"
                    }],
                    temperature=0
                )
                reason = review.choices[0].message.content
            except Exception:
                # もしGPT-5未開通や一時エラーなら 4o へフォールバック（任意）
                try:
                    review = oa.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role":"user","content": f"... {answer}"}],
                        temperature=0
                    )
                    reason = review.choices[0].message.content
                except Exception:
                    pass

        display = friendly_reply(input_hits) if RETURN_FRIENDLY else None
        log_event("input", input_hits, input_cats, req.message, client_ip)
        append_turn("assistant", display or "", thread_id=(req.thread_id or ""), actor="ai",
            meta={"kind":"shown", "blocked": True})

        if notify_input and not rl_skip:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            subj = f"[みまもりAI 警告][入力側] 不適切なリクエスト ({ts})"
            body = (
                f"日時: {ts}\n"
                f"スレッド: {getattr(req, 'thread_id', '')}\n"
                f"入力（全文）:\n{req.message}\n\n"
                f"出力（今回の表示テキスト）:\n{display or '(ブロックのため生成なし)'}\n"
                f"検知カテゴリ: {input_hits or input_cats}\n"
            )
            send_mail(subj, body)
            _rate_limit_record(client_ip, "input")

        return {"safe": False, "answer": display, "reason": reason,
                "alerts": {"side": "input", "hits": input_hits, "cats": input_cats}}

    # --- B) ここから通常生成（Ollama） ---
    system_text = "思考過程や<think>は出力せず、結論のみをやさしく回答してください。"
    if TEACHER_PROMPT:
        system_text += "\n\n" + TEACHER_PROMPT

    ollama_payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": req.message},
        ],
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as http:
            res = await http.post(f"{OLLAMA_HOST}/api/chat", json=ollama_payload)
            res.raise_for_status()
    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="Ollama応答の読取がタイムアウトしました。モデル初回起動/生成が長い可能性があります。")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Ollamaに接続できません。OLLAMA_HOSTやサービス稼働をご確認ください。")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"OllamaからのHTTPエラー: {e}")

    data = res.json()
    answer = ((data.get("message") or {}).get("content")) or data.get("response") or ""
    if not answer:
        raise HTTPException(status_code=502, detail="Ollamaの応答本文が空でした。")

    answer = sanitize(answer)

    append_turn("assistant", answer or "", thread_id=(req.thread_id or ""), actor="ai",
                meta={"kind":"raw", "blocked": False})

    # --- C) 出力の判定 ---
    output_flagged, output_cats = mod_flag(answer)
    output_hits = local_high_risk(answer)
    should_alert_output = bool(output_hits) or output_flagged

    if should_alert_output:
        rl_skip = _rate_limit_check(client_ip)

        if oa:
            try:
                review = oa.chat.completions.create(
                    model=REVIEW_MODEL,
                    messages=[{
                        "role": "user",
                        "content": f"次の入力が子ども向けに危険な理由を簡潔に述べてください: {req.message}"
                    }],
                    temperature=0
                )
                reason = review.choices[0].message.content
            except Exception:
                # もしGPT-5未開通や一時エラーなら 4o へフォールバック（任意）
                try:
                    review = oa.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role":"user","content": f"... {answer}"}],
                        temperature=0
                    )
                    reason = review.choices[0].message.content
                except Exception:
                    pass

        log_event("output", output_hits, output_cats, answer, client_ip)
        display = friendly_reply(output_hits) if RETURN_FRIENDLY else None
        append_turn("assistant", display or "", thread_id=(req.thread_id or ""), actor="ai",
                meta={"kind":"shown", "blocked": True})

        if not rl_skip:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            subj = f"[みまもりAI 警告][出力側] 不適切な返答 ({ts})"
            body = (
                f"日時: {ts}\n"
                f"スレッド: {getattr(req, 'thread_id', '')}\n"
                f"入力（全文）:\n{req.message}\n\n"
                f"出力（全文）:\n{answer or ''}\n"
                f"検知カテゴリ: {output_hits or output_cats}\n"
            )
            send_mail(subj, body)
            _rate_limit_record(client_ip, "output")

        reason = "生成結果が高リスクです。"
        return {"safe": False, "answer": display, "reason": reason,
                "alerts": {"side": "output", "hits": output_hits, "cats": output_cats}}

    return {"safe": True, "answer": answer}

@app.get("/debug/env")
def debug_env():
    keys = ["SMTP_HOST","SMTP_PORT","SMTP_USER","SMTP_PASS","SMTP_FROM","SMTP_TO","NIGHTLY_TOKEN","REVIEW_MODEL"]
    present = {k: bool(os.getenv(k)) for k in keys}
    # 値は出さない（漏洩防止）。長さだけ参考に返す
    length = {k: (len(os.getenv(k)) if os.getenv(k) else 0) for k in keys}
    return {
        "present": present,
        "length": length,
        "env_path": str(ENV_PATH),
        "cwd": os.getcwd(),
    }

@app.get("/debug/mail")
def debug_mail():
    ok = send_mail("【テスト】みまもりAIメール送信", "これはテストです。/debug/mail から送信")
    return {"ok": bool(ok)}
