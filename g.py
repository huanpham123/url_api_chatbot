import os
import json
import logging
from typing import Optional
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests

# ====== CẤU HÌNH (để trực tiếp trong code) ======
API_KEY = "AIzaSyA22-Sh4sHm7AgB2EOmyrrti-jKQnaSxfE"   # <-- Thay bằng API key của bạn
DEFAULT_MODEL = "gemini-2.0-flash"
GEMINI_ENDPOINT_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# ====== logging ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini-proxy")

app = Flask(__name__, template_folder="templates")
CORS(app)

def try_parse_response_json(data: dict) -> Optional[str]:
    """Thử nhiều cách lấy text từ JSON trả về của Gemini/Generative API."""
    if not isinstance(data, dict):
        return None

    # 1) candidates -> content.parts[*].text
    candidates = data.get("candidates") or data.get("outputs") or data.get("choices")
    if isinstance(candidates, list) and len(candidates) > 0:
        first = candidates[0]
        if isinstance(first, dict):
            # try content.parts
            content = first.get("content") or first
            parts = None
            if isinstance(content, dict):
                parts = content.get("parts") or content.get("text") or None
            if isinstance(parts, list):
                texts = []
                for p in parts:
                    if isinstance(p, dict):
                        texts.append(p.get("text") or p.get("content") or "")
                    else:
                        texts.append(str(p))
                joined = "".join(texts).strip()
                if joined:
                    return joined
            # try direct keys in first
            for k in ("text", "output", "content", "message"):
                v = first.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

    # 2) top-level known fields
    for key in ("output_text", "response", "result", "output"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # 3) if any string anywhere, pick the longest (fallback)
    longest = ""
    def walk(obj):
        nonlocal longest
        if isinstance(obj, str):
            if len(obj) > len(longest):
                longest = obj
        elif isinstance(obj, dict):
            for val in obj.values():
                walk(val)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)
    walk(data)
    return longest.strip() if longest else None

def call_gemini(question: str, model: str = DEFAULT_MODEL, timeout: int = 15) -> Optional[str]:
    """
    Gọi Generative Language API với nhiều payload thử nghiệm.
    Trả text nếu thành công, ngược lại trả None (và in log chi tiết).
    """
    url = GEMINI_ENDPOINT_TMPL.format(model=model) + f"?key={API_KEY}"
    headers = {"Content-Type": "application/json; charset=utf-8"}

    payloads = [
        # payload kiểu bạn đã dùng ban đầu
        {"contents": [{"parts": [{"text": question}]}]},
        # payload 'prompt' phổ biến
        {"prompt": {"text": question}, "temperature": 0.2},
        # payload simple
        {"input": question},
        {"text": question},
    ]

    last_text = None
    for p in payloads:
        try:
            resp = requests.post(url, headers=headers, json=p, timeout=timeout)
        except Exception as e:
            logger.warning("Request failed (exception) for payload keys %s : %s", list(p.keys()), e)
            last_text = str(e)
            continue

        status = resp.status_code
        body = resp.text or ""
        logger.info("Tried payload keys %s => status %s", list(p.keys()), status)
        logger.debug("Response body (truncated): %s", body[:1000])
        last_text = body

        if status != 200:
            # in log chi tiết để bạn biết lý do
            logger.error("API returned non-200. status=%s body=%s", status, body[:2000])
            # thử payload khác
            continue

        # status == 200
        # thử parse JSON
        try:
            data = resp.json()
        except ValueError:
            text = body.strip()
            if text:
                return text
            else:
                continue

        answer = try_parse_response_json(data)
        if answer:
            return answer

        # fallback trả toàn bộ JSON (string)
        try:
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            continue

    logger.error("Tất cả payload đều thất bại. last_response (truncated): %s", (last_text or "")[:2000])
    return None

@app.route("/ask", methods=["GET", "POST"])
def ask():
    # Hỗ trợ GET ?q=... hoặc POST JSON {"q": "..."}
    if request.method == "GET":
        q = request.args.get("q", "").strip()
        model = request.args.get("model", DEFAULT_MODEL)
    else:
        body = request.get_json(silent=True) or {}
        q = (body.get("q") or "").strip()
        model = body.get("model") or DEFAULT_MODEL

    if not q:
        return jsonify({"ok": False, "error": "Missing question parameter 'q'"}), 400

    answer = call_gemini(q, model)
    if not answer:
        # in thêm gợi ý debug cho bạn
        return jsonify({"ok": False, "error": "Error from API. Check server logs for details (status/body)."}), 502

    return jsonify({"ok": True, "answer": answer}), 200

@app.route("/")
def index():
    return render_template("test.html")

if __name__ == "__main__":
    # chạy local, debug ON để bạn thấy log
    app.run(host="0.0.0.0", port=5000, debug=True)
