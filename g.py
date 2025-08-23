import json
import logging
from typing import Optional
from flask import Flask, request, jsonify, render_template
import requests
import os

# ====== CẤU HÌNH (API KEY để trong code như yêu cầu) ======
API_KEY = "#"   # <-- API key đã được chèn ở đây
DEFAULT_MODEL = "gemini-2.0-flash"
GEMINI_ENDPOINT_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# ====== Logging ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini-proxy")

app = Flask(__name__, template_folder="templates")

def save_last_response(resp: requests.Response):
    """Lưu response raw + headers vào file để debug."""
    try:
        with open("last_gemini_response.txt", "w", encoding="utf-8") as f:
            f.write(f"STATUS: {resp.status_code}\n\n")
            f.write("HEADERS:\n")
            for k, v in resp.headers.items():
                f.write(f"{k}: {v}\n")
            f.write("\nBODY:\n")
            f.write(resp.text)
    except Exception as e:
        logger.warning("Không thể ghi last_gemini_response.txt: %s", e)

def try_parse_response_json(data: dict) -> Optional[str]:
    """Cố gắng lấy text trả về từ JSON response (nhiều dạng cấu trúc có thể xuất hiện)."""
    if not isinstance(data, dict):
        return None

    # Thử cấu trúc: candidates -> content -> parts -> text
    candidates = data.get("candidates") or data.get("outputs") or data.get("choices")
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, dict):
            content = first.get("content") or {}
            parts = content.get("parts") or []
            if isinstance(parts, list) and parts:
                texts = []
                for p in parts:
                    if isinstance(p, dict):
                        t = p.get("text") or p.get("content") or ""
                    else:
                        t = str(p)
                    texts.append(t)
                joined = "".join(texts).strip()
                if joined:
                    return joined
            # fallback trong candidate
            for k in ("text", "output", "message"):
                v = first.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

    # Thử các key top-level
    for key in ("output_text", "response", "result", "output"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Lấy chuỗi dài nhất trong json (fallback)
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

def call_gemini(question: str, model: str = DEFAULT_MODEL, timeout: int = 20) -> Optional[str]:
    """
    Gọi Google Generative Language API theo ví dụ curl (header X-goog-api-key, payload contents/parts)
    Trả về text nếu thành công, ngược lại None.
    """
    url = GEMINI_ENDPOINT_TMPL.format(model=model)
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": API_KEY
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": question}
                ]
            }
        ]
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except Exception as e:
        logger.error("Request exception khi gọi Gemini: %s", e)
        return None

    # Lưu/dump response cho debug
    save_last_response(resp)
    logger.info("Gọi Gemini -> status=%s", resp.status_code)

    if resp.status_code != 200:
        logger.error("Gemini trả lỗi status=%s, body (truncated): %s", resp.status_code, (resp.text or "")[:1500])
        return None

    # Nếu 200, parse JSON hoặc fallback về text
    try:
        j = resp.json()
    except ValueError:
        text = resp.text.strip()
        return text if text else None

    answer = try_parse_response_json(j)
    if answer:
        return answer

    # fallback: return whole JSON string
    try:
        return json.dumps(j, ensure_ascii=False)
    except Exception:
        return None

@app.route("/ask", methods=["GET", "POST"])
def ask():
    # hỗ trợ GET ?q=... và POST JSON {"q":"..."}
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
        return jsonify({"ok": False, "error": "Error from API. See server logs and last_gemini_response.txt"}), 502

    return jsonify({"ok": True, "answer": answer}), 200

@app.route("/")
def index():
    return render_template("test.html")

if __name__ == "__main__":
    # nếu muốn thay port, set env PORT trước khi chạy
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

