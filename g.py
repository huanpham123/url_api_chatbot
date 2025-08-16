import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ====== CONFIG ======
API_KEY = "AIzaSyA22-Sh4sHm7AgB2EOmyrrti-jKQnaSxfE"  # ⚠️ key của bạn
DEFAULT_MODEL = "gemini-2.0-flash"
GEMINI_ENDPOINT_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

app = Flask(__name__, template_folder="templates")
CORS(app)

def call_gemini(question: str, model: str = DEFAULT_MODEL):
    url = GEMINI_ENDPOINT_TMPL.format(model=model) + f"?key={API_KEY}"
    payload = {"contents": [{"parts": [{"text": question}]}]}
    resp = requests.post(url, json=payload)

    if resp.status_code != 200:
        return {"ok": False, "error": resp.text}

    data = resp.json()
    text = ""
    try:
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
    except Exception:
        pass

    return {"ok": True, "answer": text or "No response", "model": model}

@app.route("/ask", methods=["GET", "POST"])
def ask():
    if request.method == "GET":
        q = request.args.get("q", "").strip()
        model = request.args.get("model", DEFAULT_MODEL)
    else:
        body = request.get_json(silent=True) or {}
        q = body.get("question", "").strip()
        model = body.get("model", DEFAULT_MODEL)

    if not q:
        return jsonify({"ok": False, "error": "Missing question"}), 400

    result = call_gemini(q, model)
    return jsonify(result)

@app.route("/")
def index():
    return render_template("test.html")

if __name__ == "__main__":
    app.run(debug=True)
