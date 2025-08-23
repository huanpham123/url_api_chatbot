import requests
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS

# ====== CONFIG ======
API_KEY = "AIzaSyA22-Sh4sHm7AgB2EOmyrrti-jKQnaSxfE"   # 👉 Thay bằng API key thật của bạn
DEFAULT_MODEL = "gemini-2.0-flash"
GEMINI_ENDPOINT_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

app = Flask(__name__, template_folder="templates")
CORS(app)

def call_gemini(question: str, model: str = DEFAULT_MODEL):
    url = GEMINI_ENDPOINT_TMPL.format(model=model) + f"?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": question}]}
        ]
    }

    resp = requests.post(url, headers=headers, json=payload)

    if resp.status_code != 200:
        print("❌ API Error:", resp.status_code, resp.text)
        return None

    data = resp.json()
    try:
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)
    except Exception as e:
        print("❌ Parse error:", e)

    return None

@app.route("/ask", methods=["GET"])
def ask():
    q = request.args.get("q", "").strip()
    model = request.args.get("model", DEFAULT_MODEL)

    if not q:
        return jsonify({"ok": False, "error": "Missing question parameter 'q'"}), 400

    answer = call_gemini(q, model)
    if not answer:
        return jsonify({"ok": False, "error": "Error from API"}), 502

    return jsonify({"ok": True, "answer": answer})

@app.route("/")
def index():
    return render_template("test.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
