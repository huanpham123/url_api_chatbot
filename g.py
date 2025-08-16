import requests
from flask import Flask, request, Response, render_template
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
        return None

    data = resp.json()
    try:
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)
    except Exception:
        return None

    return None

@app.route("/ask", methods=["GET"])
def ask():
    q = request.args.get("q", "").strip()
    model = request.args.get("model", DEFAULT_MODEL)

    if not q:
        return Response("⚠️ Missing question", mimetype="text/plain", status=400)

    answer = call_gemini(q, model)
    if not answer:
        return Response("⚠️ Error from API", mimetype="text/plain", status=500)

    # 🔑 chỉ trả về text
    return Response(answer, mimetype="text/plain")

@app.route("/")
def index():
    return render_template("test.html")

if __name__ == "__main__":
    app.run(debug=True)
