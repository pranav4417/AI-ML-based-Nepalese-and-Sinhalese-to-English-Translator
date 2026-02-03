from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_DIR = "/Users/pranav/Music/nllb_offline"

app = Flask(__name__)

# -----------------------------
# Load model (ONCE)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

# -----------------------------
# Language map
# -----------------------------
LANG = {
    "si": "sin_Sinhala",
    "ne": "npi_Deva",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ta": "tam_Taml",
    "ml": "mal_Mlym",
    "hi": "hin_Deva",
    "en": "eng_Latn"
}

# -----------------------------
# Translation function
# -----------------------------
def translate(text, src, tgt):
    tokenizer.src_lang = LANG[src]

    inputs = tokenizer(text, return_tensors="pt").to(device)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(
        f"<{LANG[tgt]}>"
    )

    out = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=256
    )

    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def do_translate():
    data = request.json

    text = data["text"]
    src = data["src"]
    tgt = data["tgt"]

    result = translate(text, src, tgt)

    return jsonify({"translation": result})


if __name__ == "__main__":
    app.run(debug=True)