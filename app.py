from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer
import ctranslate2
import time
import json
import os
import io
import pypdf

app = FastAPI()

CT2_MODEL_DIR = "./models/nllb-ct2-int8"
HF_MODEL_DIR  = "./models/models--facebook--nllb-200-distilled-600M"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR, local_files_only=True)

# CTranslate2 Translator — uses all CPU cores via OpenMP, int8 quantized
print(f"Loading CTranslate2 model from {CT2_MODEL_DIR}...")
num_threads = os.cpu_count() or 4
translator = ctranslate2.Translator(
    CT2_MODEL_DIR,
    device="cpu",
    inter_threads=1,       # one request at a time (single user)
    intra_threads=num_threads,  # use all cores for matrix ops
    compute_type="int8",
)
print(f"Model ready ({num_threads} threads).")

LANGS = {
    "si": "sin_Sinhala",
    "ne": "npi_Deva",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ta": "tam_Taml",
    "ml": "mal_Mlym",
    "hi": "hin_Deva",
    "en": "eng_Latn",
}

class TranslateRequest(BaseModel):
    text: str
    src: str
    tgt: str

def tokenize_for_ct2(text: str, src_lang: str):
    """Tokenize text and prepare source tokens for CTranslate2."""
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors=None, truncation=True, max_length=400)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    return tokens

def generate_stream(text: str, src: str, tgt: str):
    start_time = time.time()

    src_lang = LANGS.get(src, "eng_Latn")
    tgt_lang = LANGS.get(tgt, "hin_Deva")

    source_tokens = tokenize_for_ct2(text, src_lang)

    results = translator.translate_batch(
        [source_tokens],
        target_prefix=[[tgt_lang]],
        beam_size=1,
        max_decoding_length=min(len(text.split()) * 4, 256),
        max_batch_size=1,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
        return_alternatives=False,
    )

    output_tokens = results[0].hypotheses[0]
    # First token is always the target lang prefix — drop it
    output_tokens = output_tokens[1:] if output_tokens else output_tokens

    # Correct decode: tokens → IDs → decode() handles SentencePiece ▁ spaces properly
    output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    full_translation = tokenizer.decode(output_ids, skip_special_tokens=True)
    latency = round((time.time() - start_time) * 1000, 2)

    words = full_translation.split(" ")
    for i, word in enumerate(words):
        token_text = word if i == 0 else " " + word
        yield f"data: {json.dumps({'token': token_text})}\n\n"

    yield f"data: {json.dumps({'done': True, 'latency_ms': latency, 'device': f'cpu-int8 ({num_threads}t)'})}\n\n"

@app.post("/translate/stream")
def translate_stream(req: TranslateRequest):
    return StreamingResponse(
        generate_stream(req.text, req.src, req.tgt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/translate")
def translate(req: TranslateRequest):
    start_time = time.time()
    src_lang = LANGS.get(req.src, "eng_Latn")
    tgt_lang = LANGS.get(req.tgt, "hin_Deva")
    source_tokens = tokenize_for_ct2(req.text, src_lang)
    results = translator.translate_batch(
        [source_tokens],
        target_prefix=[[tgt_lang]],
        beam_size=1,
        max_decoding_length=min(len(req.text.split()) * 4, 256),
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
    )
    output_tokens = results[0].hypotheses[0]
    # First token is always the target lang prefix — drop it
    output_tokens = output_tokens[1:] if output_tokens else output_tokens
    # Correct decode: tokens → IDs → decode() handles SentencePiece ▁ spaces properly
    output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    translation = tokenizer.decode(output_ids, skip_special_tokens=True)
    latency = (time.time() - start_time) * 1000
    return {"translation": translation, "latency_ms": round(latency, 2), "device": f"cpu-int8 ({num_threads}t)"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".txt", ".pdf")):
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported")

    content = await file.read()
    extracted_text = ""

    if file.filename.endswith(".txt"):
        extracted_text = content.decode("utf-8")
    elif file.filename.endswith(".pdf"):
        try:
            pdf_reader = pypdf.PdfReader(io.BytesIO(content))
            for page in pdf_reader.pages:
                extracted_text += page.extract_text() + "\n"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read PDF: {str(e)}")

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the file")

    return JSONResponse({"text": extracted_text.strip()})

app.mount("/", StaticFiles(directory="static", html=True), name="static")
