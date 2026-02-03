# ⚡ Offline Translator (NLLB Local)

**100% offline multilingual translator** running entirely in your browser + local backend.

Built using a **custom local setup** derived from Meta's **NLLB-200** (No Language Left Behind) model — specifically the distilled 600M variant.

Perfect for:

- Privacy-sensitive environments
- Low/no internet areas
- Quick language experiments & demos
- Supporting South Asian languages offline



## ✨ Features

- **Completely offline** after model download
- Real-time translation in browser
- Supports translation among 8 languages
- Clean, minimal web UI
- FastAPI / Flask lightweight local server
- No cloud APIs, no tracking, no telemetry

## 🌍 Supported Languages

| Language     | Code     | Script     | NLLB token       |
|--------------|----------|------------|------------------|
| English      | en       | Latin      | eng_Latn         |
| Hindi        | hi       | Devanagari | hin_Deva         |
| Sinhala      | si       | Sinhala    | sin_Sinhala      |
| Nepali       | ne       | Devanagari | npi_Deva         |
| Telugu       | te       | Telugu     | tel_Telu         |
| Kannada      | kn       | Kannada    | kan_Knda         |
| Tamil        | ta       | Tamil      | tam_Taml         |
| Malayalam    | ml       | Malayalam  | mal_Mlym         |

→ All combinations are supported (e.g. Tamil → Nepali, Sinhala → Kannada, English ↔ Malayalam, etc.)

## 🧠 Model

**Base model**: [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)

- ~600 million parameters (distilled version — good speed vs quality balance)
- Trained on **200 languages** with strong emphasis on low-resource languages
- Uses **FLORES-200** language codes & tokenization scheme

This project uses the official weights downloaded from Hugging Face and runs inference locally.

##🎯 Purpose

This project is intended for:
	•	multilingual NLP demos
	•	offline translation systems
	•	academic / research experiments
	•	local language technology tools
