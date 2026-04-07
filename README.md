# 🌍 GeoAI Assistant

A local AI study assistant for Geodesy and Geoinformation Science.

This project uses Retrieval-Augmented Generation (RAG) over user-provided study materials and provides an interactive Streamlit interface for learning, revision, and exam preparation.

---

## ✨ Features

- 📚 Course-aware retrieval (multi-course support)
- 🧠 Hybrid AI mode (local materials + general knowledge)
- 🎓 Professor-style explanations
- 🧪 Study modes:
  - Ask a question
  - Summarize lecture
  - Generate exam questions
- 🧠 Beginner mode (simplified explanations)
- 📂 Source tracking with direct PDF access
- 💬 Chat interface with conversation history
- 🖥️ Fully local (no external APIs required except local LLM)

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Ollama (local LLM runtime)
- Gemma (or other local models)
- NumPy

---

## 📂 Using Your Own Data

This project does not include any study materials.
To use the assistant, you need to provide your own PDF files.

---

## 🚀 How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/ingest.py
streamlit run app.py
```

---

<img width="1499" height="904" alt="Screenshot 2026-04-07 at 15 05 46" src="https://github.com/user-attachments/assets/e0bd9fcc-a914-4590-874c-19bdad8af507" />
