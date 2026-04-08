# Somnium
### Oneirology-driven dream journaling · keyword analysis · genre classification · media recommendations

![Python](https://img.shields.io/badge/Python-3.x-blue) ![NLP](https://img.shields.io/badge/NLP-spaCy-green) ![License](https://img.shields.io/badge/License-MIT-purple)

---

## Overview

Somnium is a dream journaling platform built for lucid dreamers and anyone who wants to explore the patterns in their subconscious. Users log their dreams in natural language; the system analyses the text for recurring themes and keywords, categorises each dream into a narrative genre, and recommends movies, TV series, or books that resonate with the dream's emotional and thematic content.

The name *Somnium* is Latin for *dream*. The project draws on the field of **oneirology** — the scientific study of dreams — to bring structure and meaning to otherwise fleeting experiences.

---

## Features

- 📓 **Dream Journal** — Log and save dreams with timestamps, mood tags, and recurring character tracking
- 🔍 **Keyword Analysis** — NLP-based extraction of symbols, emotions, and recurring themes from dream entries
- 🎭 **Genre Classification** — Automatically categorises dreams: adventure, horror, romance, surreal, and more
- 🎬 **Media Recommendations** — Suggests films, series, or books that match the tone and themes of your dream

---

## How It Works
User logs dream → NLP keyword extraction → Genre classification → Media recommendation engine → Results displayed

Each dream entry is processed through a text analysis pipeline that identifies key symbols and emotional signals. These are mapped to genre labels using a classification model, which then queries a recommendation dataset to surface relevant media.

---

## Tech Stack

- **Frontend** — HTML + CSS + JavaScript (AI Mode + Offline fallback)
- **Backend** — Python, Flask REST API
- **NLP** — spaCy + NLTK pipeline
- **ML** — TF-IDF + LinearSVC genre classifier
- **Database** — SQLite

---

## Getting Started

Clone the repository and install dependencies:

```bash
git clone https://github.com/Abhishek2993/somnium.git
cd somnium/backend
pip install -r requirements.txt
```

Run the application (one command):

```bash
bash start.sh
```

Or manually:

```bash
python app.py
```

Open your browser at `http://localhost:5000`

---

## Project Structure
Somnium/
├── index.html          ← Frontend (AI Mode + Offline fallback)
├── backend/
│   ├── app.py          ← Flask REST API
│   ├── nlp.py          ← spaCy + NLTK pipeline
│   ├── ml.py           ← TF-IDF + LinearSVC genre classifier
│   ├── db.py           ← SQLite layer
│   ├── requirements.txt
│   └── start.sh        ← One-command launcher
└── README.md

---
