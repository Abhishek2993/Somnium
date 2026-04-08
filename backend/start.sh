#!/bin/bash
# start.sh — Launch the Somnium backend server
# Usage: ./start.sh  (from the project root OR backend/ folder)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
  echo "⏳ Creating virtual environment..."
  python3 -m venv venv
  source venv/bin/activate
  echo "⏳ Installing dependencies..."
  pip install -q flask flask-cors spacy nltk scikit-learn joblib
  python -m spacy download en_core_web_sm
else
  source venv/bin/activate
fi

echo "🌙 Starting Somnium backend on http://localhost:5000"
python app.py
