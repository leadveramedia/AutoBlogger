import json
import os
from .config import _BASE_DIR

OUTFIT_HISTORY_FILE = os.path.join(_BASE_DIR, 'outfit_history.json')
MAX_HISTORY = 10

def load_outfit_history():
    """Load the last 10 outfit descriptions."""
    if not os.path.exists(OUTFIT_HISTORY_FILE):
        return []

    try:
        with open(OUTFIT_HISTORY_FILE, 'r') as f:
            history = json.load(f)
            return history[-MAX_HISTORY:]  # Keep only last 10
    except:
        return []

def save_outfit(outfit_description):
    """Append a new outfit to history, keeping only last 10."""
    history = load_outfit_history()
    history.append(outfit_description)
    history = history[-MAX_HISTORY:]  # Trim to last 10

    os.makedirs(os.path.dirname(OUTFIT_HISTORY_FILE), exist_ok=True)
    with open(OUTFIT_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
