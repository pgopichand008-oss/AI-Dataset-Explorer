# FILE: config.py

import os
from google import genai


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    return genai.Client(
        api_key=api_key
    )


def get_gemini_model():
    return "gemini-3.6-flash"
