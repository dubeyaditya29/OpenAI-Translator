from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

translator = pipeline("translation", model="google-t5/t5-base")

# Supported language mappings
language_codes = {
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Pidgin": "pcm"  # This may not be supported directly; we can refine it later
}
class TranslationRequest(BaseModel):
    text: str
    language: str

@app.post("/translate")
async def translate(request: TranslationRequest):
    lang_code = language_codes.get(request.language)
    
    if not lang_code:
        return {"error": f"Translation to '{request.language}' is not supported yet."}

    prompt = f"translate English to {lang_code}: {request.text}"
    result = translator(prompt, max_length=200)

    return {"translatedText": result[0]['translation_text']}
