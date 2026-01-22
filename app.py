from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

generator = pipeline(
    "text-generation",
    model="bigscience/bloom-560m"
)

class Prompt(BaseModel):
    mood: str
    topic: str

@app.get("/")
def home():
    return {"message": "Punjabi Lyrics AI is running"}

@app.post("/generate")
def generate_lyrics(data: Prompt):
    prompt = f"""
Write Punjabi song lyrics in Punjabi language only.
Do not use any English or Hinglish words.
Use modern Punjabi songwriting style.

Mood: {data.mood}
Topic: {data.topic}

Write 8 to 12 lines of lyrics.
"""

    result = generator(
        prompt,
        max_length=200,
        do_sample=True,
        temperature=0.9
    )

    return {"lyrics": result[0]["generated_text"]}
