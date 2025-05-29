import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def get_embedding(text, model="voyage-3.5"):
    response = anthropic.embeddings.create(
        model=model,
        input=text
    )
    return response.embedding 