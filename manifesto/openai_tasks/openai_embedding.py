import os

from tenacity import retry, wait_random_exponential, stop_after_attempt
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


def try_embedding():
    embedding = openai.Embedding.create(
        input='string of text data', model="text-embedding-ada-002"
    )["data"][0]["embedding"]
    print(len(embedding))


def retries_embeddings():
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
        return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]
    embedding = get_embedding("string of text data", model="text-embedding-ada-002")
    print(len(embedding))