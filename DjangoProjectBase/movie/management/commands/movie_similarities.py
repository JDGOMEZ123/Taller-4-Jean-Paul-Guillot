import os
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv

class Command(BaseCommand):
    help = "Compare two movies and optionally a prompt using OpenAI embeddings"

    def handle(self, *args, **kwargs):
        # ✅ Load OpenAI API key
        load_dotenv('../openAI.env')
        client = OpenAI(api_key=os.environ.get('openai_apikey'))

        # ✅ Change these titles for any movies you want to compare
        movie1 = Movie.objects.get(title="A Trip to the Moon")
        movie2 = Movie.objects.get(title="Cinderella")

        def get_embedding(text):
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            return np.array(response.data[0].embedding, dtype=np.float32)

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # ✅ Generate embeddings of both movies
        emb1 = get_embedding(movie1.description)
        emb2 = get_embedding(movie2.description)

        # ✅ Compute similarity between movies
        similarity = cosine_similarity(emb1, emb2)
        self.stdout.write(f"\U0001F3AC Similaridad entre '{movie1.title}' y '{movie2.title}': {similarity:.4f}")

        # ✅ Optional: Compare against a prompt
        prompt = "Películas de ciencia ficción y romance"
        prompt_emb = get_embedding(prompt)
        sim1 = cosine_similarity(emb1, prompt_emb)
        sim2 = cosine_similarity(emb2, prompt_emb)
        self.stdout.write(f"\U0001F4AC Similaridad de '{movie1.title}' con el prompt: {sim1:.4f}")
        self.stdout.write(f"\U0001F4AC Similaridad de '{movie2.title}' con el prompt: {sim2:.4f}")
