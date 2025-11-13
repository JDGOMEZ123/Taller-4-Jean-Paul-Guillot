import os
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv


class Command(BaseCommand):
    help = "Regenera y almacena embeddings para todas las películas usando OpenAI (tamaño 1536)"

    def handle(self, *args, **kwargs):
        # Cargar la API Key
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'openAI.env'))
        client = OpenAI(api_key=os.environ.get('openai_apikey'))

        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies in the database")

        for movie in movies:
            try:
                # Generar embedding usando la descripción
                response = client.embeddings.create(
                    input=[movie.description],
                    model="text-embedding-3-small"
                )
                embedding_array = np.array(response.data[0].embedding, dtype=np.float32)
                if embedding_array.shape[0] != 1536:
                    raise ValueError(f"Embedding size for '{movie.title}' is {embedding_array.shape[0]}, expected 1536.")
                # Guardar como binario en el campo emb
                movie.emb = embedding_array.tobytes()
                movie.save()
                self.stdout.write(f"👌 Embedding stored for: {movie.title}")
            except Exception as e:
                self.stderr.write(f"❌ Error storing embedding for {movie.title}: {e}")

        self.stdout.write("🌟 Finished generating embeddings for all movies")


