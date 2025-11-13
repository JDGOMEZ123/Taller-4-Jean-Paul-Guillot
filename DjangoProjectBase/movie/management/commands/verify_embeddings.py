import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = "Verifica el tamaño de los embeddings de cada película"

    def handle(self, *args, **kwargs):
        movies = Movie.objects.all()
        for movie in movies:
            if not movie.emb:
                print(f"{movie.title}: Sin embedding")
                continue
            emb_arr = np.frombuffer(movie.emb, dtype=np.float32)
            print(f"{movie.title}: embedding size = {emb_arr.shape[0]}")
