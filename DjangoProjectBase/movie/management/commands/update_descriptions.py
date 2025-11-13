from django.core.management.base import BaseCommand
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from movie.models import Movie
from openai_connect import get_completion

class Command(BaseCommand):
    help = 'Actualiza la descripción de una película usando IA (GPT)'

    def handle(self, *args, **options):
        movies = Movie.objects.all()
        for movie in movies:
            instruction = "Actualiza la descripción"
            prompt = f"{instruction} '{movie.description}' de la película '{movie.title}'"
            response = get_completion(prompt)
            movie.description = response
            movie.save()
            self.stdout.write(self.style.SUCCESS(f"Descripción actualizada para: {movie.title}"))
            break  # Solo actualiza una película por ejecución
