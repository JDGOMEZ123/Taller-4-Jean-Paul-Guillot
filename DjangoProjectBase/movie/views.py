from django.shortcuts import render
from django.http import HttpResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt

import numpy as np
import os
import sys
import io
import urllib, base64
import matplotlib.pyplot as plt
import matplotlib

from openai import OpenAI
from dotenv import load_dotenv

from .models import Movie

# Asegurar que se pueda importar el conector de OpenAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from openai_connect import get_completion


# Vista para recomendar películas usando embeddings
class RecommendMovieView(View):
    def get(self, request):
        return render(request, "recommend.html")

    def post(self, request):
        prompt = request.POST.get("prompt", "")
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'openAI.env'))
        client = OpenAI(api_key=os.environ.get('openai_apikey'))

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # Generar embedding del prompt
        response = client.embeddings.create(
            input=[prompt],
            model="text-embedding-3-small"
        )
        prompt_emb = np.array(response.data[0].embedding, dtype=np.float32)

        # Buscar la película más similar
        best_movie = None
        max_similarity = -1
        for movie in Movie.objects.all():
            if not movie.emb:
                continue
            movie_emb = np.frombuffer(movie.emb, dtype=np.float32)
            if movie_emb.shape[0] != 1536:
                continue
            similarity = cosine_similarity(prompt_emb, movie_emb)
            if similarity > max_similarity:
                max_similarity = similarity
                best_movie = movie

        return render(request, "recommend.html", {
            "prompt": prompt,
            "best_movie": best_movie,
            "similarity": f"{max_similarity:.4f}"
        })


# Página principal
def home(request):
    searchTerm = request.GET.get('searchMovie')
    if searchTerm:
        movies = Movie.objects.filter(title__icontains=searchTerm)
    else:
        movies = Movie.objects.all()
    return render(request, 'home.html', {'searchTerm': searchTerm, 'movies': movies})


def about(request):
    return render(request, 'about.html')


def signup(request):
    email = request.GET.get('email')
    return render(request, 'signup.html', {'email': email})


# Estadísticas
def statistics_view(request):
    matplotlib.use('Agg')

    all_movies = Movie.objects.all()

    # Películas por año
    movie_counts_by_year = {}
    for movie in all_movies:
        year = movie.year if movie.year else "None"
        movie_counts_by_year[year] = movie_counts_by_year.get(year, 0) + 1
    year_graphic = generate_bar_chart(movie_counts_by_year, 'Year', 'Number of movies')

    # Películas por género
    movie_counts_by_genre = {}
    for movie in all_movies:
        genre = movie.genre.split(',')[0].strip() if movie.genre else "None"
        movie_counts_by_genre[genre] = movie_counts_by_genre.get(genre, 0) + 1
    genre_graphic = generate_bar_chart(movie_counts_by_genre, 'Genre', 'Number of movies')

    return render(request, 'statistics.html', {
        'year_graphic': year_graphic,
        'genre_graphic': genre_graphic
    })


def generate_bar_chart(data, xlabel, ylabel):
    keys = [str(key) for key in data.keys()]
    plt.bar(keys, data.values())
    plt.title('Movies Distribution')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic


# Actualiza descripciones con IA
@csrf_exempt
def update_movie_descriptions(request):
    if request.method == "POST":
        movies = Movie.objects.all()
        for movie in movies:
            instruction = "Mejora la siguiente descripción de película: "
            prompt = f"{instruction} '{movie.description}'"
            response = get_completion(prompt)
            movie.description = response
            movie.save()
            break  # Solo una película por ejecución
        return HttpResponse("Descripción actualizada para una película.")
    return HttpResponse("Método no permitido.", status=405)
