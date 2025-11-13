import os
from dotenv import load_dotenv
from openai import OpenAI
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Update movie images using OpenAI API"

    def handle(self, *args, **kwargs):
        # Load environment variables from the .env file
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), '.env'))
        # Initialize the OpenAI client with the API key
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

        # Crear la carpeta de imágenes si no existe
        images_folder = os.path.join('media', 'movie', 'images')
        os.makedirs(images_folder, exist_ok=True)

        # Consultar la base de datos y traer todas las películas
        from movie.models import Movie
        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies")

        # Función auxiliar para generar y descargar imagen
        import requests
        def generate_and_download_image(client, movie_title, save_folder):
            prompt = f"Movie poster of {movie_title}"
            response = client.images.generate(
                model="dall-e-2",
                prompt=prompt,
                size="256x256",
                n=1,
            )
            image_url = response.data[0].url
            image_filename = f"m_{movie_title}.png"
            image_path_full = os.path.join(save_folder, image_filename)
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            with open(image_path_full, 'wb') as f:
                f.write(image_response.content)
            return os.path.join('movie/images', image_filename)

        # Recorrer las películas y actualizar la imagen de la primera
        for movie in movies:
            image_relative_path = generate_and_download_image(client, movie.title, images_folder)
            movie.image = image_relative_path
            movie.save()
            self.stdout.write(self.style.SUCCESS(f"Saved and updated image for: {movie.title}"))
            break
