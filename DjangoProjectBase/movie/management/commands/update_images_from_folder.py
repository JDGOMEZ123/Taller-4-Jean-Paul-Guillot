import os
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = "Actualiza el campo image de las películas usando las imágenes en media/movie/images/"

    def handle(self, *args, **kwargs):
        images_folder = os.path.join('media', 'movie', 'images')
        updated_count = 0
        for movie in Movie.objects.all():
            image_filename = f"m_{movie.title}.png"
            image_path = os.path.join(images_folder, image_filename)
            if os.path.exists(image_path):
                movie.image = os.path.join('movie/images', image_filename)
                movie.save()
                updated_count += 1
                self.stdout.write(self.style.SUCCESS(f"Updated image for: {movie.title}"))
            else:
                self.stdout.write(self.style.WARNING(f"Image not found for: {movie.title}"))
        self.stdout.write(self.style.SUCCESS(f"Finished updating {updated_count} movies from folder."))
