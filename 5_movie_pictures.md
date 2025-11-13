# 🎨 Generación de imágenes por película usando OpenAI y actualización en la base de datos

## ✅ Objetivo
Generar imágenes personalizadas por cada película usando la API de OpenAI y actualizar la base de datos con la imagen correspondiente.

---

## 📌 1. ¿Qué vas a hacer en esta parte?
✅ Conectarte a la API de OpenAI  
✅ Generar una imagen para una película usando una función auxiliar  
✅ Descargar y almacenar la imagen en el proyecto  
✅ Actualizar la base de datos con la nueva imagen generada

⚠️ Por ahora, el comando está configurado para generar y actualizar **solo la primera película**.

✅ Esto es intencional para **ahorrar recursos y evitar costos de API**.

✅ **Las imágenes de todas las películas ya fueron generadas** y se entregan en una carpeta lista para usar.

🚫 **No debes quitar el `break`.**

El código lo puedes encontrar en el archivo [update_images.py](aux_files/update_images.py)

---

## 📌 2. Descripción del comando `update_images`
El comando se debe ubicar en:
```
movie/management/commands/update_images.py
```

### ✅ ¿Qué hace cada parte?

```python
# ✅ Load environment variables from the .env file
load_dotenv('../openAI.env')

# ✅ Initialize the OpenAI client with the API key
client = OpenAI(
    api_key=os.environ.get('openai_apikey'),
    )
```
- Se conecta a la API de OpenAI cargando la clave desde el archivo `.env`

---

```python
images_folder = 'media/movie/images/'
os.makedirs(images_folder, exist_ok=True)
```
- Crea la carpeta de imágenes si no existe.

---

```python
movies = Movie.objects.all()
self.stdout.write(f"Found {movies.count()} movies")
```
- Consulta la base de datos y trae todas las películas.

---

### ✅ Función auxiliar que hace todo el trabajo con la API:
```python
def generate_and_download_image(self, client, movie_title, save_folder):
    prompt = f"Movie poster of {movie_title}"
    response = client.images.generate(
        model="dall-e-2",
        prompt=prompt,
        size="256x256",
        quality="standard",
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
```
✅ Genera la imagen en OpenAI, descarga la imagen y la almacena en la carpeta del proyecto.

---

### ✅ Dentro del ciclo:
```python
for movie in movies:
    image_relative_path = self.generate_and_download_image(client, movie.title, images_folder)
    movie.image = image_relative_path
    movie.save()
    self.stdout.write(self.style.SUCCESS(f"Saved and updated image for: {movie.title}"))
    break
```
- Llama la función auxiliar
- Actualiza la imagen en la base de datos
- Detiene la ejecución en la primera película (luego debes quitar el `break`)

---

## 📌 3. Ejecuta el comando
```bash
python manage.py update_images
```

✅ Verás mensajes indicando qué película se procesó y si la imagen fue descargada y almacenada.

---

## Actividad: 
✅ Ejecuta el comando.

✅ Levanta el servidor de Django:
```bash
python manage.py runserver
```

✅ Accede a la vista de las películas y verifica que la imagen de la primera película corresponde a la generada.

✅ **Toma una captura de pantalla** y guárdala.

<div align="center">
  <img src="imgs/updatemovies1.png" alt="Imagenes.txt">
</div>


✅ Esta es la evidencia de tu proceso funcionando.

---

## 📌 5. Actividad: Cargar imágenes desde la carpeta entregada

### ✅ ¿Qué hicmos por ti?
El proceso completo de generación de imágenes se ejecutó una sola vez y se entrega la carpeta con todas las imágenes generadas en este [link](https://eafit-my.sharepoint.com/:u:/g/personal/jdmartinev_eafit_edu_co/EZWTYwpkbHhHjIEZfkgc_mwBuknZR5cizHhIhRuDo9qrnQ?e=u2ynMP) .

👉 Las imágenes se deben ubicar en la carpeta:
```
media/movie/images/
```
con nombres como:
```
m_NOMBRE_PELICULA.png
```

---

## 📌 ¿Qué debes hacer?
✅ Crear un nuevo comando llamado:
```
movie/management/commands/update_images_from_folder.py
```

Este comando debe:
- Recorrer las películas en la base de datos
- Asignar la imagen correspondiente de la carpeta `media/movie/images/`
- Actualizar la base de datos con la ruta de la imagen

### 📥 Ejecuta:
```bash
python manage.py update_images_from_folder
```

---

## 📌 Resultado esperado
✅ Al terminar, la base de datos tendrá asignada la imagen correcta para cada película.

✅ Al visualizar el sitio, verás cada película con su respectiva imagen generada por la API.

📸 Entregable: Debes adjuntar una captura de pantalla donde se visualice la base de datos o la interfaz de la aplicación mostrando las imágenes generadas y actualizadas para las películas.

---

## 📌 Nota final
No es necesario generar nuevamente las imágenes ni quitar el `break`.  
**Solo debes cargar las imágenes entregadas y actualizar la base de datos.**

---

