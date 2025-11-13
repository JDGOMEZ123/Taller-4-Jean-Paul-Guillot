# 🎯 Similitud de Películas usando Embeddings y OpenAI

## ✅ Objetivo
Crear un sistema de recomendación de películas utilizando **embeddings generados por OpenAI** y calcular la similitud entre las películas.

---

✅ Conectarte a la API de OpenAI  
✅ Generar un embedding para representar la descripción de cada película  
✅ Comparar embeddings utilizando la similitud del coseno

---

# 🎯 1. Introducción - ¿Qué es un Embedding y la Similitud de Coseno?

## ✅ ¿Qué es un Embedding?
Un **embedding** es una representación numérica de un texto o dato en un espacio vectorial. En nuestro caso:
- La **descripción de una película** se transforma en un vector de números
- Cada número representa una característica semántica de la descripción
- Películas con descripciones similares tendrán vectores (embeddings) cercanos

👉 Estos embeddings son generados por un modelo de OpenAI (`text-embedding-3-small`).

![embeddings](imgs/sr1.svg)

---

## ✅ ¿Qué es la Similitud de Coseno?
La **similitud de coseno** mide el ángulo entre dos vectores:
- Valor cercano a 1 → descripciones muy parecidas
- Valor cercano a 0 → descripciones diferentes

### ✅ Fórmula:
```
sim(a, b) = (a · b) / (||a|| * ||b||)
```
Donde:
- `a · b` → producto punto
- `||a||` → norma o magnitud del vector

---

## 📌 2. ¿Qué haremos a continuación?

Para comprender cómo funciona la generación de **embeddings** y el cálculo de **similitud**, crearemos un **comando de Django** que:

✅ Seleccionará **dos películas** desde la base de datos  
✅ Generará el **embedding de la descripción** de cada una usando la API de OpenAI  
✅ Calculará la **similitud de coseno** entre ambas películas

De esta forma, podrás observar cómo el sistema mide qué tan parecidas son dos películas a partir de sus descripciones.

---
## 📌 3. Comparación contra un prompt

Además, para entender mejor el funcionamiento del sistema de recomendación, el comando también permitirá:

✅ Definir un **prompt o descripción cualquiera** (por ejemplo: "película sobre la Segunda Guerra Mundial")  
✅ Generar el **embedding del prompt** usando la API de OpenAI  
✅ Calcular la **similitud de coseno** entre el embedding del prompt y el embedding de una película seleccionada

Esto permitirá observar cómo el sistema puede recomendar una película basada en la similitud temática con un texto de entrada libre.

---
## 📌 4. Explicación del Código
### ✅ Conectarse a la API de OpenAI y cargar las películas
```python
load_dotenv('openAI.env')
client = OpenAI(api_key=os.environ.get('openai_apikey'))

movie1 = Movie.objects.get(title="La lista de Schindler")
movie2 = Movie.objects.get(title="El club de la pelea")
```
- Carga la API Key
- Selecciona las películas desde la base de datos por su título

---

### ✅ Funciones principales

#### Obtener el embedding de cualquier texto o prompt
```python
def get_embedding(text):
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return np.array(response.data[0].embedding, dtype=np.float32)
```
- Envía la descripción a OpenAI
- Recibe el embedding como un vector numérico

#### Calcular la similitud de coseno
```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```
- Calcula qué tan parecidas son dos películas o una película y un prompt

---

### ✅ Cómo se usan las funciones en el comando

#### Calcular la similitud entre dos películas
```python
emb1 = get_embedding(movie1.description)
emb2 = get_embedding(movie2.description)

similarity = cosine_similarity(emb1, emb2)
self.stdout.write(f"🎬 {movie1.title} vs {movie2.title}: {similarity:.4f}")
```
- Genera los embeddings
- Calcula la similitud entre las dos descripciones
- Imprime el resultado

#### Calcular la similitud entre un prompt y las películas
```python
prompt = "película sobre la Segunda Guerra Mundial"
prompt_emb = get_embedding(prompt)

sim_prompt_movie1 = cosine_similarity(prompt_emb, emb1)
sim_prompt_movie2 = cosine_similarity(prompt_emb, emb2)

self.stdout.write(f"📝 Similitud prompt vs '{movie1.title}': {sim_prompt_movie1:.4f}")
self.stdout.write(f"📝 Similitud prompt vs '{movie2.title}': {sim_prompt_movie2:.4f}")
```
- Calcula qué tan similar es cada película respecto al prompt

---

## ✅ Resultado esperado

```
🎬 La lista de Schindler vs El club de la pelea: 0.37
📝 Similitud prompt vs La lista de Schindler: 0.45
📝 Similitud prompt vs El club de la pelea: 0.28
```

✅ Esto sugiere que ambas películas están relacionadas y el sistema puede recomendar la más cercana al prompt.

---

## ✅ Actividad

La actividad en este punto es **cambiar las películas y el prompt** en el código y verificar los resultados de la similitud del coseno para diferentes combinaciones. El código del comando lo pueden encontrar en el archivo [movie_similarities.py](aux_files/movie_similarities.py).

✅ Entregable: Captura de pantalla del comando donde se puedan observar las películas seleccionadas y el prompt de búsqueda generado, y captura de pantalla donde se pueda observar en consola los resultados de similitud del coseno generados.
