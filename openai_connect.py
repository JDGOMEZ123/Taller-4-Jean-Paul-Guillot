from openai import OpenAI
import os
from dotenv import load_dotenv

# Carga las variables de entorno desde el archivo .env
from pathlib import Path
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Inicializa el cliente de OpenAI con la API Key
gpt_client = OpenAI(api_key=os.environ.get('openai_apikey'))

# Ejemplo: obtener información de la API
# response = gpt_client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user", "content": "Hola, ¿quién eres?"}]
# )
# print(response.choices[0].message.content)


def get_completion(prompt, model="gpt-3.5-turbo"):
    # Define el mensaje con el rol 'user' y el contenido que enviamos
    messages = [{"role": "user", "content": prompt}]

    # Llama a la API con el modelo y los mensajes
    response = gpt_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0  # Controla la creatividad (0 = más preciso)
    )

    # Retorna solo el contenido de la respuesta generada
    return response.choices[0].message.content.strip()
