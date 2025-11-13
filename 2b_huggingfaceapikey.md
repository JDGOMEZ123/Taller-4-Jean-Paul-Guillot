## Creación de cuenta y access token en huggingface

Una alternativa para los modelos de IA generativa es HuggingFace, que es una plataforma líder que actúa como un hub para compartir, entrenar y desplegar modelos de inteligencia artificial, especialmente en procesamiento de lenguaje natural y otras tareas de IA, facilitando el acceso a herramientas y modelos preentrenados.

Crea una cuenta en dicha pagina para posteriormente generar el token (https://huggingface.co/welcome):
![image](https://github.com/user-attachments/assets/39e90049-eed2-4499-9924-4d9082c7a623)

Una vez creada, instala el paquete y accede al apartado de tokens (si no lo incluiste en el archivo ``requirements.txt``).
```bash
pip install huggingface_hub
```
![image](https://github.com/user-attachments/assets/e1a233bb-88d1-40fc-a08e-ce3d9023d4ad)


Vamos a settings/Access tokens
![image](https://github.com/user-attachments/assets/c4725a9b-002d-45f2-b331-0bed7c771bad)


y generaremos nuestro nuevo token:
![image](https://github.com/user-attachments/assets/071ab31d-4c0b-4074-8e99-b20065ed92b0)


Le ponemos un nombre y le otorgamos los permisos necesarios: 

![image](https://github.com/user-attachments/assets/c619f6c5-f21e-4c63-8e82-bc12a114c491)


![image](https://github.com/user-attachments/assets/5c6dec90-7189-472a-b470-d681523a9c64)

Bajamos, creamos y copiamos el codigo del token **(guardalo en el archivo .env de tu proyecto!)**
