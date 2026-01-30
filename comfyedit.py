import gradio as gr
import json
import requests
import asyncio
import aiohttp
import uuid
import tempfile
import os

# Configuración centralizada
class AppConfig:
    SERVER_ADDR = "127.0.0.1:8188"
    COMFY_URL = f"http://{SERVER_ADDR}"
    WS_URL = f"ws://{SERVER_ADDR}/ws"
    CLIENT_ID = str(uuid.uuid4())

    # Configuración del flujo de trabajo
    WORKFLOW_CONFIG = {
        "prompt_node_id": "3",
        "image_node_id": "7",
        "output_node_id": "6"
    }

# Caché para el flujo de trabajo
_workflow_cache = None

# Lista para almacenar la información de las imágenes
def initialize_image_list():
    return []

async def process_image(prompt_text, source_path):
    global _workflow_cache

    # Subir imagen
    with open(source_path, "rb") as f:
        resp = requests.post(
            f"{AppConfig.COMFY_URL}/upload/image",
            files={"image": f},
            data={"overwrite": "true"},
            timeout=30
        )

    if not resp.ok:
        raise Exception(f"Error uploading image: {resp.status_code}")

    comfy_name = resp.json()["name"]

    # Cargar flujo de trabajo desde caché o archivo
    if _workflow_cache is None:
        with open("Qwen-Rapid-AIO.json", "r", encoding="utf-8") as f:
            _workflow_cache = json.load(f)

    # Copiar flujo de workflow desde caché
    workflow = json.loads(json.dumps(_workflow_cache))  # Deep copy simple

    # Usar la configuración fija del flujo de trabajo
    prompt_node_id = AppConfig.WORKFLOW_CONFIG["prompt_node_id"]
    image_node_id = AppConfig.WORKFLOW_CONFIG["image_node_id"]
    output_node_id = AppConfig.WORKFLOW_CONFIG["output_node_id"]

    # Actualizar el prompt en el flujo de trabajo
    workflow[prompt_node_id]["inputs"]["prompt"] = prompt_text

    # Actualizar la imagen en el flujo de trabajo
    workflow[image_node_id]["inputs"]["image"] = comfy_name

    # Actualizar seed para aleatoriedad
    workflow["2"]["inputs"]["seed"] = int(uuid.uuid4().hex, 16) >> 96

    # Enviar solicitud de prompt
    response = requests.post(
        f"{AppConfig.COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": AppConfig.CLIENT_ID},
        timeout=30
    ).json()

    prompt_id = response['prompt_id']

    # Esperar resultado usando WebSocket
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        async with session.ws_connect(f"{AppConfig.WS_URL}?clientId={AppConfig.CLIENT_ID}") as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data['type'] == 'executing' and data['data']['node'] is None and data['data']['prompt_id'] == prompt_id:
                        break

    # Obtener historial
    history_resp = requests.get(f"{AppConfig.COMFY_URL}/history/{prompt_id}", timeout=30)
    if not history_resp.ok:
        raise Exception(f"Error getting history: {history_resp.status_code}")

    history = history_resp.json()

    # Verificar si el nodo de salida existe en el historial
    if output_node_id not in history[prompt_id]['outputs']:
        raise Exception(f"Output node {output_node_id} not found in workflow result")

    output_data = history[prompt_id]['outputs'][output_node_id]

    # Para Qwen-Rapid-AIO.json, el nodo de salida es "PreviewImage" que tiene "images"
    # según la estructura del archivo JSON
    if 'images' not in output_data:
        raise Exception(f"No images found in output node {output_node_id} for Qwen-Rapid-AIO workflow")

    output_images = output_data['images']

    if output_images and len(output_images) > 0:
        output = output_images[0]
        filename = output['filename']
        subfolder = output.get('subfolder', '')
        file_type = output.get('type', 'output')

        # Obtener datos de imagen
        if subfolder:
            img_data_resp = requests.get(
                f"{AppConfig.COMFY_URL}/view?filename={filename}&subfolder={subfolder}&type={file_type}",
                timeout=30
            )
        else:
            img_data_resp = requests.get(
                f"{AppConfig.COMFY_URL}/view?filename={filename}&type={file_type}",
                timeout=30
            )

        if not img_data_resp.ok:
            raise Exception(f"Error downloading image: {img_data_resp.status_code}")

        img_data = img_data_resp.content

        # Crear archivo temporal en el directorio actual para que Gradio pueda accederlo
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"output_{uuid.uuid4().hex[:8]}.png")

        with open(temp_path, 'wb') as f:
            f.write(img_data)

        return temp_path
    else:
        raise Exception("No output images found in workflow result")

async def generate_handler(prompt, image_list, selected_index):
    # Si no hay lista de imágenes o está vacía, retornar error
    if image_list is None or len(image_list) == 0:
        return image_list, [], "⚠️ Sube o selecciona una imagen primero.", selected_index

    # Determinar el índice a usar
    if selected_index is not None and 0 <= selected_index < len(image_list):
        idx = selected_index
    else:
        idx = len(image_list) - 1

    # Obtener la información de la imagen original
    original_item = image_list[idx]
    original_path = original_item['path']

    # Asegurarse de que original_path es una cadena y no una tupla
    if isinstance(original_path, tuple):
        original_path = original_path[0]

    if not original_path:
        return image_list, [], "⚠️ Sube o selecciona una imagen primero.", selected_index

    # Procesar la imagen con el prompt actual
    new_img = await process_image(prompt, original_path)

    # Actualizar la lista de imágenes
    # 1. Actualizar la imagen original:
    # - su last_prompt ahora es el prompt actual (porque se usó para generar la nueva imagen)
    original_item['last_prompt'] = prompt

    # 2. Insertar la nueva imagen EN LA POSICIÓN ACTUAL (en lugar de después de la imagen original)
    # - su last_prompt es el prompt actual (porque se usó para generarla)
    # - su new_prompt también es el prompt actual
    new_item = {'path': new_img, 'last_prompt': prompt, 'new_prompt': prompt}
    image_list.insert(idx, new_item)

    # Convertir la lista a la que necesita la galería
    gallery_list = []
    for item in image_list:
        path = item['path']
        # Asegurarse de que el path es una cadena
        if isinstance(path, tuple):
            path = path[0]
        # La etiqueta de la imagen debe mostrar el new_prompt
        gallery_list.append((path, item['new_prompt']))

    # Devolver la lista actualizada, la lista para la galería,
    # y el índice de la nueva imagen para que se seleccione automáticamente
    # Mantener el prompt actual en el input textbox
    input_prompt = prompt

    # Devolver el índice de la nueva imagen (que es idx porque se insertó en la posición actual)
    new_image_index = idx

    # Devolver la lista actualizada, la galería actualizada con la nueva imagen,
    # el input manteniendo el último prompt usado y el índice de la nueva imagen
    # Forcing the gallery to update with the new selected index
    return image_list, gr.Gallery(value=gallery_list, selected_index=new_image_index, visible=True), input_prompt, new_image_index

def handle_upload(files, image_list):
    # Actualiza el estado con los archivos subidos
    if files is None:
        return image_list

    # Si no hay lista de imágenes, inicializar una
    if image_list is None:
        image_list = initialize_image_list()

    # Agregar las nuevas imágenes a la lista
    for item in files:
        # Asegurarse de que file_path es solo la ruta, no una tupla
        if isinstance(item, tuple):
            file_path = item[0]  # Tomar solo la primera parte si es una tupla
        else:
            file_path = item  # Si no es tupla, usar directamente

        new_item = {
            'path': file_path,
            'last_prompt': "",
            'new_prompt': ""
        }
        image_list.append(new_item)

    return image_list

def on_select(evt: gr.SelectData, image_list):
    idx = evt.index
    # No actualizar el input textbox cuando se selecciona una imagen
    # El input textbox debe mantener siempre el último prompt usado
    text_update = gr.update()  # Esto mantiene el valor actual del textbox

    return idx, text_update

with gr.Blocks(title="Nunchaku Pro Mobile") as demo:
    # Estado para mantener la lista con la información de las imágenes
    image_list = gr.State(lambda: initialize_image_list())
    selected_index = gr.State(None)

    gr.Markdown("# 🎨 Nunchaku Qwen Editor")

    with gr.Column():
        gallery = gr.Gallery(
            label="Historial / Click para subir",
            columns=2,
            height=550,
            preview=True,
            interactive=True,
            object_fit="contain",
            selected_index=0,
        )

        btn_send = gr.Button("🚀 GENERAR EDICIÓN", variant="primary", size="lg")

        msg_input = gr.Textbox(
            label="¿Qué quieres cambiar?",
            placeholder="Escribe y presiona Enter...",
            autofocus=True
        )

    # Eventos
    # Al subir archivos, actualizamos el estado interno
    gallery.upload(handle_upload, [gallery, image_list], [image_list])

    # Al presionar Enter o Clic en Generar
    msg_input.submit(generate_handler, [msg_input, image_list, selected_index], [image_list, gallery, msg_input, selected_index])
    btn_send.click(generate_handler, [msg_input, image_list, selected_index], [image_list, gallery, msg_input, selected_index])

    # Guardar selección al tocar
    gallery.select(on_select, image_list, [selected_index, msg_input])

if __name__ == "__main__":
    print("[STARTING] Iniciando servidor Gradio...")
    print("   - Para acceso local/LAN: http://<tu_ip_local>:7860")
    print("   - Si usas Caddy, ya debería estar accesible en tu dominio.")

    # Se lanza en 0.0.0.0 para ser accesible desde la red local y por Caddy.
    # share is always False as requested
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)