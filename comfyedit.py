import gradio as gr
import json
import requests
import asyncio
import aiohttp
import uuid
import tempfile
import os


class AppConfig:
    SERVER_ADDR = "127.0.0.1:8188"
    COMFY_URL = f"http://{SERVER_ADDR}"
    WS_URL = f"ws://{SERVER_ADDR}/ws"
    CLIENT_ID = str(uuid.uuid4())
    WORKFLOW_CONFIG = {
        "prompt_node_id": "3",
        "image_node_id": "7",
        "output_node_id": "16"
    }


# Global cache for workflow
_workflow_cache = None


def load_workflow():
    global _workflow_cache
    if _workflow_cache is None:
        with open("Qwen-Rapid-AIO.json", "r", encoding="utf-8") as f:
            _workflow_cache = json.load(f)
    return json.loads(json.dumps(_workflow_cache))


def upload_image_to_server(image_path):
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{AppConfig.COMFY_URL}/upload/image",
            files={"image": f},
            data={"overwrite": "true"},
            timeout=30
        )
    if not resp.ok:
        raise Exception(f"Error uploading image: {resp.status_code}")
    return resp.json()["name"]


def download_generated_image(filename, subfolder="", file_type="output"):
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
    return img_data_resp.content


def save_image_locally(image_data):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, f"output_{uuid.uuid4().hex[:8]}.png")
    with open(temp_path, 'wb') as f:
        f.write(image_data)
    return temp_path


async def wait_for_completion(prompt_id):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        async with session.ws_connect(f"{AppConfig.WS_URL}?clientId={AppConfig.CLIENT_ID}") as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data['type'] == 'executing' and data['data']['node'] is None and data['data']['prompt_id'] == prompt_id:
                        return


def get_execution_history(prompt_id):
    history_resp = requests.get(f"{AppConfig.COMFY_URL}/history/{prompt_id}", timeout=30)
    if not history_resp.ok:
        raise Exception(f"Error getting history: {history_resp.status_code}")
    return history_resp.json()


async def process_image(prompt_text, source_path):
    comfy_name = upload_image_to_server(source_path)
    workflow = load_workflow()
    config = AppConfig.WORKFLOW_CONFIG

    # Update workflow with prompt and image
    workflow[config["prompt_node_id"]]["inputs"]["prompt"] = prompt_text
    workflow[config["image_node_id"]]["inputs"]["image"] = comfy_name
    workflow["2"]["inputs"]["seed"] = int(uuid.uuid4().hex, 16) >> 96

    # Submit prompt
    response = requests.post(
        f"{AppConfig.COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": AppConfig.CLIENT_ID},
        timeout=30
    ).json()
    prompt_id = response['prompt_id']

    # Wait for completion
    await wait_for_completion(prompt_id)

    # Get execution history
    history = get_execution_history(prompt_id)

    # Extract output data
    output_node_id = config["output_node_id"]
    if output_node_id not in history[prompt_id]['outputs']:
        raise Exception(f"Output node {output_node_id} not found in workflow result")

    output_data = history[prompt_id]['outputs'][output_node_id]
    if 'images' not in output_data:
        raise Exception(f"No images found in output node {output_node_id} for Qwen-Rapid-AIO workflow")

    output_images = output_data['images']
    if output_images and len(output_images) > 0:
        output = output_images[0]
        filename = output['filename']
        subfolder = output.get('subfolder', '')
        file_type = output.get('type', 'output')

        # Download the generated image
        img_data = download_generated_image(filename, subfolder, file_type)

        # Save to local temporary file
        return save_image_locally(img_data)
    else:
        raise Exception("No output images found in workflow result")


def get_image_index(selected_index, image_list):
    if selected_index is not None and 0 <= selected_index < len(image_list):
        return selected_index
    else:
        return len(image_list) - 1


def validate_image_path(image_path):
    if isinstance(image_path, tuple):
        image_path = image_path[0]
    if hasattr(image_path, 'name'):
        image_path = image_path.name
    return image_path


def update_image_list_with_new_item(image_list, original_idx, new_img_path, prompt):
    # Update the original item's last prompt
    image_list[original_idx]['last_prompt'] = prompt

    # Create and insert the new item at the position after the original
    new_item = {'path': new_img_path, 'last_prompt': prompt, 'new_prompt': prompt}
    image_list.insert(original_idx, new_item)


def create_gallery_list(image_list):
    gallery_list = []
    for item in image_list:
        path = validate_image_path(item['path'])
        gallery_list.append((path, item['new_prompt']))
    return gallery_list


async def generate_handler(prompt, image_list, selected_index):
    if image_list is None or len(image_list) == 0:
        return image_list, [], "⚠️ Sube o selecciona una imagen primero.", selected_index

    idx = get_image_index(selected_index, image_list)
    original_item = image_list[idx]
    original_path = validate_image_path(original_item['path'])

    if not original_path:
        return image_list, [], "⚠️ Sube o selecciona una imagen primero.", selected_index

    new_img = await process_image(prompt, original_path)
    update_image_list_with_new_item(image_list, idx, new_img, prompt)

    gallery_list = create_gallery_list(image_list)
    input_prompt = prompt
    # La nueva imagen se inserta EN la posición actual, desplazando la anterior
    new_image_index = idx

    return image_list, gr.Gallery(value=gallery_list, selected_index=new_image_index, visible=True), input_prompt, new_image_index


def handle_upload(files, image_list, selected_index):
    if files is None:
        return image_list, selected_index, gr.update()

    if image_list is None or callable(image_list):
        image_list = []

    # Crear un set de rutas existentes para evitar duplicados
    existing_paths = set()
    for item in image_list:
        p = validate_image_path(item['path'])
        if p: existing_paths.add(p)

    new_items = []
    for item in files:
        file_path = validate_image_path(item)
        if file_path not in existing_paths:
            new_item = {
                'path': file_path,
                'last_prompt': "",
                'new_prompt': ""
            }
            new_items.append(new_item)
            existing_paths.add(file_path)

    if not new_items:
        return image_list, selected_index, gr.update()

    # Insertar EN la selección actual para desplazar y seleccionar la nueva
    if selected_index is None:
        insert_idx = len(image_list)
    else:
        insert_idx = selected_index

    for item in new_items:
        image_list.insert(insert_idx, item)
        insert_idx += 1

    # Seleccionar la última imagen insertada
    new_selected_index = insert_idx - 1
    gallery_list = create_gallery_list(image_list)

    return image_list, new_selected_index, gr.Gallery(value=gallery_list, selected_index=new_selected_index)



def on_select(evt: gr.SelectData, image_list):
    idx = evt.index
    text_update = gr.update()
    return idx, text_update


with gr.Blocks(title="Nunchaku Pro Mobile", fill_width=True, fill_height=True) as demo:
    image_list = gr.State([])
    selected_index = gr.State(None)

    gr.Markdown("# 🎨 Nunchaku Qwen Editor")

    with gr.Column(scale=1):
        gallery = gr.Gallery(
            label="Historial / Click para subir",
            columns=2,
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

    gallery.upload(handle_upload, [gallery, image_list, selected_index], [image_list, selected_index, gallery])
    msg_input.submit(generate_handler, [msg_input, image_list, selected_index], [image_list, gallery, msg_input, selected_index])
    btn_send.click(generate_handler, [msg_input, image_list, selected_index], [image_list, gallery, msg_input, selected_index])
    gallery.select(on_select, image_list, [selected_index, msg_input])


if __name__ == "__main__":
    print("[STARTING] Iniciando servidor Gradio...")
    print("   - Para acceso local/LAN: http://<tu_ip_local>:7860")
    print("   - Si usas Caddy, ya debería estar accesible en tu dominio.")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)