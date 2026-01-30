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
        "output_node_id": "6"
    }

_workflow_cache = None

def initialize_image_list():
    return []

async def process_image(prompt_text, source_path):
    global _workflow_cache
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
    if _workflow_cache is None:
        with open("Qwen-Rapid-AIO.json", "r", encoding="utf-8") as f:
            _workflow_cache = json.load(f)
    workflow = json.loads(json.dumps(_workflow_cache))
    prompt_node_id = AppConfig.WORKFLOW_CONFIG["prompt_node_id"]
    image_node_id = AppConfig.WORKFLOW_CONFIG["image_node_id"]
    output_node_id = AppConfig.WORKFLOW_CONFIG["output_node_id"]
    workflow[prompt_node_id]["inputs"]["prompt"] = prompt_text
    workflow[image_node_id]["inputs"]["image"] = comfy_name
    workflow["2"]["inputs"]["seed"] = int(uuid.uuid4().hex, 16) >> 96
    response = requests.post(
        f"{AppConfig.COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": AppConfig.CLIENT_ID},
        timeout=30
    ).json()
    prompt_id = response['prompt_id']
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        async with session.ws_connect(f"{AppConfig.WS_URL}?clientId={AppConfig.CLIENT_ID}") as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data['type'] == 'executing' and data['data']['node'] is None and data['data']['prompt_id'] == prompt_id:
                        break
    history_resp = requests.get(f"{AppConfig.COMFY_URL}/history/{prompt_id}", timeout=30)
    if not history_resp.ok:
        raise Exception(f"Error getting history: {history_resp.status_code}")
    history = history_resp.json()
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
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"output_{uuid.uuid4().hex[:8]}.png")
        with open(temp_path, 'wb') as f:
            f.write(img_data)
        return temp_path
    else:
        raise Exception("No output images found in workflow result")

async def generate_handler(prompt, image_list, selected_index):
    if image_list is None or len(image_list) == 0:
        return image_list, [], "⚠️ Sube o selecciona una imagen primero.", selected_index
    if selected_index is not None and 0 <= selected_index < len(image_list):
        idx = selected_index
    else:
        idx = len(image_list) - 1
    original_item = image_list[idx]
    original_path = original_item['path']
    if isinstance(original_path, tuple):
        original_path = original_path[0]
    if not original_path:
        return image_list, [], "⚠️ Sube o selecciona una imagen primero.", selected_index
    new_img = await process_image(prompt, original_path)
    original_item['last_prompt'] = prompt
    new_item = {'path': new_img, 'last_prompt': prompt, 'new_prompt': prompt}
    image_list.insert(idx, new_item)
    gallery_list = []
    for item in image_list:
        path = item['path']
        if isinstance(path, tuple):
            path = path[0]
        gallery_list.append((path, item['new_prompt']))
    input_prompt = prompt
    new_image_index = idx
    return image_list, gr.Gallery(value=gallery_list, selected_index=new_image_index, visible=True), input_prompt, new_image_index

def handle_upload(files, image_list):
    if files is None:
        return image_list, None
    if image_list is None:
        image_list = initialize_image_list()
    initial_length = len(image_list)
    for item in files:
        if isinstance(item, tuple):
            file_path = item[0]
        else:
            file_path = item
        new_item = {
            'path': file_path,
            'last_prompt': "",
            'new_prompt': ""
        }
        image_list.append(new_item)
    if len(image_list) > initial_length:
        new_selected_index = initial_length
    else:
        new_selected_index = len(image_list) - 1 if len(image_list) > 0 else None
    return image_list, new_selected_index

def on_select(evt: gr.SelectData, image_list):
    idx = evt.index
    text_update = gr.update()
    return idx, text_update

with gr.Blocks(title="Nunchaku Pro Mobile") as demo:
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
    gallery.upload(handle_upload, [gallery, image_list], [image_list, selected_index])
    msg_input.submit(generate_handler, [msg_input, image_list, selected_index], [image_list, gallery, msg_input, selected_index])
    btn_send.click(generate_handler, [msg_input, image_list, selected_index], [image_list, gallery, msg_input, selected_index])
    gallery.select(on_select, image_list, [selected_index, msg_input])

if __name__ == "__main__":
    print("[STARTING] Iniciando servidor Gradio...")
    print("   - Para acceso local/LAN: http://<tu_ip_local>:7860")
    print("   - Si usas Caddy, ya debería estar accesible en tu dominio.")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)