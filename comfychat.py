import gradio as gr
import json
import requests
import asyncio
import aiohttp
import uuid
import os
import tempfile
import logging
import time

# ----------------------------------------------------
# Configuración
# ----------------------------------------------------
SERVER_ADDR = "127.0.0.1:8188"
COMFY_URL = f"http://{SERVER_ADDR}"
WS_URL = f"ws://{SERVER_ADDR}/ws"
CLIENT_ID = str(uuid.uuid4())

OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "comfychat_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------
# Estado global
# ----------------------------------------------------
base_image_name = None
base_image_path = None

MODE_CONFIG = {
    "image": {
        "default_label": "🖼️ Generar imagen",
    },
    "video": {
        "default_label": "🎬 Generate video",
    },
}

# ------------------------------------
# Utilidades
# ------------------------------------
async def upload_to_comfy(file_path):
    """Upload image to ComfyUI and update global state (Async)."""
    global base_image_name, base_image_path
    async with aiohttp.ClientSession() as session:
        with open(file_path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field('image', f)
            data.add_field('overwrite', 'true')
            async with session.post(f"{COMFY_URL}/upload/image", data=data) as resp:
                result = await resp.json()
    base_image_name = result["name"]
    base_image_path = file_path
    logger.info(f"📤 Uploaded to ComfyUI: {base_image_name}")
    return base_image_name

async def submit_and_wait(workflow):
    """Submit workflow and wait via WebSocket."""
    payload = {"prompt": workflow, "client_id": CLIENT_ID}
    response = requests.post(f"{COMFY_URL}/prompt", json=payload).json()
    if "prompt_id" not in response:
        raise Exception(f"ComfyUI no retornó prompt_id: {response}")

    prompt_id = response["prompt_id"]

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f"{WS_URL}?clientId={CLIENT_ID}") as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if (
                        data["type"] == "executing"
                        and data["data"]["node"] is None
                        and data["data"]["prompt_id"] == prompt_id
                    ):
                        return prompt_id
    raise Exception("WebSocket cerró sin confirmar")


async def poll_until_output(prompt_id, output_node, timeout_secs=600):
    """Poll until output appears."""
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        history = requests.get(f"{COMFY_URL}/history/{prompt_id}").json()
        if prompt_id in history:
            status = history[prompt_id]
            if status.get("status", {}).get("status_str") == "error":
                raise Exception("ComfyUI reportó error")
            outputs = status.get("outputs", {})
            if output_node in outputs:
                return outputs[output_node]
        await asyncio.sleep(2)
    raise Exception(f"Timeout ({timeout_secs}s)")


# ----------------------------------------------------
# Generadores
# ----------------------------------------------------
async def process_image(prompt_text, duration=0, enhance_enabled=False):
    """Generate image with Qwen. Returns local path and ComfyUI URL."""
    global base_image_name, base_image_path

    if not base_image_name:
        raise Exception("No hay imagen base")

    with open("Qwen-Rapid-AIO.json", "r", encoding="utf-8") as f:
        workflow = json.load(f)

    workflow["17"]["inputs"]["prompt"] = prompt_text
    workflow["17"]["inputs"]["bypass"] = 'false' if enhance_enabled else 'true'
    workflow["7"]["inputs"]["image"] = base_image_name
    workflow["2"]["inputs"]["seed"] = int(uuid.uuid4().hex, 16) >> 96

    prompt_id = await submit_and_wait(workflow)
    logger.info("🟢 Image prompt: %s", prompt_id)

    history = requests.get(f"{COMFY_URL}/history/{prompt_id}").json()
    outputs = history.get(prompt_id, {}).get("outputs", {})

    # Recuperar prompt mejorado del nodo 18
    enhanced_prompt = None
    try:
        if "18" in outputs:
            val = outputs["18"]
            # Intentar extraer texto (formato habitual: {"string": ["texto"]} o {"text": ["texto"]})
            if isinstance(val, dict):
                for k in ["text", "string", "value"]:
                    if k in val and isinstance(val[k], list) and val[k]:
                        enhanced_prompt = val[k][0]
                        break
    except Exception as e:
        logger.warning(f"⚠️ No se pudo recuperar prompt del nodo 18: {e}")

    expected_node = "16"
    if expected_node not in outputs:
        raise Exception(f"No output en nodo {expected_node}: {list(outputs.keys())}")

    info = outputs[expected_node]["images"][0]
    comfy_url = f"{COMFY_URL}/view?filename={info['filename']}&type={info['type']}"

    logger.info(f"🖼️ Image URL: {comfy_url}")

    # Descargar temporalmente para mostrar Y para re-upload
    resp = requests.get(comfy_url)
    if resp.status_code != 200:
        raise Exception(f"Download failed: HTTP {resp.status_code}")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.write(resp.content)
    temp_file.close()

    # Re-upload como nueva base (Await Async version)
    await upload_to_comfy(temp_file.name)

    return temp_file.name, comfy_url, enhanced_prompt


async def process_video(prompt_text, duration=5, enhance_enabled=False):
    """Generate video with LTXV. Returns local path and ComfyUI URL."""
    if not base_image_name:
        raise Exception("Se requiere imagen base")

    with open("LTXV-DoAlmostEverything-v3.json", "r", encoding="utf-8") as f:
        workflow = json.load(f)

    workflow["5180"]["inputs"]["image"] = base_image_name
    workflow["5257"]["inputs"]["value"] = prompt_text
    workflow["5253:5189:5111"]["inputs"]["noise_seed"] = int(uuid.uuid4().hex, 16) % (2**32)
    workflow["5237"]["inputs"]["value"] = duration
    workflow["5259"]["inputs"]["bypass"] = 'false' if enhance_enabled else 'true'

    payload = {"prompt": workflow, "client_id": CLIENT_ID}
    response = requests.post(f"{COMFY_URL}/prompt", json=payload).json()
    if "prompt_id" not in response:
        raise Exception(f"No prompt_id: {response}")

    prompt_id = response["prompt_id"]
    logger.info("🎬 Video prompt: %s", prompt_id)

    node_output = await poll_until_output(prompt_id, "4958")
    logger.info("🎬 Video completed")

    # Recuperar prompt mejorado del nodo 5248
    enhanced_prompt = None
    try:
        history_data = requests.get(f"{COMFY_URL}/history/{prompt_id}").json()
        outputs = history_data.get(prompt_id, {}).get("outputs", {})
        if "5248" in outputs:
            val = outputs["5248"]
            # Intentar extraer texto (formato habitual: {"string": ["texto"]} o {"text": ["texto"]})
            if isinstance(val, dict):
                for k in ["text", "string", "value"]:
                    if k in val and isinstance(val[k], list) and val[k]:
                        enhanced_prompt = val[k][0]
                        break
    except Exception as e:
        logger.warning(f"⚠️ No se pudo recuperar prompt del nodo 5248: {e}")

    output_info = (
        node_output.get("gifs", node_output.get("images", [None]))[0]
        if node_output else None
    )
    if not output_info:
        raise Exception(f"No video output: {node_output.keys() if node_output else 'None'}")

    filename = output_info["filename"]
    subfolder = output_info.get("subfolder", "")
    file_type = output_info.get("type", "output")
    comfy_url = f"{COMFY_URL}/view?filename={filename}&type={file_type}&subfolder={subfolder}"

    logger.info(f"🎬 Video URL: {comfy_url}")

    # Descargar a temporal para reproducción
    resp = requests.get(comfy_url)
    if resp.status_code != 200:
        raise Exception(f"Download failed: HTTP {resp.status_code}")
    if len(resp.content) == 0:
        raise Exception("Video vacío")

    video_path = os.path.join(
        OUTPUT_DIR,
        f"video_{prompt_id[:8]}_{int(time.time())}_{filename.replace('/', '_')}"
    )
    with open(video_path, "wb") as f:
        f.write(resp.content)

    logger.info(f"Video guardado: {video_path} ({os.path.getsize(video_path)} bytes)")
    return video_path, comfy_url, enhanced_prompt


GENERATORS = {
    "image": process_image,
    "video": process_video,
}


# ----------------------------------------------------
# Chat function
# ----------------------------------------------------
async def chat_fn(message, history, mode, original_text=None, enhanced_text=None, duration=5, enhance_enabled=False):
    """Main chat orchestrator."""
    yield history
    text = message.get("text", "")
    files = message.get("files", []) or []

    # Extraer path de forma robusta
    src_path = None
    if files:
        f = files[0]
        if isinstance(f, dict):
            src_path = f.get("path")
        elif isinstance(f, (list, tuple)):
            src_path = f[0]
        else:
            src_path = f

    config = MODE_CONFIG[mode]

    # Upload new image if provided (Await Async version)
    if src_path:
        await upload_to_comfy(src_path)
        history.append({
            "role": "user",
            "content": {"path": src_path}
        })
        yield history

    # Validate base image exists
    if not base_image_name:
        history.append({
            "role": "assistant",
            "content": "⚠️ Se requiere una imagen. Sube una primero."
        })
        yield history
        return

    # Add user text
    display_text = original_text if original_text is not None else text
    if display_text or not src_path:
        history.append({
            "role": "user",
            "content": display_text or config["default_label"]
        })
        yield history

    # Show enhanced prompt immediately if available
    if enhanced_text and enhanced_text != original_text:
        # En modo video, esperamos el prompt de ComfyUI (nodo 5248), así que ocultamos el de Llama aquí
        if mode != "video":
            history.append({
                "role": "assistant",
                "content": f"✨ *Prompt mejorado:* {enhanced_text}"
            })
            yield history

    try:
        # Generate (retorna path local + URL de ComfyUI)
        comfy_prompt = None
        result_path, comfy_url, comfy_prompt = await GENERATORS[mode](text, duration, enhance_enabled)

        logger.info("%s OK: %s | URL: %s", mode.upper(), result_path, comfy_url)

        # Add result con path local (para mostrar) y metadata con URL (para persistencia)
        history.append({
            "role": "assistant",
            "content": {"path": result_path},
            "metadata": {
                "comfy_url": comfy_url,
                "media_type": mode
            }
        })

        # Si hay prompt mejorado desde ComfyUI, agregarlo al historial
        if comfy_prompt:
            history.append({
                "role": "assistant",
                "content": f"✨ *Prompt mejorado (LTXV):* {comfy_prompt}"
            })

        yield history

    except Exception as e:
        logger.exception("❗ Error:")
        history.append({
            "role": "assistant",
            "content": f"Error procesando {mode}: {e}"
        })
        yield history


# ----------------------------------------------------
# Handler
# ----------------------------------------------------
async def handle_generation(message, history, auto_enhance_checked, mode, duration, progress=gr.Progress()):
    """Unified handler with optional enhancement."""
    progress(0, desc="Iniciando generación...") # Start
    yield history # Ensure initial spinner

    user_text = message.get("text", "") if message else ""
    files = (message.get("files", []) or []) if message else []
    src_path = None
    if files:
        f = files[0]
        if isinstance(f, dict):
            src_path = f.get("path")
        elif isinstance(f, (list, tuple)):
            src_path = f[0]
        else:
            src_path = f

    prompt_to_use = user_text
    enhanced_prompt = None

    progress(0.4, desc="Generando contenido con ComfyUI...") # Progress update before ComfyUI call

    gen_message = {"text": prompt_to_use, "files": files}
    async for h in chat_fn(gen_message, history, mode, user_text, enhanced_prompt, duration, auto_enhance_checked):
        yield h

    progress(1.0, desc="Generación completa.") # End


# ----------------------------------------------------
# UI
# ----------------------------------------------------
with gr.Blocks(fill_width=True, fill_height=True) as demo:
    gr.Markdown("## 💬 ComfyUI Multi-Modal Chat (Image + Video)")

    chatbot = gr.Chatbot(
        label="QwenAI",
        height="75vh",
        elem_id="chatbot",
        autoscroll=True,
        value=[],  # load_chat_history() desactivado temporalmente
        group_consecutive_messages=False,
    )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        placeholder="Enter message and upload image...",
        show_label=False,
    )

    with gr.Row():
        auto_enhance = gr.Checkbox(label="🤖 Mejorar con IA", value=False, scale=1)
        duration_slider = gr.Slider(label="⏱️ Duración (Video)", minimum=1, maximum=10, value=3, step=1, scale=2)
        btn_image = gr.Button("Generar Imagen 🖼️", variant="primary", scale=3)
        btn_video = gr.Button("Generar Video 🎬", variant="primary", scale=3)
        btn_clear = gr.Button("🗑️ Limpiar", variant="secondary", scale=2)

    # Handlers wrapper for streaming
    async def handle_image(msg, hist, enhance, dur, progress=gr.Progress()):
        async for h in handle_generation(msg, hist, enhance, "image", dur, progress):
            yield h
    async def handle_video(msg, hist, enhance, dur, progress=gr.Progress()):
        async for h in handle_generation(msg, hist, enhance, "video", dur, progress):
            yield h
    # Wire buttons
    btn_image.click(
        fn=handle_image,
        inputs=[chat_input, chatbot, auto_enhance, duration_slider],
        outputs=[chatbot],
        show_progress="minimal",
    ).then(
        fn=lambda: gr.update(value=None),
        outputs=[chat_input]
    )

    btn_video.click(
        fn=handle_video,
        inputs=[chat_input, chatbot, auto_enhance, duration_slider],
        outputs=[chatbot],
        show_progress="minimal",
    ).then(
        fn=lambda: gr.update(value=None),
        outputs=[chat_input]
    )

    # Enter key → imagen
    chat_input.submit(
        fn=handle_image,
        inputs=[chat_input, chatbot, auto_enhance, duration_slider],
        outputs=[chatbot],
        show_progress="minimal",
    ).then(
        fn=lambda: gr.update(value=None),
        outputs=[chat_input]
    )

    # Clear button
    def clear_all():
        global base_image_name, base_image_path
        base_image_name = None
        base_image_path = None
        return []

    btn_clear.click(
        fn=clear_all,
        outputs=[chatbot],
    )

if __name__ == "__main__":
    demo.launch()