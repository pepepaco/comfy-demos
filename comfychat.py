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
# JS: solicitar permiso al cargar y exponer showNotification(url)
# ----------------------------------------------------
notification_js = """
function() {
    if ("Notification" in window) {
        Notification.requestPermission().then(function(perm) {
            console.log("Notification permission:", perm);
        });
    }

    // showNotification recibe la URL del resultado como argumento
    window.showNotification = function(url) {
        console.log("showNotification called, url=", url);
        if (!("Notification" in window)) {
            console.warn("Notifications not supported");
            return;
        }
        if (Notification.permission !== "granted") {
            console.warn("Notification permission not granted:", Notification.permission);
            return;
        }
        const options = {
            body: url
                ? "Haz clic para descargar tu resultado."
                : "Tu video/imagen ha terminado de procesarse.",
            icon: "https://em-content.zobj.net/source/google/387/magic-wand_1fa84.png",
            requireInteraction: false,
        };
        const n = new Notification("✅ Generación Completada", options);
        if (url) {
            n.onclick = function() {
                window.open(url, "_blank");
                n.close();
            };
        }
        if (navigator.vibrate) {
            navigator.vibrate([200, 100, 200]);
        }
    };
}
"""

# JS que Gradio llamará pasando el valor del componente hidden como primer argumento
trigger_notification_js = """
(url) => {
    console.log("trigger_notification_js called, url =", url);
    if (window.showNotification) {
        window.showNotification(url);
    } else {
        console.warn("showNotification no está definido todavía");
    }
    return url;
}
"""

# ----------------------------------------------------
# Estado global
# ----------------------------------------------------
base_image_name = None
base_image_path = None

MODE_CONFIG = {
    "image": {"default_label": "🖼️ Generar imagen"},
    "video": {"default_label": "🎬 Generate video"},
}

# ------------------------------------
# Utilidades
# ------------------------------------
async def upload_to_comfy(file_path):
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

def find_node_id(workflow, title):
    """Busca el ID de un nodo por su título en _meta."""
    for node_id, node in workflow.items():
        if node.get("_meta", {}).get("title") == title:
            return node_id
    return None

async def submit_and_wait(workflow):
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


async def poll_until_output(prompt_id, output_node, timeout_secs=600, progress_cb=None):
    """
    Poll until output appears.
    progress_cb(value, desc) se llama periódicamente para mantener la barra activa.
    """
    deadline = time.time() + timeout_secs
    elapsed_start = time.time()
    tick = 0
    while time.time() < deadline:
        history = requests.get(f"{COMFY_URL}/history/{prompt_id}").json()
        if prompt_id in history:
            status = history[prompt_id]
            if status.get("status", {}).get("status_str") == "error":
                raise Exception("ComfyUI reportó error")
            outputs = status.get("outputs", {})
            if output_node in outputs:
                if progress_cb:
                    progress_cb(0.95, desc="Descargando resultado...")
                return outputs[output_node]

        tick += 1
        elapsed = time.time() - elapsed_start
        # Progreso simulado que sube lentamente hasta 0.9 sin llegar nunca (asíntota)
        fake_progress = 0.15 + 0.75 * (1 - 1 / (1 + elapsed / 60))
        if progress_cb:
            progress_cb(fake_progress, desc=f"⏳ ComfyUI procesando... {int(elapsed)}s")

        await asyncio.sleep(2)

    raise Exception(f"Timeout ({timeout_secs}s)")


# ----------------------------------------------------
# Generadores
# ----------------------------------------------------
async def process_image(prompt_text, duration=0, enhance_enabled=False, progress_cb=None):
    global base_image_name, base_image_path

    if not base_image_name:
        raise Exception("No hay imagen base")

    if progress_cb:
        progress_cb(0.10, desc="Cargando workflow de imagen...")

    with open("Qwen-Rapid-AIO.json", "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # Usar find_node_id para no depender de IDs fijos
    openai_id = find_node_id(workflow, "OpenAICompat")
    workflow[openai_id]["inputs"]["prompt"] = prompt_text
    workflow[openai_id]["inputs"]["bypass"] = 'false' if enhance_enabled else 'true'

    workflow[find_node_id(workflow, "Optional Input Image")]["inputs"]["image"] = base_image_name
    workflow[find_node_id(workflow, "KSampler")]["inputs"]["seed"] = int(uuid.uuid4().hex, 16) >> 96

    output_node_id = find_node_id(workflow, "Save Image for Chat")

    if progress_cb:
        progress_cb(0.15, desc="Enviando a ComfyUI...")

    prompt_id = await submit_and_wait(workflow)
    logger.info("🟢 Image prompt: %s", prompt_id)

    # Esperar con polling y progreso continuo
    await poll_until_output(prompt_id, output_node_id, timeout_secs=300, progress_cb=progress_cb)

    if progress_cb:
        progress_cb(0.95, desc="Recuperando imagen...")

    history = requests.get(f"{COMFY_URL}/history/{prompt_id}").json()
    outputs = history.get(prompt_id, {}).get("outputs", {})

    enhanced_prompt = None
    preview_id = find_node_id(workflow, "Preview as Text")
    try:
        if preview_id and preview_id in outputs:
            val = outputs[preview_id]
            if isinstance(val, dict):
                for k in ["text", "string", "value"]:
                    if k in val and isinstance(val[k], list) and val[k]:
                        enhanced_prompt = val[k][0]
                        break
    except Exception as e:
        logger.warning(f"⚠️ No se pudo recuperar prompt del nodo 18: {e}")

    if output_node_id not in outputs:
        raise Exception(f"No output en nodo {output_node_id}: {list(outputs.keys())}")

    info = outputs[output_node_id]["images"][0]
    comfy_url = f"{COMFY_URL}/view?filename={info['filename']}&type={info['type']}"
    logger.info(f"🖼️ Image URL: {comfy_url}")

    resp = requests.get(comfy_url)
    if resp.status_code != 200:
        raise Exception(f"Download failed: HTTP {resp.status_code}")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.write(resp.content)
    temp_file.close()

    await upload_to_comfy(temp_file.name)

    return temp_file.name, comfy_url, enhanced_prompt


async def process_video(prompt_text, duration=5, enhance_enabled=False, progress_cb=None):
    if not base_image_name:
        raise Exception("Se requiere imagen base")

    if progress_cb:
        progress_cb(0.10, desc="Cargando workflow de video...")

    with open("LTXV-DoAlmostEverything-v3.json", "r", encoding="utf-8") as f:
        workflow = json.load(f)

    workflow[find_node_id(workflow, "Load Image")]["inputs"]["image"] = base_image_name
    openai_id = find_node_id(workflow, "OpenAICompat")
    workflow[openai_id]["inputs"]["prompt"] = prompt_text
    workflow[openai_id]["inputs"]["bypass"] = 'false' if enhance_enabled else 'true'

    workflow[find_node_id(workflow, "RandomNoise")]["inputs"]["noise_seed"] = int(uuid.uuid4().hex, 16) % (2**32)
    workflow[find_node_id(workflow, "Duration")]["inputs"]["value"] = duration

    output_node_id = find_node_id(workflow, "Save Video")

    if progress_cb:
        progress_cb(0.15, desc="Enviando a ComfyUI...")

    payload = {"prompt": workflow, "client_id": CLIENT_ID}
    response = requests.post(f"{COMFY_URL}/prompt", json=payload).json()
    if "prompt_id" not in response:
        raise Exception(f"No prompt_id: {response}")

    prompt_id = response["prompt_id"]
    logger.info("🎬 Video prompt: %s", prompt_id)

    # Esperar con polling y progreso continuo
    node_output = await poll_until_output(
        prompt_id, output_node_id, timeout_secs=600, progress_cb=progress_cb
    )
    logger.info("🎬 Video completed")

    if progress_cb:
        progress_cb(0.96, desc="Recuperando prompt mejorado...")

    enhanced_prompt = None
    preview_id = find_node_id(workflow, "Preview as Text")
    try:
        history_data = requests.get(f"{COMFY_URL}/history/{prompt_id}").json()
        out = history_data.get(prompt_id, {}).get("outputs", {})
        if preview_id and preview_id in out:
            val = out[preview_id]
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
        raise Exception(f"No video output: {list(node_output.keys()) if node_output else 'None'}")

    filename = output_info["filename"]
    subfolder = output_info.get("subfolder", "")
    file_type = output_info.get("type", "output")
    comfy_url = f"{COMFY_URL}/view?filename={filename}&type={file_type}&subfolder={subfolder}"
    logger.info(f"🎬 Video URL: {comfy_url}")

    if progress_cb:
        progress_cb(0.97, desc="Descargando video...")

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
async def chat_fn(message, history, mode, original_text=None, enhanced_text=None,
                  duration=5, enhance_enabled=False, progress_cb=None):
    text = message.get("text", "")
    files = message.get("files", []) or []

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

    if src_path:
        await upload_to_comfy(src_path)
        history.append({"role": "user", "content": {"path": src_path}})
        yield history, ""

    if not base_image_name:
        history.append({
            "role": "assistant",
            "content": "⚠️ Se requiere una imagen. Sube una primero."
        })
        yield history, ""
        return

    display_text = original_text if original_text is not None else text
    if display_text or not src_path:
        history.append({
            "role": "user",
            "content": display_text or config["default_label"]
        })
        yield history, ""

    if enhanced_text and enhanced_text != original_text and mode != "video":
        history.append({
            "role": "assistant",
            "content": f"✨ *Prompt mejorado:* {enhanced_text}"
        })
        yield history, ""

    try:
        result_path, comfy_url, comfy_prompt = await GENERATORS[mode](
            text, duration, enhance_enabled, progress_cb=progress_cb
        )
        logger.info("%s OK: %s | URL: %s", mode.upper(), result_path, comfy_url)

        history.append({
            "role": "assistant",
            "content": {"path": result_path},
            "metadata": {"comfy_url": comfy_url, "media_type": mode}
        })

        if comfy_prompt:
            history.append({
                "role": "assistant",
                "content": f"✨ *Prompt mejorado (LTXV):* {comfy_prompt}"
            })

        # Devuelve el historial Y la URL para la notificación
        yield history, comfy_url

    except Exception as e:
        logger.exception("❗ Error:")
        history.append({
            "role": "assistant",
            "content": f"Error procesando {mode}: {e}"
        })
        yield history, ""


# ----------------------------------------------------
# Handler principal
# ----------------------------------------------------
async def handle_generation(message, history, auto_enhance_checked, mode, duration,
                             progress=gr.Progress()):
    """
    Genera imagen o video.
    Yields: (chatbot_history, notification_url)
    notification_url se pone en el componente hidden que dispara la notificación JS.
    """
    # gr.Progress() en generators: se mantiene activa mientras el generador sigue abierto.
    # Llamamos progress() periódicamente desde poll_until_output vía progress_cb.
    progress(0.05, desc="Iniciando generación...")

    user_text = (message.get("text", "") if message else "")
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

    # Wrapper para convertir gr.Progress a un callback simple
    def progress_cb(value, desc=""):
        progress(value, desc=desc)

    progress_cb(0.08, desc="Preparando workflow...")

    gen_message = {"text": prompt_to_use, "files": files}

    last_history = history
    last_url = ""

    async for h, url in chat_fn(
        gen_message, history, mode,
        user_text, enhanced_prompt, duration,
        auto_enhance_checked, progress_cb=progress_cb
    ):
        last_history = h
        last_url = url
        # Yield intermedio para actualizar el chatbot (sin notificación aún)
        yield last_history, ""

    # Progreso al 100% solo aquí, cuando ya tenemos la respuesta de ComfyUI
    progress(1.0, desc="¡Completado!")

    # Yield final con la URL real → dispara la notificación JS
    yield last_history, last_url


# ----------------------------------------------------
# UI
# ----------------------------------------------------
with gr.Blocks(
    fill_width=True,
    fill_height=True,
    head=f"<script>{notification_js}</script>"
) as demo:

    gr.Markdown("## 💬 ComfyUI Multi-Modal Chat (Image + Video)")

    chatbot = gr.Chatbot(
        label="QwenAI",
        height="75vh",
        elem_id="chatbot",
        autoscroll=True,
        value=[],
        group_consecutive_messages=False,
    )

    # Componente oculto que transporta la URL desde Python hacia el JS de notificación
    notification_url = gr.Textbox(value="", visible=False, elem_id="notification_url")

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        placeholder="Escribe un mensaje y sube una imagen...",
        show_label=False,
    )

    with gr.Row():
        auto_enhance = gr.Checkbox(label="🤖 Mejorar con IA", value=False, scale=1)
        duration_slider = gr.Slider(
            label="⏱️ Duración (Video)", minimum=1, maximum=10,
            value=3, step=1, scale=2
        )
        btn_image = gr.Button("Generar Imagen 🖼️", variant="primary", scale=3)
        btn_video  = gr.Button("Generar Video 🎬",  variant="primary", scale=3)
        btn_clear  = gr.Button("🗑️ Limpiar",         variant="secondary", scale=2)

    # --- Wrappers por modo ---
    async def handle_image_stream(msg, hist, enhance, dur, progress=gr.Progress()):
        async for h, url in handle_generation(msg, hist, enhance, "image", dur, progress):
            yield h, url

    async def handle_video_stream(msg, hist, enhance, dur, progress=gr.Progress()):
        async for h, url in handle_generation(msg, hist, enhance, "video", dur, progress):
            yield h, url

    # --- Imagen: click del botón O Enter ---
    gr.on(
        triggers=[btn_image.click, chat_input.submit],
        fn=handle_image_stream,
        inputs=[chat_input, chatbot, auto_enhance, duration_slider],
        outputs=[chatbot, notification_url],   # ← dos outputs
        show_progress="minimal",
    ).then(
        fn=lambda: gr.update(value=None),
        outputs=[chat_input],
    ).then(
        # Llama showNotification(url) pasando el valor del componente hidden
        fn=None,
        inputs=[notification_url],
        js=trigger_notification_js,
        outputs=[notification_url],
    )

    # --- Video: click del botón ---
    btn_video.click(
        fn=handle_video_stream,
        inputs=[chat_input, chatbot, auto_enhance, duration_slider],
        outputs=[chatbot, notification_url],
        show_progress="minimal",
    ).then(
        fn=lambda: gr.update(value=None),
        outputs=[chat_input],
    ).then(
        fn=None,
        inputs=[notification_url],
        js=trigger_notification_js,
        outputs=[notification_url],
    )

    # --- Limpiar ---
    def clear_all():
        global base_image_name, base_image_path
        base_image_name = None
        base_image_path = None
        return [], ""

    btn_clear.click(
        fn=clear_all,
        outputs=[chatbot, notification_url],
    )


if __name__ == "__main__":
    demo.launch(allowed_paths=[os.getcwd()], pwa=True)