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
import base64
from pathlib import Path

# ----------------------------------------------------
# Configuración
# ----------------------------------------------------
SERVER_ADDR = "127.0.0.1:8188"
COMFY_URL = f"http://{SERVER_ADDR}"
WS_URL = f"ws://{SERVER_ADDR}/ws"
CLIENT_ID = str(uuid.uuid4())
LLAMA_CPP_URL = "http://127.0.0.1:8080"

# Directorio para historial (solo JSON, archivos están en ComfyUI)
PERSISTENT_DIR = Path(__file__).parent / "gradio_cache"
PERSISTENT_DIR.mkdir(exist_ok=True)
CHAT_HISTORY_FILE = PERSISTENT_DIR / "chat_history.json"

OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "comfychat_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------
# System Prompts
# ----------------------------------------------------
IMAGE_SYSTEM_PROMPT = """# ROLE
You are a Senior Prompt Engineer specialized in Image Editing for models like Qwen Image Edit and Flux Kontext. Your goal is to transform a user's edit request and an input image into a precise, high-fidelity technical prompt that ensures seamless integration.

# CORE OPERATIONAL RULES
1. **ANALYZE FIRST**: Identify the subject's identity, lighting direction, material textures, and the overall artistic style (e.g., cinematic, 3D render, oil painting).
2. **ANCHORING**: Explicitly name the elements that MUST remain identical (e.g., "keeping the background, pose, and facial features unchanged").
3. **PHYSICAL INTEGRATION**: When adding or changing objects, describe how they interact with the environment. Mention shadows, reflections, and how the new element sits or rests within the space.
4. **MATERIAL CONSISTENCY**: Match the texture of the new element to the existing scene (e.g., if the scene is grainy film, the edit must have film grain).
5. **SPATIAL LOGIC**: Use precise positioning (e.g., "placed directly behind," "resting on the left edge of") to avoid floating or misaligned objects.

# PROMPT GUIDELINES
- **Style Continuity**: Always reaffirm the original medium/style at the end of the prompt to prevent "style drift."
- **Natural Narrative**: Write in fluid, descriptive English sentences. Avoid keyword stuffing or comma-separated lists.
- **Negative Avoidance**: Describe what the new state looks like rather than telling the model what to "remove."
- **No Fluff**: Do not use "masterpiece," "8k," or "ultra-detailed" unless the user specifically asks for a quality boost.

# OUTPUT STRUCTURE
Generate a single paragraph (50-120 words) following this flow:
[Subject/Context Preservation] + [Specific Modification with Physical & Lighting Details] + [Spatial Placement] + [Style/Camera Anchor].

# EXAMPLE
User Request: "Add a black leather jacket to the man"
Improved Output: "The man from the original image maintains his exact pose and facial expression, but is now wearing a premium black leather jacket with a subtle matte finish. The jacket features realistic creases and highlights that match the existing side-lighting of the scene. The original urban alleyway background and the cinematic 35mm film photography style remain untouched, ensuring the jacket looks perfectly integrated into the environment."

# FINAL INSTRUCTION
Output ONLY the final prompt text. No explanations, no greetings, no markdown blocks."""

VIDEO_SYSTEM_PROMPT = """## Role: LTX-2 Motion Engineer
You transform an image and a user request into a high-motion video prompt.

## Rules for LTX-2 Success:
1. NO REDUNDANCY: Do not describe colors or clothes already visible in the image.
2. IMMEDIATE ACTION: Start with the action. Use "is currently [action]" or "[Subject] [verb]s". Never use "starts to" or "begins".
3. MOTION PHYSICS: Describe weight and inertia. Use phrases like "weight shifts," "fabric ripples," or "momentum carries" to give life to the motion.
4. STABLE CAMERA (Default): Unless the user specifies a movement, the camera must remain at a safe, consistent distance, keeping the main subject perfectly centered. Use "Steady tracking shot" or "Fixed medium shot" to maintain focus on the subject's movement without distorting the background.
5. FLOW: One fluid paragraph (max 90 words). Output ONLY the final prompt in English.

## Output Template:
[Immediate Action + Physics] + [Atmospheric/Environmental interaction] + [Steady centered camera shot].

## Examples:
User: "She is dancing"
Target: "The character performs a fluid dance, her body rotating with natural momentum while her hair and clothes react to the centrifugal force. A steady tracking shot keeps her perfectly centered at a medium distance, capturing her full range of motion without any abrupt camera shifts. The lighting remains consistent as she moves."

User: "He is running"
Target: "The man runs forward with powerful strides, his feet pressing into the ground and his arms pumping rhythmically. A stable tracking shot follows him at a constant distance, keeping him locked in the center of the frame. The background blurs slightly to emphasize his speed while maintaining temporal coherence.
"""

# ----------------------------------------------------
# Estado global
# ----------------------------------------------------
base_image_name = None
base_image_path = None

MODE_CONFIG = {
    "image": {
        "system_prompt": IMAGE_SYSTEM_PROMPT,
        "default_label": "🖼️ Generar imagen",
    },
    "video": {
        "system_prompt": VIDEO_SYSTEM_PROMPT,
        "default_label": "🎬 Generate video",
    },
}

# ----------------------------------------------------
# Persistencia del historial (solo URLs de ComfyUI)
# ----------------------------------------------------
def save_chat_history(history):
    """Guarda historial con URLs de ComfyUI (sin descargar archivos)."""
    serializable = []
    for msg in history:
        if not isinstance(msg, dict) or "role" not in msg:
            continue

        meta = msg.get("metadata", {})
        content = msg.get("content")

        # Si tiene metadata con comfy_url, guardar la URL
        if meta and meta.get("comfy_url"):
            serializable.append({
                "role": msg["role"],
                "comfy_url": meta["comfy_url"],
                "media_type": meta.get("media_type", "image")
            })
        # Si es texto, guardarlo
        elif isinstance(content, str):
            serializable.append({
                "role": msg["role"],
                "content": content
            })

    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Historial guardado: {len(serializable)} mensajes (solo URLs)")
    except Exception as e:
        logger.error(f"❌ Error guardando historial: {e}")


def load_chat_history():
    """Carga historial apuntando directamente a URLs de ComfyUI."""
    if not CHAT_HISTORY_FILE.exists():
        return []

    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)

        history = []
        last_image_url = None

        for msg in saved:
            if not isinstance(msg, dict) or "role" not in msg:
                continue

            comfy_url = msg.get("comfy_url")
            media_type = msg.get("media_type")

            # Si tiene URL de ComfyUI, usar dict con path (Gradio puede cargar URLs)
            if comfy_url:
                history.append({
                    "role": msg["role"],
                    "content": {"path": comfy_url},
                    "metadata": {
                        "comfy_url": comfy_url,
                        "media_type": media_type
                    }
                })
                # Rastrear última imagen (no video) para restaurar como base
                if media_type == "image":
                    last_image_url = comfy_url
            # Si es texto
            elif msg.get("content"):
                history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Restaurar última imagen como base (necesitamos descargarla solo para re-upload)
        if last_image_url:
            try:
                resp = requests.get(last_image_url, timeout=10)
                if resp.status_code == 200:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    temp_file.write(resp.content)
                    temp_file.close()
                    upload_to_comfy(temp_file.name)
                    logger.info(f"🔄 Base restaurada: {base_image_name}")
                else:
                    logger.warning(f"⚠️ No se pudo descargar imagen base: HTTP {resp.status_code}")
            except Exception as e:
                logger.error(f"❌ Error restaurando base: {e}")

        logger.info(f"📂 Historial cargado: {len(history)} mensajes")
        return history
    except Exception as e:
        logger.error(f"❌ Error cargando historial: {e}")
        return []


def clear_chat_history():
    """Elimina solo el archivo de historial (no borra archivos de ComfyUI)."""
    if CHAT_HISTORY_FILE.exists():
        os.remove(CHAT_HISTORY_FILE)
    logger.info("🗑️ Historial eliminado (archivos en ComfyUI permanecen)")


# ----------------------------------------------------
# Utilidades
# ----------------------------------------------------
def upload_to_comfy(file_path):
    """Upload image to ComfyUI and update global state."""
    global base_image_name, base_image_path
    with open(file_path, "rb") as f:
        resp = requests.post(
            f"{COMFY_URL}/upload/image",
            files={"image": f},
            data={"overwrite": "true"},
        )
    base_image_name = resp.json()["name"]
    base_image_path = file_path
    logger.info(f"📤 Uploaded to ComfyUI: {base_image_name}")
    return base_image_name


def encode_image_to_base64(file_path, max_size_mb=1.0):
    """Encodes image to Base64, resizing if needed."""
    from PIL import Image
    import io

    target_size_bytes = max_size_mb * 1024 * 1024

    with open(file_path, "rb") as f:
        img_bytes = f.read()

    if len(img_bytes) > target_size_bytes:
        logger.info(f"⏳ Image exceeds {max_size_mb}MB, resizing...")
        img = Image.open(io.BytesIO(img_bytes))

        output_buffer = io.BytesIO()
        img.save(output_buffer, format="JPEG", quality=95)

        quality = 90
        while len(output_buffer.getvalue()) > target_size_bytes and quality > 10:
            output_buffer = io.BytesIO()
            new_width = int(img.width * 0.9)
            new_height = int(img.height * 0.9)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            img.save(output_buffer, format="JPEG", quality=quality)
            quality -= 10

        img_bytes = output_buffer.getvalue()
        logger.info(f"✅ Image resized to {len(img_bytes) / 1024 / 1024:.2f}MB")

    return base64.b64encode(img_bytes).decode("utf-8")


def enhance_prompt_with_llama(user_text, image_path=None, system_prompt=None, mode="image"):
    """Call llama.cpp to enhance prompt."""
    system_prompt = system_prompt or IMAGE_SYSTEM_PROMPT
    default_text = f"Improve this prompt for AI {mode} generation"
    final_text = user_text or default_text

    user_content = [{"type": "text", "text": final_text}]
    has_image = image_path and os.path.exists(image_path)
    if has_image:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(image_path)}"},
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    logger.info("🪄 LLAMA.CPP REQUEST | has_image=%s | text='%s'", has_image, final_text[:80])

    try:
        resp = requests.post(
            f"{LLAMA_CPP_URL}/v1/chat/completions",
            json={"messages": messages, "temperature": 0.7, "max_tokens": 512, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        enhanced = resp.json()["choices"][0]["message"]["content"].strip().replace("```", "")
        logger.info("✨ Enhanced: %s", enhanced[:100])
        return enhanced
    except requests.exceptions.HTTPError as e:
        error_details = e.response.text if e.response else "No details"
        logger.error(f"❌ Llama.cpp HTTP Error {e.response.status_code}: {error_details}")
        raise Exception(f"Llama.cpp regresó error {e.response.status_code}: {error_details}")
    except requests.exceptions.ConnectionError:
        raise Exception(f"llama.cpp no disponible en {LLAMA_CPP_URL}")
    except Exception as e:
        raise Exception(f"Error en llama.cpp: {e}")


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
async def process_image(prompt_text):
    """Generate image with Qwen. Returns local path and ComfyUI URL."""
    global base_image_name, base_image_path

    if not base_image_name:
        raise Exception("No hay imagen base")

    with open("Qwen-Rapid-AIO.json", "r", encoding="utf-8") as f:
        workflow = json.load(f)

    workflow["3"]["inputs"]["prompt"] = prompt_text
    workflow["7"]["inputs"]["image"] = base_image_name
    workflow["2"]["inputs"]["seed"] = int(uuid.uuid4().hex, 16) >> 96

    prompt_id = await submit_and_wait(workflow)
    logger.info("🟢 Image prompt: %s", prompt_id)

    history = requests.get(f"{COMFY_URL}/history/{prompt_id}").json()
    outputs = history.get(prompt_id, {}).get("outputs", {})

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

    # Re-upload como nueva base
    upload_to_comfy(temp_file.name)

    return temp_file.name, comfy_url


async def process_video(prompt_text):
    """Generate video with LTXV. Returns local path and ComfyUI URL."""
    if not base_image_name:
        raise Exception("Se requiere imagen base")

    with open("LTXV-DoAlmostEverything-v3.json", "r", encoding="utf-8") as f:
        workflow = json.load(f)

    workflow["106"]["inputs"]["image"] = base_image_name
    workflow["35"]["inputs"]["image"] = base_image_name
    workflow["59"]["inputs"]["value"] = prompt_text
    workflow["128"]["inputs"]["value"] = int(uuid.uuid4().hex, 16) % (2**32)

    payload = {"prompt": workflow, "client_id": CLIENT_ID}
    response = requests.post(f"{COMFY_URL}/prompt", json=payload).json()
    if "prompt_id" not in response:
        raise Exception(f"No prompt_id: {response}")

    prompt_id = response["prompt_id"]
    logger.info("🎬 Video prompt: %s", prompt_id)

    node_output = await poll_until_output(prompt_id, "17")
    logger.info("🎬 Video completed")

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
    return video_path, comfy_url


GENERATORS = {
    "image": process_image,
    "video": process_video,
}


# ----------------------------------------------------
# Chat function
# ----------------------------------------------------
async def chat_fn(message, history, mode, original_text=None, enhanced_text=None):
    """Main chat orchestrator."""
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

    # Upload new image if provided
    if src_path:
        upload_to_comfy(src_path)
        history.append({
            "role": "user",
            "content": {"path": src_path}
        })

    # Validate base image exists
    if not base_image_name:
        history.append({
            "role": "assistant",
            "content": "⚠️ Se requiere una imagen. Sube una primero."
        })
        save_chat_history(history)
        return history

    # Add user text
    display_text = original_text if original_text is not None else text
    if display_text or not src_path:
        history.append({
            "role": "user",
            "content": display_text or config["default_label"]
        })

    try:
        # Generate (retorna path local + URL de ComfyUI)
        result_path, comfy_url = await GENERATORS[mode](text)
        logger.info("%s OK: %s | URL: %s", mode.upper(), result_path, comfy_url)

        # Show enhanced prompt if applicable
        if enhanced_text and enhanced_text != original_text:
            history.append({
                "role": "assistant",
                "content": f"✨ *Prompt mejorado:* {enhanced_text}"
            })

        # Add result con path local (para mostrar) y metadata con URL (para persistencia)
        history.append({
            "role": "assistant",
            "content": {"path": result_path},
            "metadata": {
                "comfy_url": comfy_url,
                "media_type": mode
            }
        })

    except Exception as e:
        logger.exception("❗ Error:")
        history.append({
            "role": "assistant",
            "content": f"Error procesando {mode}: {e}"
        })

    save_chat_history(history)
    return history


# ----------------------------------------------------
# Handler
# ----------------------------------------------------
async def handle_generation(message, history, auto_enhance_checked, mode):
    """Unified handler with optional enhancement."""
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

    if auto_enhance_checked:
        logger.info("🤖 Auto-enhancing for %s...", mode)
        try:
            image_for_enhance = src_path or base_image_path
            enhanced_prompt = enhance_prompt_with_llama(
                user_text, image_for_enhance, MODE_CONFIG[mode]["system_prompt"], mode
            )
            prompt_to_use = enhanced_prompt
        except Exception as e:
            logger.error("❌ Enhance failed: %s", e)

    gen_message = {"text": prompt_to_use, "files": files}
    return await chat_fn(gen_message, history, mode, user_text, enhanced_prompt)


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
    )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        placeholder="Enter message and upload image...",
        show_label=False,
    )

    with gr.Row():
        auto_enhance = gr.Checkbox(label="🤖 Mejorar con IA", value=False, scale=1)
        btn_image = gr.Button("Generar Imagen 🖼️", variant="primary", scale=3)
        btn_video = gr.Button("Generar Video 🎬", variant="primary", scale=3)
        btn_clear = gr.Button("🗑️ Limpiar", variant="secondary", scale=2)

    # Wire buttons (sin localStorage, solo guardado automático)
    def wire(trigger, mode):
        def sync_handler(msg, hist, enhance):
            return asyncio.run(handle_generation(msg, hist, enhance, mode))

        trigger.click(
            fn=sync_handler,
            inputs=[chat_input, chatbot, auto_enhance],
            outputs=[chatbot],
        ).then(
            fn=lambda: gr.update(value=None),
            outputs=[chat_input]
        )

    wire(btn_image, "image")
    wire(btn_video, "video")

    # Enter key → imagen
    def sync_image_handler(msg, hist, enhance):
        return asyncio.run(handle_generation(msg, hist, enhance, "image"))

    chat_input.submit(
        fn=sync_image_handler,
        inputs=[chat_input, chatbot, auto_enhance],
        outputs=[chatbot],
    ).then(
        fn=lambda: gr.update(value=None),
        outputs=[chat_input]
    )

    # Clear button
    def clear_all():
        global base_image_name, base_image_path
        base_image_name = None
        base_image_path = None
        clear_chat_history()
        return []

    btn_clear.click(
        fn=clear_all,
        outputs=[chatbot],
    )

if __name__ == "__main__":
    demo.launch()