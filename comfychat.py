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

# Configuración de red
SERVER_ADDR = "127.0.0.1:8188"
COMFY_URL = f"http://{SERVER_ADDR}"
WS_URL = f"ws://{SERVER_ADDR}/ws"
CLIENT_ID = str(uuid.uuid4())

# Configuración de persistencia
CHAT_HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.json")

# Configuración de llama.cpp
LLAMA_CPP_URL = "http://127.0.0.1:8080"  # URL por defecto de llama.cpp server

# System prompt para mejorar prompts de IMAGEN (Qwen Image Edit, Flux Kontext, etc.)
IMAGE_SYSTEM_PROMPT = """You are an expert Prompt Engineer specializing in image editing and variation models like Qwen Image Edit and Flux Kontext. Your sole purpose is to analyze input images and user edit requests, then generate optimized, high-quality prompts that produce precise, consistent results.

## Core Responsibilities:
1. **Analyze** the input image: identify subjects, style, lighting, composition, colors, textures, and mood.
2. **PRIORITIZE User Intent**: The user's request is paramount. Follow their instructions exactly as stated. Do not override or ignore the user's explicit intent.
3. **Interpret** the user's edit request: understand what must change vs. what must remain identical. Respect the user's creative vision above all.
4. **Synthesize** an optimized prompt that maintains visual consistency while applying the requested modification.

## Prompt Guidelines:
- **Context Preservation**: Explicitly mention elements that must stay unchanged (identity, pose, background elements, lighting style) to prevent unwanted drift.
- **Precise Editing**: Describe the requested change with specific visual details (colors, materials, positions, styles) rather than vague concepts.
- **Natural Language**: Use flowing, descriptive sentences rather than tag lists or comma-separated keywords. The model understands semantic context better than keyword stuffing.
- **Style Anchoring**: If the image has a distinct aesthetic (photorealistic, anime, oil painting, cinematic), reaffirm it in the prompt to maintain coherence.
- **Negative Space**: Avoid describing what to remove; instead, describe what replaces it or the new desired state.

## Output Format:
Provide ONLY the final prompt. No code blocks, no explanations, no greetings.

**Structure:**
[Subject description maintaining identity], [current context/setting], [specific requested modification with visual details], [preserved stylistic elements: lighting, texture, artistic style], [camera angle/composition if relevant]

## Examples of Good vs. Bad Prompts:
- **Bad**: "make it red car, same background, realistic"
- **Good**: "A metallic crimson red sports car with glossy finish, parked on the exact same cobblestone street with the warm golden hour sunlight and soft shadows maintaining the photorealistic cinematic photography style, keeping the background buildings and atmospheric perspective identical to the original"

## Constraints:
- Never add artistic flourishes not requested (e.g., don't add "dramatic" or "8k" unless specified).
- If the user requests a style change (e.g., "turn this photo into anime"), explicitly mention that the original composition and subject identity are preserved while applying the new medium/style.
- Keep prompts between 50-150 words for optimal token balance.

Output ONLY the improved prompt text. Do not include explanations, greetings, or markdown formatting."""

# System prompt para mejorar prompts de VIDEO (LTXV, etc.)
VIDEO_SYSTEM_PROMPT = """You are an expert Cinematic Prompt Engineer specializing in **LTXV** (Lightricks Text-to-Video), a high-speed flow-matching video generation model. Your purpose is to analyze static input images and user requests, then generate optimized video prompts that create smooth, temporally coherent 5-second video clips.

## CRITICAL RULE - USER ACTION IS PRIORITY:
The user's requested action MUST dominate the entire prompt and occur THROUGHOUT the video from the very first frame of motion. LTXV has only 5 seconds. If you describe static setup first, LTXV will spend most frames on stillness and the action will barely appear. The action must be described as something that IS HAPPENING, not something that WILL happen.

## Core Responsibilities:
1. **PRIORITIZE User Intent ABOVE ALL**: The user's requested action is the single most important element. Build everything else around it.
2. **Analyze** the input image: identify subjects, environment, lighting, style — but only to SUPPORT the action, not to replace it.
3. **Immediate Motion**: The action must start immediately. Do not describe a static starting state before the action.
4. **Engineer Camera Work**: Specify camera movements that complement and FOLLOW the subject action.

## LTXV Prompt Structure (Action-First):
[User's requested action happening NOW, actively, continuously], [subject and visual context from image], [camera movement following the action], [atmospheric details], [quality and consistency anchors]

## CRITICAL Words to AVOID (these make LTXV delay the action):
- "then begins to..." / "starts to..." / "begins..."
- "gradually" / "slowly building" / "from stillness"
- "stands poised at frame start" / "static at the beginning"
- "transitioning from X to Y" (implies X happens first, eating up frames)

## Words that WORK (these make LTXV execute the action immediately):
- "actively" / "continuously" / "throughout the entire clip"
- "already in motion" / "mid-action" / "actively performing"
- "from the very first frame" / "the entire duration"
- "keeps walking" / "continues moving" / "sustained motion"

## Prompt Guidelines:
- **Action Dominance**: The user's action should occupy 60%+ of the prompt words. Everything else is secondary.
- **Present Tense, Active Voice**: Describe motion as currently happening. "She walks" not "She begins to walk".
- **Repetition is OK**: Reinforce the action multiple times in different ways. LTXV responds to emphasis.
- **Camera follows action**: If subject moves right, camera tracks right. Don't describe camera doing its own thing.
- **Preserve identity and style**: Mention the subject's appearance and the image's visual style, but briefly.

## Output Format:
Provide ONLY the final prompt in a single paragraph (80-120 words). No explanations, no greetings.

## Example Transformation:

**User Input Image:** Portrait of woman in a blue dress standing in a garden
**User Request:** "ella va caminando"

**BAD (what NOT to generate — action delayed to end):**
"A woman in a flowing blue dress stands poised at frame start, then begins walking with graceful, measured steps, fabric swaying naturally with each stride. Slow cinematic tracking shot from medium distance. Gentle momentum building from stillness to fluid walk."

**GOOD (action-first, immediate, dominant):**
"The woman in the blue dress walks forward with confident, steady steps, actively moving throughout the entire clip. Her legs move continuously, arms swing naturally, and the blue fabric flows with each stride as she moves along the garden path. A smooth tracking shot follows her forward motion from a medium angle, keeping her centered. Continuous, fluid walking motion sustained for the full duration, photorealistic quality, consistent lighting and identity throughout."
"""

# Directorio persistente para medios generados (imágenes y videos)
MEDIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_media")
os.makedirs(MEDIA_DIR, exist_ok=True)

# Directorio temporal para procesamiento (evita que se borren antes de reproducir)
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "comfychat_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# 0️⃣ Set up a logger (mirrors the logger used in run.py)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------
# 0️⃣ Persistencia del historial del chat
# ----------------------------------------------------
def save_chat_history(history):
    """Guarda el historial del chat en un archivo JSON, usando URLs de ComfyUI para medios"""
    try:
        serializable_history = []
        for msg in history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                content = msg["content"]
                metadata = msg.get("metadata", {})
                # Guardar mensajes de texto
                if isinstance(content, str):
                    serializable_history.append({
                        "role": msg["role"],
                        "content": content
                    })
                # Guardar URLs de ComfyUI para imágenes/videos (desde metadata)
                elif metadata and metadata.get("comfy_url"):
                    serializable_history.append({
                        "role": msg["role"],
                        "content": None,
                        "comfy_url": metadata["comfy_url"],
                        "media_type": metadata.get("media_type", "unknown")
                    })
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Historial guardado: {len(serializable_history)} mensajes (con URLs de ComfyUI)")
    except Exception as e:
        logger.error(f"❌ Error guardando historial: {e}")

def load_chat_history():
    """Carga el historial del chat desde el archivo JSON, descargando archivos de ComfyUI"""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                saved_history = json.load(f)
            # Reconstruir el historial descargando archivos a temporal
            history = []
            for msg in saved_history:
                if isinstance(msg, dict) and "role" in msg:
                    comfy_url = msg.get("comfy_url")
                    media_type = msg.get("media_type")
                    # Si tiene comfy_url, descargar a archivo temporal
                    if comfy_url:
                        try:
                            response = requests.get(comfy_url, timeout=30)
                            if response.status_code == 200:
                                ext = ".mp4" if media_type == "video" else ".png"
                                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                                temp_file.write(response.content)
                                temp_file.close()
                                temp_path = temp_file.name
                                history.append({
                                    "role": msg["role"],
                                    "content": {"path": temp_path},
                                    "metadata": {"comfy_url": comfy_url, "media_type": media_type}
                                })
                                logger.info(f"📥 Descargado: {comfy_url}")
                            else:
                                logger.warning(f"⚠️ HTTP {response.status_code}: {comfy_url}")
                                history.append({
                                    "role": msg["role"],
                                    "content": f"[Media no disponible]"
                                })
                        except Exception as e:
                            logger.error(f"❌ Error descargando: {e}")
                            history.append({
                                "role": msg["role"],
                                "content": f"[Media no disponible]"
                            })
                    elif msg.get("content"):  # Mensaje de texto
                        history.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
            logger.info(f"📂 Historial cargado: {len(history)} mensajes")
            return history
    except Exception as e:
        logger.error(f"❌ Error cargando historial: {e}")
    return []

def clear_chat_history():
    """Limpia el archivo de historial del chat"""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            os.remove(CHAT_HISTORY_FILE)
            logger.info("🗑️ Historial eliminado")
        # También limpiar archivos temporales de media
        import shutil
        if os.path.exists(MEDIA_DIR):
            shutil.rmtree(MEDIA_DIR)
            os.makedirs(MEDIA_DIR, exist_ok=True)
            logger.info("🗑️ Directorio de media limpiado")
    except Exception as e:
        logger.error(f"❌ Error eliminando historial: {e}")

# ----------------------------------------------------
# 0️⃣ Global state – keeps the **current base image name** that will be
#     used for the next processing step.  Updated when a NEW image is
#     uploaded AND after each successful generation.
base_image_name = None  # e.g. "ComfyUI_temp_xxxxx.png"
base_image_path = None  # Local file path to the base image (for llama.cpp enhancement)

def upload_to_comfy(file_path):
    """Upload a new image and make it the current base image."""
    with open(file_path, "rb") as f:
        files = {"image": f}
        resp = requests.post(
            f"{COMFY_URL}/upload/image", files=files, data={"overwrite": "true"}
        )
        name = resp.json()["name"]
    # ----- update global base image ------------------------------------
    global base_image_name, base_image_path
    base_image_name = name
    base_image_path = file_path  # Save local path for llama.cpp enhancement
    return name


def encode_image_to_base64(file_path):
    """Encode image file to base64 for llama.cpp vision models."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def enhance_prompt_with_llama(user_text, image_path=None, system_prompt=None, mode="image"):
    """
    Call llama.cpp to enhance the user's prompt using a vision model.
    Returns the enhanced prompt text.

    Args:
        user_text: The user's prompt text
        image_path: Optional path to an image for vision analysis
        system_prompt: The system prompt to use (IMAGE_SYSTEM_PROMPT or VIDEO_SYSTEM_PROMPT)
        mode: "image" or "video" - used for fallback text when user_text is empty
    """
    if system_prompt is None:
        system_prompt = IMAGE_SYSTEM_PROMPT

    # Build the messages payload
    messages = [{"role": "system", "content": system_prompt}]

    # Build user message with text and optional image
    # NOTE: Text goes FIRST, then image - this helps the model focus on user instructions
    user_content = []

    # Add text FIRST (so model reads instructions before seeing image)
    # Dynamic fallback based on explicit mode parameter
    default_text = "Improve this prompt for AI video generation" if mode == "video" else "Improve this prompt for AI image generation"
    final_user_text = user_text if user_text else default_text
    user_content.append({"type": "text", "text": final_user_text})

    # Add image AFTER text
    has_image = image_path and os.path.exists(image_path)
    if has_image:
        # Encode image to base64
        img_base64 = encode_image_to_base64(image_path)
        # llama.cpp vision format (llava style)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
        })

    messages.append({"role": "user", "content": user_content})

    # Prepare the request payload for llama.cpp
    payload = {
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False
    }

    # Debug logging - show what we're sending
    logger.info("=" * 60)
    logger.info("🪄 LLAMA.CPP REQUEST")
    logger.info("=" * 60)
    logger.info(f"URL: {LLAMA_CPP_URL}/v1/chat/completions")
    logger.info(f"Has image: {has_image}")
    logger.info(f"Image path: {image_path}")
    logger.info(f"User text: '{user_text}'")
    logger.info(f"Final user text sent: '{final_user_text}'")
    logger.info(f"System prompt: {system_prompt[:100]}...")
    logger.info("Messages structure:")
    for i, msg in enumerate(messages):
        content_preview = str(msg.get('content', ''))[:200]
        logger.info(f"  [{i}] role={msg.get('role')}, content={content_preview}...")
    logger.info("=" * 60)

    try:
        response = requests.post(
            f"{LLAMA_CPP_URL}/v1/chat/completions",
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        result = response.json()

        # Debug logging - show the full response
        logger.info("=" * 60)
        logger.info("📥 LLAMA.CPP RESPONSE")
        logger.info("=" * 60)
        logger.info(f"Full response: {json.dumps(result, indent=2)}")
        logger.info("=" * 60)

        enhanced_prompt = result["choices"][0]["message"]["content"].strip()

        # Clean up any markdown formatting that might have slipped through
        enhanced_prompt = enhanced_prompt.replace("```", "").strip()

        logger.info(f"✨ Enhanced prompt result: {enhanced_prompt}")
        return enhanced_prompt

    except requests.exceptions.ConnectionError:
        logger.error(f"❌ Could not connect to llama.cpp at {LLAMA_CPP_URL}")
        raise Exception(f"llama.cpp server not available at {LLAMA_CPP_URL}. Please start the server with a vision model.")
    except Exception as e:
        logger.error(f"❌ Error calling llama.cpp: {e}")
        raise Exception(f"Error enhancing prompt with llama.cpp: {str(e)}")

async def wait_for_completion(prompt_id):
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
                        return True
    return False

# ----------------------------------------------------
# 1️⃣ Process IMAGE workflow (Qwen-Rapid-AIO.json)
async def process_image(prompt_text, source_path=None):
    """
    *prompt_text* – text prompt (may be empty).
    *source_path* – path to a NEW image supplied by the user.
    If *source_path* is None we reuse the current ``base_image_name``.
    After successful generation, the produced image is re-uploaded to ComfyUI
    and becomes the new base for the next call.
    """
    global base_image_name

    # ---- 1️⃣ If a new image is provided, upload it and set it as base ----
    if source_path:
        base_image_name = upload_to_comfy(source_path)

    # ---- 2️⃣ Ensure we have a base image to work with --------------------
    if not base_image_name:
        raise Exception("No base image available – upload an image first.")

    # ---- 3️⃣ Load and prepare workflow ----------------------------------
    with open("Qwen-Rapid-AIO.json", "r", encoding="utf-8") as f:
        workflow = json.load(f)

    workflow["3"]["inputs"]["prompt"] = prompt_text
    workflow["7"]["inputs"]["image"] = base_image_name
    workflow["2"]["inputs"]["seed"] = int(uuid.uuid4().hex, 16) >> 96

    # ---- 4️⃣ Send prompt ------------------------------------------------
    payload = {"prompt": workflow, "client_id": CLIENT_ID}
    logger.debug("Sending IMAGE prompt payload to %s/prompt", COMFY_URL)
    response = requests.post(f"{COMFY_URL}/prompt", json=payload).json()
    logger.debug("Prompt response: %s", response)

    if "prompt_id" not in response:
        raise Exception(
            f"ComfyUI did not return a prompt_id. Full response: {response}"
        )
    prompt_id = response["prompt_id"]
    logger.info("🟢 Image Prompt submitted, id=%s", prompt_id)

    completed = await wait_for_completion(prompt_id)
    if not completed:
        raise Exception("Timeout waiting for image generation (WebSocket closed)")

    # ---- 5️⃣ Retrieve result --------------------------------------------
    history_res = requests.get(f"{COMFY_URL}/history/{prompt_id}").json()
    prompt_data = history_res.get(prompt_id, {})
    if "outputs" not in prompt_data or "6" not in prompt_data["outputs"]:
        raise Exception(f"Image generation completed but no output found in node 6. Available outputs: {list(prompt_data.get('outputs', {}).keys())}")
    output_info = prompt_data["outputs"]["6"]["images"][0]

    # ---- 6️⃣ Construir URL de ComfyUI (el archivo ya está guardado ahí) ----
    comfy_url = f"{COMFY_URL}/view?filename={output_info['filename']}&type={output_info['type']}"

    # ---- 7️⃣ Descargar temporalmente para re-upload y llama.cpp ---------
    img_response = requests.get(comfy_url)
    if img_response.status_code != 200:
        raise Exception(f"Failed to download image: HTTP {img_response.status_code}")
    img_data = img_response.content

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.write(img_data)
    temp_file.close()

    # Re-upload the generated image to make it the new base
    base_image_name = upload_to_comfy(temp_file.name)

    # Update base_image_path for llama.cpp enhancement
    global base_image_path
    base_image_path = temp_file.name

    return temp_file.name, "image", comfy_url

# ----------------------------------------------------
# 2️⃣ Process VIDEO workflow (LTXV-DoAlmostEverything-v3.json) - FIXED
async def process_video(prompt_text, source_path=None):
    """
    LTXV Video generation - REQUIRES an image (First Image node 106).
    Uses source_path if provided, otherwise falls back to base_image_name.
    """
    global base_image_name

    if source_path:
        # Use the newly uploaded image
        uploaded_name = upload_to_comfy(source_path)
        base_image_name = uploaded_name
        logger.info(f"Uploaded new image for LTXV: {uploaded_name}")
    elif base_image_name:
        # Use the existing base image
        uploaded_name = base_image_name
        logger.info(f"Using existing base image for LTXV: {uploaded_name}")
    else:
        raise Exception("Video generation requires an image upload (First Image).")

    with open("LTXV-DoAlmostEverything-v3.json", "r", encoding="utf-8") as f:
        workflow = json.load(f)

    workflow["106"]["inputs"]["image"] = uploaded_name
    workflow["35"]["inputs"]["image"] = uploaded_name
    workflow["59"]["inputs"]["value"] = prompt_text
    workflow["128"]["inputs"]["value"] = int(uuid.uuid4().hex, 16) % (2**32)

    payload = {"prompt": workflow, "client_id": CLIENT_ID}
    logger.debug("Sending VIDEO prompt payload to %s/prompt", COMFY_URL)
    response = requests.post(f"{COMFY_URL}/prompt", json=payload).json()
    logger.debug("Prompt response: %s", response)

    if "prompt_id" not in response:
        raise Exception(f"ComfyUI did not return a prompt_id. Response: {response}")
    prompt_id = response["prompt_id"]
    logger.info("🎬 Video Prompt submitted, id=%s", prompt_id)

    # Esperar a que termine usando polling (más confiable para VHS)
    max_retries = 300  # 10 minutos máximo (300 * 2 segundos)
    retries = 0
    history_res = None

    while retries < max_retries:
        history_res = requests.get(f"{COMFY_URL}/history/{prompt_id}").json()

        if prompt_id in history_res:
            # Verificar que el prompt haya terminado sin errores
            status_data = history_res[prompt_id]
            if status_data.get("status", {}).get("status_str") == "error":
                raise Exception(f"ComfyUI reported an error processing the video. Check ComfyUI logs.")

            # Verificar que el output del nodo 17 exista
            outputs = status_data.get("outputs", {})
            if "17" in outputs:
                logger.info("🎬 Video processing completed!")
                break
            else:
                # El prompt terminó pero no hay output del nodo 17
                raise Exception(f"Video generation completed but no output found in node 17. Available outputs: {list(outputs.keys())}")

        logger.debug(f"Waiting for video processing... (retry {retries + 1}/{max_retries})")
        await asyncio.sleep(2)
        retries += 1
    else:
        # Timeout alcanzado
        raise Exception(f"Timeout waiting for video generation after {max_retries * 2} seconds")

    # Retrieve video from node 17 (VHS_VideoCombine)
    node_output = history_res[prompt_id]["outputs"]["17"]
    logger.debug(f"Node 17 output keys: {node_output.keys()}")

    # VHS_VideoCombine guarda videos en 'gifs' aunque sean MP4
    if "gifs" in node_output and len(node_output["gifs"]) > 0:
        output_info = node_output["gifs"][0]
    elif "images" in node_output and len(node_output["images"]) > 0:
        output_info = node_output["images"][0]
    else:
        raise Exception(f"No video output found in node 17. Keys: {node_output.keys()}")

    filename = output_info["filename"]
    subfolder = output_info.get("subfolder", "")
    file_type = output_info.get("type", "output")

    logger.info(f"Video from ComfyUI: filename={filename}, subfolder={subfolder}, type={file_type}")

    # Construir URL de ComfyUI (el video ya está guardado ahí)
    comfy_url = f"{COMFY_URL}/view?filename={filename}&type={file_type}&subfolder={subfolder}"
    logger.debug(f"Video URL: {comfy_url}")

    # Descargar temporalmente para reproducción local
    video_response = requests.get(comfy_url)
    if video_response.status_code != 200:
        raise Exception(f"Failed to download video: HTTP {video_response.status_code}")
    video_data = video_response.content

    if len(video_data) == 0:
        raise Exception("Downloaded video is empty (0 bytes)")

    # Guardar en ubicación temporal para reproducción
    timestamp = int(time.time())
    safe_filename = filename.replace("/", "_").replace("\\", "_")
    video_path = os.path.join(OUTPUT_DIR, f"video_{prompt_id[:8]}_{timestamp}_{safe_filename}")

    with open(video_path, "wb") as f:
        f.write(video_data)

    file_size = os.path.getsize(video_path)
    logger.info(f"Video cached to {video_path}, size: {file_size} bytes")

    if file_size == 0:
        raise Exception("Saved video file is empty")

    return video_path, "video", comfy_url

# ----------------------------------------------------
# 4️⃣ Updated chat_fn – handles both image and video modes
async def chat_fn(message, history, mode, original_text=None, enhanced_text=None):
    """
    mode: "image" or "video"
    original_text: the user's original prompt (shown in chat if enhanced)
    enhanced_text: the AI-enhanced prompt (shown in chat if different from original)
    """
    text = message.get("text", "")
    files = message.get("files", []) or []

    src_path = files[0] if files else None
    if isinstance(src_path, tuple):
        src_path = src_path[0]

    # Validate that we have an image for generation (either attached or base image exists)
    if not src_path and not base_image_name:
        history.append({
            "role": "assistant",
            "content": "⚠️ Se requiere una imagen. Sube una primero."
        })
        return history

    # Determine what to show in chat
    display_text = original_text if original_text is not None else text

    # Store original image and text in chat history BEFORE processing (so it appears even if generation fails)
    if src_path:
        history.append({
            "role": "user",
            "content": gr.Image(
                src_path
            )
        })

    # Add user text to history (appears even if generation fails)
    if mode == "video":
        history.append({"role": "user", "content": display_text if display_text else "🎬 Generate video"})
    else:
        history.append({"role": "user", "content": display_text if display_text else "🖼️ Generar imagen"})

    try:
        comfy_url = None
        if mode == "video":
            result_path, result_type, comfy_url = await process_video(text, src_path)
            logger.info(f"Returning video path: {result_path}, ComfyURL: {comfy_url}")

            # If enhanced, show it as assistant message before the result
            if enhanced_text and enhanced_text != original_text:
                history.append({
                    "role": "assistant",
                    "content": f"✨ *Prompt mejorado:* {enhanced_text}"
                })

            # Guardar la URL de ComfyUI en metadata del mensaje
            history.append({
                "role": "assistant",
                "content": gr.Video(value=result_path),
                "metadata": {"comfy_url": comfy_url, "media_type": "video"}
            })
        else:  # mode == "image"
            result_path, result_type, comfy_url = await process_image(text, src_path)
            logger.info(f"Returning image path: {result_path}, ComfyURL: {comfy_url}")

            # If enhanced, show it as assistant message before the result
            if enhanced_text and enhanced_text != original_text:
                history.append({
                    "role": "assistant",
                    "content": f"✨ *Prompt mejorado:* {enhanced_text}"
                })

            # Guardar la URL de ComfyUI en metadata del mensaje
            history.append({
                "role": "assistant",
                "content": gr.Image(value=result_path),
                "metadata": {"comfy_url": comfy_url, "media_type": "image"}
            })

        # Guardar historial después de cada interacción exitosa
        save_chat_history(history)
        return history

    except Exception as e:
        logger.exception("❗ Error processing %s:", mode)
        history.append(
            {"role": "assistant", "content": f"Error procesando {mode}: {str(e)}"}
        )
        # Guardar historial incluso si hay error
        save_chat_history(history)
        return history

# ----------------------------------------------------
# 5️⃣ UI with three buttons (including magic wand)
with gr.Blocks(fill_width=True, fill_height=True) as demo:
    gr.Markdown("## 💬 ComfyUI Multi-Modal Chat (Image + Video)")

    chatbot = gr.Chatbot(
        label="QwenAI",
        height="75vh",
        elem_id="chatbot",
        autoscroll=True,
        value=load_chat_history(),  # Cargar historial al iniciar
    )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        placeholder="Enter message and upload image...",
        show_label=False,
    )

    with gr.Row():
        auto_enhance = gr.Checkbox(label="🤖 Mejorar con IA", value=False, scale=1)
        btn_send = gr.Button("Generar Imagen 🖼️ (default)", variant="primary", scale=3)
        btn_video = gr.Button("Generar Video 🎬", variant="primary", scale=3)
        btn_clear = gr.Button("🗑️ Limpiar Chat", variant="secondary", scale=2)

    # Image processing with optional auto-enhance
    async def process_image_with_enhance(message, history, auto_enhance_checked):
        """Enhance prompt if checked, then generate image."""
        user_text = message.get("text", "") if message else ""
        files = (message.get("files", []) or []) if message else []
        src_path = files[0] if files else None
        if isinstance(src_path, tuple):
            src_path = src_path[0]

        prompt_to_use = user_text
        enhanced_prompt = None

        if auto_enhance_checked:
            logger.info("🤖 Auto-enhancing prompt for image generation...")
            try:
                image_for_enhance = src_path if src_path else base_image_path
                enhanced_prompt = enhance_prompt_with_llama(user_text, image_for_enhance, IMAGE_SYSTEM_PROMPT, mode="image")
                prompt_to_use = enhanced_prompt
                logger.info(f"✨ Enhanced prompt: {enhanced_prompt[:100]}...")
            except Exception as e:
                logger.error(f"❌ Failed to enhance prompt: {e}")

        gen_message = {"text": prompt_to_use, "files": files}
        return await chat_fn(gen_message, history, "image", user_text, enhanced_prompt)

    # Video processing with optional auto-enhance
    async def process_video_with_enhance(message, history, auto_enhance_checked):
        """Enhance prompt if checked, then generate video."""
        user_text = message.get("text", "") if message else ""
        files = (message.get("files", []) or []) if message else []
        src_path = files[0] if files else None
        if isinstance(src_path, tuple):
            src_path = src_path[0]

        prompt_to_use = user_text
        enhanced_prompt = None

        if auto_enhance_checked:
            logger.info("🤖 Auto-enhancing prompt for video generation...")
            try:
                image_for_enhance = src_path if src_path else base_image_path
                enhanced_prompt = enhance_prompt_with_llama(user_text, image_for_enhance, VIDEO_SYSTEM_PROMPT, mode="video")
                prompt_to_use = enhanced_prompt
                logger.info(f"✨ Enhanced prompt: {enhanced_prompt[:100]}...")
            except Exception as e:
                logger.error(f"❌ Failed to enhance prompt: {e}")

        gen_message = {"text": prompt_to_use, "files": files}
        return await chat_fn(gen_message, history, "video", user_text, enhanced_prompt)

    # Image button
    btn_send.click(
        fn=process_image_with_enhance,
        inputs=[chat_input, chatbot, auto_enhance],
        outputs=[chatbot],
    ).then(
        fn=lambda: gr.update(value=None),
        outputs=[chat_input],
    )

    # Video button
    btn_video.click(
        fn=process_video_with_enhance,
        inputs=[chat_input, chatbot, auto_enhance],
        outputs=[chatbot],
    ).then(
        fn=lambda: gr.update(value=None),
        outputs=[chat_input],
    )

    # Enter key - generate image
    chat_input.submit(
        fn=process_image_with_enhance,
        inputs=[chat_input, chatbot, auto_enhance],
        outputs=[chatbot],
    ).then(
        fn=lambda: gr.update(value=None),
        outputs=[chat_input],
    )

    # Clear chat button
    def clear_chat():
        clear_chat_history()
        return []  # Retorna historial vacío

    btn_clear.click(
        fn=clear_chat,
        outputs=[chatbot],
    )

# ----------------------------------------------------
if __name__ == "__main__":
    demo.launch()