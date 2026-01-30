import gradio as gr
import json
import requests
import asyncio
import aiohttp
import uuid
import os
import tempfile
import logging

# Configuración de red
SERVER_ADDR = "127.0.0.1:8188"
COMFY_URL = f"http://{SERVER_ADDR}"
WS_URL = f"ws://{SERVER_ADDR}/ws"
CLIENT_ID = str(uuid.uuid4())

# ----------------------------------------------------
# 0️⃣ Set up a logger (mirrors the logger used in run.py)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------
# 0️⃣ Global state – keeps the **current base image name** that will be
#     used for the next processing step.  Updated when a NEW image is
#     uploaded AND after each successful generation.
base_image_name = None  # e.g. "ComfyUI_temp_xxxxx.png"

def upload_to_comfy(file_path):
    """Upload a new image and make it the current base image."""
    with open(file_path, "rb") as f:
        files = {"image": f}
        resp = requests.post(
            f"{COMFY_URL}/upload/image", files=files, data={"overwrite": "true"}
        )
        name = resp.json()["name"]
    # ----- update global base image ------------------------------------
    global base_image_name
    base_image_name = name
    return name

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
# 1️⃣ Modified **process_image** – after generating an image, re-uploads it
#     to ComfyUI so it becomes the new base for subsequent edits.
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
    workflow["2"]["inputs"]["seed"] = uuid.uuid4().int >> 96

    # ---- 4️⃣ Send prompt ------------------------------------------------
    payload = {"prompt": workflow, "client_id": CLIENT_ID}
    logger.debug("Sending prompt payload to %s/prompt", COMFY_URL)
    response = requests.post(f"{COMFY_URL}/prompt", json=payload).json()
    logger.debug("Prompt response: %s", response)

    if "prompt_id" not in response:
        raise Exception(
            f"ComfyUI did not return a prompt_id. Full response: {response}"
        )
    prompt_id = response["prompt_id"]
    logger.info("🟢 Prompt submitted, id=%s", prompt_id)

    await wait_for_completion(prompt_id)

    # ---- 5️⃣ Retrieve result --------------------------------------------
    history_res = requests.get(f"{COMFY_URL}/history/{prompt_id}").json()
    output_info = history_res[prompt_id]["outputs"]["6"]["images"][0]

    # ---- 6️⃣ Download generated image ------------------------------------
    view_url = f"{COMFY_URL}/view?filename={output_info['filename']}&type={output_info['type']}"
    img_data = requests.get(view_url).content

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.write(img_data)
    temp_file.close()

    # ---- 7️⃣ Re-upload the generated image to make it the new base ------
    # This ensures base_image_name always points to a file ComfyUI can access
    base_image_name = upload_to_comfy(temp_file.name)

    return temp_file.name

# ----------------------------------------------------
# 2️⃣ Updated **chat_fn** – adds the original unprocessed image to history
#     so users can refer back to it or reuse it later. Then continues normally.
async def chat_fn(message, history):
    text = message.get("text", "")
    files = message.get("files", []) or []

    src_path = files[0] if files else None
    if isinstance(src_path, tuple):
        src_path = src_path[0]

    # Store original image in chat history for reference/reuse
    if src_path:
        history.append({"role": "user", "content": gr.Image(src_path)})

    try:
        local_img_path = await process_image(text, src_path)

        # Append user's text prompt and assistant's processed image
        history.append({"role": "user", "content": text})
        history.append(
            {"role": "assistant", "content": gr.Image(local_img_path)}
        )

        return history, gr.update(value=None)

    except Exception as e:
        logger.exception("❗ Error processing image:")
        history.append(
            {"role": "assistant", "content": f"Error procesando imagen: {e}"}
        )
        return history, gr.update(value=None)

# ----------------------------------------------------
# 3️⃣ UI – unchanged apart from using MultimodalTextbox
with gr.Blocks() as demo:
    gr.Markdown("## 💬 Nunchaku Qwen Chat")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=550)

            chat_input = gr.MultimodalTextbox(
                interactive=True,
                placeholder="Enter message or upload file...",
                show_label=False,
            )
        # No separate Image component needed

    # Events
    chat_input.submit(
        fn=lambda msg, hist: asyncio.run(chat_fn(msg, hist)),
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input],
    )
    btn_send = gr.Button("Enviar 🚀")
    btn_send.click(
        fn=lambda msg, hist: asyncio.run(chat_fn(msg, hist)),
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input],
    )

# ----------------------------------------------------
if __name__ == "__main__":
    demo.launch()