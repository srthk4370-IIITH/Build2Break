# Python module for PDF -> 1-2 min Video pipeline (orchestrator)
# Writes the script to /mnt/data/pdf_to_video_pipeline.py for you to download and use.
# NOTE: This is a template and orchestrator. You must set real API keys as environment variables
# and install required packages. The environment running this notebook has no internet access,
# so the API calls will fail here; this file is intended to be taken and run in your machine / server.
#
# Requirements (pip):
# pip install requests jsonschema pdfplumber python-dotenv tqdm
# ffmpeg must be installed on the host (ffmpeg binary accessible in PATH).
#
# Environment variables expected (set in your shell or .env):
# - ADOBE_CLIENT_ID, ADOBE_CLIENT_SECRET
# - HUGGINGFACE_API_KEY   (for summarization & flan-t5 calls if using HF Inference)
# - JSON2VIDEO_API_KEY
# - ELEVENLABS_API_KEY
#
# Save this file and follow README instructions printed at the end of execution in this cell.

import os
import re
import json
import time
import math
import logging
import subprocess
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

import requests
import pdfplumber
from jsonschema import validate, ValidationError
from tqdm import tqdm

# ----------------------------- Configuration ---------------------------------

# Timeouts and retry policy
HTTP_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # multiply after each retry

# API Endpoints (placeholders)
ADOBE_EXTRACT_ENDPOINT = "https://pdf-services.adobe.io/operation/extract"  # placeholder
HUGGINGFACE_SUMMARY_ENDPOINT = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HUGGINGFACE_PROMPT_ENDPOINT = "https://api-inference.huggingface.co/models/google/flan-t5-base"
JSON2VIDEO_ENDPOINT = "https://api.json2video.com/v1/create"  # placeholder
ELEVENLABS_TTS_ENDPOINT = "https://api.elevenlabs.io/v1/text-to-speech"  # placeholder

# Allowed asset library (no external fetches allowed)
ALLOWED_IMAGE_ASSETS = {
    "default_bg": "https://example.com/assets/default_bg.jpg",  # replace with your safe asset CDN
}


# ----------------------------- Logging --------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("pdf_to_video")


# ------------------------- Sanitizers & Validators ---------------------------

def sanitize_text_for_model(s: str, max_chars: int = 6000) -> str:
    """
    Basic sanitizer to remove dangerous characters and control sequences.
    - Removes non-printable characters
    - Collapses whitespace
    - Truncates to max_chars
    """
    if not s:
        return s
    # Remove control chars
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", s)
    # Remove HTML tags
    s = re.sub(r"<[^>]+>", " ", s)
    # Replace sequences that look like code or shell commands (heuristic)
    s = re.sub(r"(?i)(rm\s+-rf|sudo|curl\b|wget\b|fetch\(|open\(|exec\(|os\.system|subprocess\.)", "[REDACTED]", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Truncate safely
    if len(s) > max_chars:
        s = s[:max_chars]
    return s


VIDEO_SCRIPT_SCHEMA = {
    "type": "object",
    "required": ["scenes"],
    "properties": {
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "duration", "text"],
                "properties": {
                    "id": {"type": "integer", "minimum": 0},
                    "duration": {"type": "number", "minimum": 0.5},
                    "text": {"type": "string", "maxLength": 1000},
                    "background": {"type": "string"},
                    "image_asset": {"type": "string"},
                    "animation": {"type": "string"},
                    "caption": {"type": "string"}
                }
            }
        }
    }
}


def validate_video_script(script: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        validate(instance=script, schema=VIDEO_SCRIPT_SCHEMA)
        return True, ""
    except ValidationError as e:
        return False, str(e)


# ------------------------- Utilities ----------------------------------------

def retry_request(method, url, headers=None, json=None, files=None, params=None):
    backoff = RETRY_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.request(method, url, headers=headers, json=json, files=files, params=params, timeout=HTTP_TIMEOUT)
            if resp.status_code in (200, 201, 202):
                return resp
            # Some APIs return 4xx with JSON explaining quota; treat as permanent
            if 400 <= resp.status_code < 500:
                logger.error("Permanent error %s from %s: %s", resp.status_code, url, resp.text[:300])
                return resp
            # else retryable server error
            logger.warning("Retryable response %s, attempt %d/%d", resp.status_code, attempt + 1, MAX_RETRIES)
        except requests.RequestException as e:
            logger.warning("Request exception: %s", str(e))
        time.sleep(backoff)
        backoff *= 2
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {url}")


# ------------------------- Module 1: PDF Extraction --------------------------

def extract_text_from_pdf_local(pdf_path: str) -> str:
    """
    Local extraction fallback using pdfplumber. Preferred for hackathon when Adobe creds not present.
    """
    logger.info("Extracting text locally from %s", pdf_path)
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text_parts.append(txt)
    text = "\n\n".join(text_parts)
    text = sanitize_text_for_model(text, max_chars=20000)  # keep reasonably large for summarization chunking
    return text


# ------------------------- Module 2: Summarization --------------------------

def summarize_with_huggingface(text: str, model_endpoint: str = HUGGINGFACE_SUMMARY_ENDPOINT) -> str:
    """
    Call HuggingFace Inference API for summarization. Uses HF API key from env var.
    """
    hf_key = os.getenv("HUGGINGFACE_API_KEY", "")
    if not hf_key:
        raise RuntimeError("HUGGINGFACE_API_KEY not set in environment. Provide key or mock locally.")

    payload = {"inputs": text, "parameters": {"max_new_tokens": 256, "min_length": 60}}
    headers = {"Authorization": f"Bearer {hf_key}"}
    logger.info("Summarizing text with HuggingFace model %s", model_endpoint)
    resp = retry_request("POST", model_endpoint, headers=headers, json=payload)
    if resp.ok:
        data = resp.json()
        # HF may return varying formats; try to extract 'summary_text' or join 'generated_text'
        if isinstance(data, dict) and "summary_text" in data:
            summary = data["summary_text"]
        elif isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            summary = data[0]["generated_text"]
        else:
            # fallback to raw text
            summary = data if isinstance(data, str) else json.dumps(data)[:10000]
        return sanitize_text_for_model(summary, max_chars=5000)
    else:
        raise RuntimeError(f"HuggingFace summarization failed: {resp.status_code} - {resp.text[:400]}")


# ------------------- Module 3: Unified Prompt Generator ---------------------

PROMPT_TEMPLATE = """
You are a strict formatter. Input: a short summary of the content. Output must be valid JSON ONLY (no prose).
Produce a JSON object with two keys:
- video_script: {{ "scenes": [ {{ "id": int, "duration": float, "text": string, "background": string (optional), "image_asset": string (optional), "animation": string (optional) }} ] }}
- audio_script: a single string containing the narration in natural language concatenated to follow the scenes in order.
Constraints:
- Total duration across scenes must be between 55 and 125 seconds (aim for 60-90s if possible).
- Max 8 scenes.
- Do not include external URLs. Use asset keys from allowed list (e.g., 'default_bg') or null.
- Validate numeric fields. All text fields must be plain text (no HTML, no code).
Summary: {summary}
"""

def generate_unified_prompt(summary_text: str) -> Dict[str, Any]:
    """
    Call flan-t5-base (or any HF model) to produce the unified prompt output.
    Returns parsed JSON as Python dict after validation & sanitization.
    """
    hf_key = os.getenv("HUGGINGFACE_API_KEY", "")
    if not hf_key:
        raise RuntimeError("HUGGINGFACE_API_KEY not set. Set it to call HF Inference API.")

    prompt = PROMPT_TEMPLATE.format(summary=sanitize_text_for_model(summary_text, max_chars=4000))
    headers = {"Authorization": f"Bearer {hf_key}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 800}}
    logger.info("Calling HF prompt generator")
    resp = retry_request("POST", HUGGINGFACE_PROMPT_ENDPOINT, headers=headers, json=payload)
    if not resp.ok:
        raise RuntimeError("Prompt generator failed: %s %s" % (resp.status_code, resp.text[:400]))
    # HF sometimes returns list; unify
    data = resp.json()
    # extract text
    if isinstance(data, list) and data and "generated_text" in data[0]:
        generated = data[0]["generated_text"]
    elif isinstance(data, dict) and "generated_text" in data:
        generated = data["generated_text"]
    elif isinstance(data, dict) and "summary_text" in data:
        generated = data["summary_text"]
    elif isinstance(data, str):
        generated = data
    else:
        generated = json.dumps(data)  # fallback
    # Try to parse JSON block in generated text
    try:
        # find first { and last } to extract JSON chunk
        first = generated.find("{")
        last = generated.rfind("}")
        if first == -1 or last == -1:
            raise ValueError("No JSON detected in model output")
        json_text = generated[first:last+1]
        parsed = json.loads(json_text)
    except Exception as e:
        logger.error("Failed to parse JSON from model output: %s", str(e))
        # Attempt regeneration with stricter prompt (one-shot)
        strict_prompt = "OUTPUT ONLY JSON. Strict. " + prompt
        payload2 = {"inputs": strict_prompt, "parameters": {"max_new_tokens": 800}}
        resp2 = retry_request("POST", HUGGINGFACE_PROMPT_ENDPOINT, headers=headers, json=payload2)
        if not resp2.ok:
            raise RuntimeError("Regeneration failed: %s" % resp2.text[:300])
        text2 = resp2.json()
        generated2 = text2[0]["generated_text"] if isinstance(text2, list) else (text2.get("generated_text") if isinstance(text2, dict) else str(text2))
        first = generated2.find("{"); last = generated2.rfind("}")
        if first == -1 or last == -1:
            raise RuntimeError("Could not recover valid JSON from model output after retry.")
        parsed = json.loads(generated2[first:last+1])
    # Basic sanitization of parsed fields
    if "video_script" not in parsed or "audio_script" not in parsed:
        raise RuntimeError("Parsed JSON missing required keys (video_script/audio_script)")
    # Validate video script schema
    ok, msg = validate_video_script(parsed["video_script"])
    if not ok:
        raise RuntimeError(f"Video script schema validation failed: {msg}")
    # Ensure asset keys are allowed or normalized
    for scene in parsed["video_script"]["scenes"]:
        asset = scene.get("image_asset")
        if asset and asset not in ALLOWED_IMAGE_ASSETS:
            scene["image_asset"] = None
        bg = scene.get("background")
        if bg and bg not in ALLOWED_IMAGE_ASSETS:
            scene["background"] = None
        # sanitize texts
        scene["text"] = sanitize_text_for_model(scene.get("text", ""), max_chars=1000)
    parsed["audio_script"] = sanitize_text_for_model(parsed["audio_script"], max_chars=5000)
    return parsed


# ------------------- Module 4: Video Generation (JSON2Video) -----------------

def call_json2video_create(video_json: Dict[str, Any], output_name: str) -> Dict[str, Any]:
    """
    Calls the JSON2Video API to create the video. Expects caller to have generated audio file and included its URL in the JSON
    under a safe asset key.
    """
    key = os.getenv("JSON2VIDEO_API_KEY", "")
    if not key:
        raise RuntimeError("JSON2VIDEO_API_KEY not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "template": video_json,
        "options": {"output_format": "mp4", "resolution": "720p", "output_name": output_name}
    }
    logger.info("Submitting job to JSON2Video")
    resp = retry_request("POST", JSON2VIDEO_ENDPOINT, headers=headers, json=payload)
    if not resp.ok:
        raise RuntimeError("JSON2Video failed: %s - %s" % (resp.status_code, resp.text[:400]))
    return resp.json()


# ------------------- Module 5: TTS (ElevenLabs) ------------------------------

def call_elevenlabs_tts(text: str, voice: str = "alloy", output_path: str = "/tmp/narration.wav") -> str:
    """
    Convert text -> WAV using ElevenLabs. Returns path to saved file. Caller should upload to a CDN or provide accessible url for JSON2Video.
    """
    key = os.getenv("ELEVENLABS_API_KEY", "")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    headers = {"xi-api-key": key, "Content-Type": "application/json"}
    payload = {"text": text, "voice": voice, "format": "wav"}
    logger.info("Calling ElevenLabs TTS for narration")
    resp = retry_request("POST", ELEVENLABS_TTS_ENDPOINT, headers=headers, json=payload)
    if not resp.ok:
        raise RuntimeError("ElevenLabs TTS failed: %s - %s" % (resp.status_code, resp.text[:400]))
    # ElevenLabs typically returns binary in real API; here we assume json with url or binary
    # For real implementation, handle binary streaming and write to output_path
    # We'll mock-write a placeholder file path for this template
    with open(output_path, "wb") as fh:
        fh.write(b"")  # placeholder empty wav - replace by actual audio bytes
    return output_path


# ------------------- Module 6: Assembly (FFmpeg) ----------------------------

def assemble_video_with_ffmpeg(scene_video_paths: List[str], narration_path: str, output_path: str) -> None:
    """
    Given a list of scene video file paths and narration audio path, concatenate and overlay audio.
    Uses ffmpeg. Ensures correct codecs and trims to shortest stream to avoid desync.
    """
    logger.info("Assembling final video with ffmpeg")
    # Create a file list for ffmpeg concat demuxer
    list_file = "/tmp/concat_list.txt"
    with open(list_file, "w") as fh:
        for p in scene_video_paths:
            fh.write(f"file '{p}'\n")
    intermediate = "/tmp/assembled_video.mp4"
    # Concatenate videos
    cmd_concat = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", intermediate]
    logger.info("Running ffmpeg concat: %s", " ".join(cmd_concat))
    subprocess.run(cmd_concat, check=True)
    # Merge audio and video, trimming to shortest
    cmd_merge = ["ffmpeg", "-y", "-i", intermediate, "-i", narration_path, "-c:v", "copy", "-c:a", "aac", "-shortest", output_path]
    logger.info("Running ffmpeg merge: %s", " ".join(cmd_merge))
    subprocess.run(cmd_merge, check=True)
    logger.info("Final video at %s", output_path)


# ------------------- Master Orchestrator ------------------------------------

@dataclass
class PipelineConfig:
    pdf_path: str
    work_dir: str = "/tmp/pdf2video_work"
    final_output: str = "/tmp/final_output.mp4"
    tmp_audio: str = "/tmp/narration.wav"


def orchestrate_pipeline(cfg: PipelineConfig):
    os.makedirs(cfg.work_dir, exist_ok=True)
    # 1. Extract text (local)
    text = extract_text_from_pdf_local(cfg.pdf_path)
    if not text.strip():
        raise RuntimeError("No text extracted from PDF")
    logger.info("Extracted %d characters", len(text))

    # 2. Summarize
    summary = summarize_with_huggingface(text)
    logger.info("Summary length %d", len(summary))

    # 3. Prompt generator (unified)
    unified = generate_unified_prompt(summary)
    logger.info("Generated unified prompt with %d scenes", len(unified["video_script"]["scenes"]))

    # 4. Generate TTS audio for audio_script
    narration_path = os.path.join(cfg.work_dir, "narration.wav")
    tts_path = call_elevenlabs_tts(unified["audio_script"], output_path=narration_path)
    logger.info("Saved narration to %s", tts_path)

    # 5. Prepare JSON2Video template (map asset keys to real safe urls)
    # Example transformation: replace asset keys with ALLOWED_IMAGE_ASSETS values
    template = {"timeline": {"tracks": [{"clips": []}] } }
    current_time = 0.0
    scene_paths = []
    for scene in unified["video_script"]["scenes"]:
        duration = float(scene.get("duration", 5))
        bg_key = scene.get("background") or "default_bg"
        bg_url = ALLOWED_IMAGE_ASSETS.get(bg_key, ALLOWED_IMAGE_ASSETS["default_bg"])
        # Simple per-scene JSON2Video clip object (adapt for target API)
        clip = {
            "asset": {"type": "image", "src": bg_url},
            "start": current_time,
            "length": duration,
            "overlay_text": scene.get("text")
        }
        template["timeline"]["tracks"][0]["clips"].append(clip)
        current_time += duration

    # Insert audio asset: in JSON2Video you often supply an accessible URL; here we assume you'll upload narration to a CDN
    # For template demo, attach a local path (won't work until uploaded)
    template["audio"] = {"type": "audio", "src": "UPLOAD_YOUR_NARRATION_AND_PUT_URL_HERE"}

    # 6. Call JSON2Video to render video(s)
    job = call_json2video_create(template, output_name="hackathon_demo")
    logger.info("JSON2Video job submitted: %s", job)

    # 7. For demo purposes: assume JSON2Video returned scene video files that we can download to scene_paths.
    # In reality, JSON2Video returns final video URL or job id. Here we skip actual download and assembly.
    # If scene video files are available locally, call assemble_video_with_ffmpeg([...], tts_path, cfg.final_output)

    logger.info("Pipeline complete (note: final rendering requires valid API keys and accessible assets).")
    return {
        "summary": summary,
        "unified_prompt": unified,
        "json2video_template": template,
        "json2video_job": job
    }


# ----------------------------- Write file -----------------------------------

MODULE_CODE = r'''# pdf_to_video_pipeline.py
# Generated by ChatGPT for Saarthak - template orchestrator for PDF -> 1-2 minute video

# (See the top of the notebook cell for usage, env vars, and requirements)
'''

file_path = "/mnt/data/pdf_to_video_pipeline.py"
with open(file_path, "w") as f:
    f.write("# pdf_to_video_pipeline.py\n")
    f.write("# Full orchestrator template generated by ChatGPT\n\n")
    # write this module's source for user to inspect
    import inspect, sys
    source_text = inspect.getsource(extract_text_from_pdf_local) + "\n\n" + inspect.getsource(summarize_with_huggingface) + "\n\n" + inspect.getsource(generate_unified_prompt) + "\n\n" + inspect.getsource(call_elevenlabs_tts) + "\n\n" + inspect.getsource(call_json2video_create) + "\n\n" + inspect.getsource(assemble_video_with_ffmpeg) + "\n\n" + inspect.getsource(orchestrate_pipeline)
    f.write(source_text)

print("Wrote orchestrator template to:", file_path)

# Print a small example of a validated video JSON and audio script so you can preview the format.
example = {
    "video_script": {
        "scenes": [
            {"id": 0, "duration": 10, "text": "Introduction: what this PDF explains", "background": "default_bg"},
            {"id": 1, "duration": 40, "text": "Key idea one explained succinctly", "background": "default_bg"},
            {"id": 2, "duration": 40, "text": "Conclusion and next steps", "background": "default_bg"}
        ]
    },
    "audio_script": "Introduction: what this PDF explains. Key idea one explained succinctly. Conclusion and next steps."
}

valid, reason = validate_video_script(example["video_script"])
print("\nExample unified prompt valid:", valid, "reason:", reason)
print("\nExample JSON preview:")
print(json.dumps(example, indent=2))

print("\nNext steps (copy/paste):")
print("1) Install requirements: pip install requests jsonschema pdfplumber python-dotenv tqdm")
print("2) Put your API keys into environment or .env (ADOBE_CLIENT_ID/SECRET, HUGGINGFACE_API_KEY, JSON2VIDEO_API_KEY, ELEVENLABS_API_KEY)")
print("3) Upload safe asset images to your CDN and replace ALLOWED_IMAGE_ASSETS values in the script")
print("4) Run the orchestrator: python /mnt/data/pdf_to_video_pipeline.py  (or import orchestrate_pipeline and call it)")
print("\nThe generated file contains core functions but must be extended to your endpoint details and deployment environment.")

# show path to file for download
"/mnt/data/pdf_to_video_pipeline.py"


