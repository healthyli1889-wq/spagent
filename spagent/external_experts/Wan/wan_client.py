import os
import time
import base64
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported sizes per aspect ratio
SIZES = {
    "16:9": "1280*720",
    "9:16": "720*1280",
    "1:1":  "960*960",
}

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/api/v1"


class WanClient:
    """Client for Alibaba Wan video generation via DashScope API.

    Supports text-to-video (t2v) and image-to-video (i2v).
    - t2v default model: wanx2.1-t2v-turbo
    - i2v default model: wanx2.1-i2v-turbo  (auto-selected when image_path given)
    """

    def __init__(self, api_key: str = None, model: str = "wanx2.1-t2v-turbo"):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DashScope API key is required. Set DASHSCOPE_API_KEY env variable "
                "or pass api_key to WanClient."
            )
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }

    def generate_video(
        self,
        prompt: str,
        image_path: str = None,
        duration: int = 5,
        aspect_ratio: str = "16:9",
    ) -> dict:
        """Generate a video from a text prompt, optionally conditioned on an image.

        Args:
            prompt: Text description of the video to generate.
            image_path: Optional local path or public URL to a reference image
                        for image-to-video generation.
            duration: Video duration in seconds (3–10). Default 5.
            aspect_ratio: "16:9", "9:16", or "1:1". Default "16:9".

        Returns:
            dict with keys: success, output_path, error.
        """
        try:
            size = SIZES.get(aspect_ratio, SIZES["16:9"])
            duration = max(3, min(10, duration))

            # Auto-switch to i2v model when image provided
            model = self.model
            input_payload: dict = {"prompt": prompt}

            if image_path:
                model = self._to_i2v_model(model)
                img_url = self._resolve_image(image_path)
                if img_url is None:
                    return {"success": False, "error": f"Image not found: {image_path}"}
                input_payload["img_url"] = img_url

            payload = {
                "model": model,
                "input": input_payload,
                "parameters": {
                    "size": size,
                    "duration": duration,
                },
            }

            url = f"{DASHSCOPE_BASE}/services/aigc/video-generation/video-synthesis"
            logger.info(f"Sending video generation request to Wan (model={model})...")
            resp = requests.post(url, headers=self.headers, json=payload, timeout=60)

            if not resp.ok:
                logger.error(f"Wan API error ({resp.status_code}): {resp.text}")
                return {"success": False, "error": f"Wan API {resp.status_code}: {resp.text}"}

            data = resp.json()
            task_id = data.get("output", {}).get("task_id")
            if not task_id:
                return {"success": False, "error": f"No task_id in Wan response: {data}"}

            logger.info(f"Wan task created: {task_id}")
            return self._poll_task(task_id)

        except requests.exceptions.RequestException as e:
            logger.error(f"Wan API request failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Wan generation error: {e}")
            return {"success": False, "error": str(e)}

    def _to_i2v_model(self, model: str) -> str:
        """Swap t2v model to the corresponding i2v variant."""
        if "t2v-turbo" in model:
            return "wanx2.1-i2v-turbo"
        if "t2v-plus" in model:
            return "wanx2.1-i2v-plus"
        # Already an i2v model or unknown — return as-is
        return model

    def _resolve_image(self, image_path: str):
        """Return a public URL or base64 data-URL for the given image.

        DashScope i2v accepts a URL in img_url.  For local files we encode
        them as a data-URL (data:image/<ext>;base64,...).
        """
        # Already a URL
        if image_path.startswith("http://") or image_path.startswith("https://"):
            return image_path

        # Local file — encode as data URL
        if not os.path.exists(image_path):
            return None

        with open(image_path, "rb") as f:
            image_bytes = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lstrip(".").lower()
        mime = f"image/{ext}" if ext in ("png", "jpeg", "jpg", "webp") else "image/jpeg"
        return f"data:{mime};base64,{image_bytes}"

    def _poll_task(self, task_id: str, timeout: int = 600, interval: int = 10) -> dict:
        """Poll the DashScope task endpoint until completion."""
        url = f"{DASHSCOPE_BASE}/tasks/{task_id}"
        poll_headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        start = time.time()

        while time.time() - start < timeout:
            resp = requests.get(url, headers=poll_headers, timeout=30)
            if not resp.ok:
                logger.warning(f"Poll {resp.status_code}: {resp.text}")
                time.sleep(interval)
                continue

            data = resp.json()
            output = data.get("output", {})
            status = output.get("task_status", "")

            if status == "SUCCEEDED":
                video_url = output.get("video_url")
                if not video_url:
                    return {"success": False, "error": f"No video_url in response: {output}"}
                return self._download_video(video_url)

            if status == "FAILED":
                err = output.get("message", "Wan generation failed.")
                return {"success": False, "error": err}

            logger.info(
                f"Wan generation: status={status} ({int(time.time() - start)}s elapsed)"
            )
            time.sleep(interval)

        return {"success": False, "error": f"Wan generation timed out after {timeout}s"}

    def _download_video(self, video_url: str) -> dict:
        """Download the generated video and save locally."""
        try:
            os.makedirs("outputs", exist_ok=True)
            output_path = f"outputs/wan_{int(time.time())}.mp4"

            dl_resp = requests.get(video_url, timeout=120, stream=True)
            dl_resp.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in dl_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = os.path.getsize(output_path)
            logger.info(f"Wan video saved to: {output_path} ({file_size} bytes)")
            return {
                "success": True,
                "output_path": output_path,
                "file_size_bytes": file_size,
            }
        except Exception as e:
            logger.error(f"Failed to download Wan video: {e}")
            return {"success": False, "error": str(e)}
