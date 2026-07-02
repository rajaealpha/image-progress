"""
Azure AI Foundry client — wraps the /openai/v1/responses endpoint.
All vision calls (object detection, inpainting description, progress scoring)
go through this single client so the API key / endpoint are managed in one place.
"""

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Optional

import httpx

from construction_progress.config import AZURE_ENDPOINT, AZURE_API_KEY, DEPLOYMENT_NAME, API_IMAGE_SIZE


def _encode_image(image_path: str) -> str:
    """Return base64-encoded image string from a file path."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _image_media_type(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/jpeg")


class AzureVisionClient:
    """
    Thin wrapper around the Azure AI Foundry responses endpoint.
    Supports text-only and image+text (vision) calls.
    """

    def __init__(
        self,
        endpoint: str = AZURE_ENDPOINT,
        api_key: str = AZURE_API_KEY,
        deployment: str = DEPLOYMENT_NAME,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment = deployment
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout, verify=False)

    def _post(self, payload: dict) -> dict:
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.post(self.endpoint, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in (429, 500, 502, 503):
                    time.sleep(2 ** attempt)
                    continue
                raise
            except httpx.RequestError as e:
                last_error = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Azure API call failed after {self.max_retries} retries: {last_error}")

    def _extract_text(self, response: dict) -> str:
        """Pull text content from the responses API output."""
        try:
            for item in response.get("output", []):
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            return content.get("text", "")
        except (KeyError, TypeError):
            pass
        return str(response)

    def ask(self, prompt: str, max_tokens: int = 1000) -> str:
        """Simple text-only call."""
        payload = {
            "model": self.deployment,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }
        return self._extract_text(self._post(payload))

    def ask_with_images(
        self,
        prompt: str,
        image_paths: list[str],
        max_tokens: int = 2000,
        image_bytes_list: Optional[list[bytes]] = None,
        temperature: float = 0.0,
    ) -> str:
        """
        Vision call — sends prompt + one or more images.
        Pass either image_paths (file paths) or image_bytes_list (raw bytes).
        Uses image_url with data URI — required format for this Azure endpoint.
        """
        content_parts = []

        # Add images as data URIs via image_url field
        if image_bytes_list:
            for img_bytes in image_bytes_list:
                b64 = _encode_image_bytes(img_bytes)
                content_parts.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}",
                })
        else:
            for path in image_paths:
                media_type = _image_media_type(path)
                b64 = _encode_image(path)
                content_parts.append({
                    "type": "input_image",
                    "image_url": f"data:{media_type};base64,{b64}",
                })

        # Add text prompt
        content_parts.append({"type": "input_text", "text": prompt})

        payload = {
            "model": self.deployment,
            "input": [{"role": "user", "content": content_parts}],
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        return self._extract_text(self._post(payload))

    def ask_json(
        self,
        prompt: str,
        image_paths: Optional[list[str]] = None,
        image_bytes_list: Optional[list[bytes]] = None,
        max_tokens: int = 2000,
        temperature: float = 0.0,
    ) -> dict:
        """
        Vision or text call that expects a JSON response.
        Strips markdown fences and parses the result.
        """
        json_prompt = (
            prompt
            + "\n\nRespond ONLY with valid JSON. No explanation, no markdown fences."
        )
        if image_paths or image_bytes_list:
            raw = self.ask_with_images(
                json_prompt,
                image_paths=image_paths or [],
                max_tokens=max_tokens,
                image_bytes_list=image_bytes_list,
                temperature=temperature,
            )
        else:
            raw = self.ask(json_prompt, max_tokens=max_tokens)

        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Attempt to extract first JSON object/array
            import re
            match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise ValueError(f"Could not parse JSON from response: {raw[:500]}")

    async def _async_post(self, payload: dict) -> dict:
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self.timeout, verify=False) as ac:
            for attempt in range(self.max_retries):
                try:
                    r = await ac.post(self.endpoint, headers=headers, json=payload)
                    r.raise_for_status()
                    return r.json()
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (429, 500, 502, 503):
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise
                except httpx.RequestError:
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError("Azure API call failed after retries")

    async def ask_json_async(
        self,
        prompt: str,
        image_bytes_list: list[bytes],
        max_tokens: int = 700,
        temperature: float = 0.0,
        return_raw: bool = False,
    ):
        content_parts = [
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{_encode_image_bytes(b)}"}
            for b in image_bytes_list
        ]
        content_parts.append({"type": "input_text", "text": prompt + "\n\nRespond ONLY with valid JSON. No explanation, no markdown fences."})
        payload = {
            "model": self.deployment,
            "input": [{"role": "user", "content": content_parts}],
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        raw_response = await self._async_post(payload)
        raw = self._extract_text(raw_response)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            import re
            m = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if m:
                parsed = json.loads(m.group(1))
            else:
                raise ValueError(f"Could not parse JSON: {raw[:300]}")
        if return_raw:
            return parsed, raw_response
        return parsed

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
