"""OpenAI-compatible HTTP API for Anima-MLX image generation."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .pipeline import AnimaPipeline
from .tokenizer import AnimaTokenizer

DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, score_1, score_2, score_3, artist name, "
    "blurry, jpeg artifacts, chromatic aberration, text, watermark, nsfw, explicit"
)
logger = logging.getLogger("uvicorn.error")


class ImageGenerationRequest(BaseModel):
    """Subset of the OpenAI Images request plus Anima sampling controls."""

    model: str = "anima-mlx"
    prompt: str = Field(min_length=1, max_length=4000)
    n: int = 1
    size: str = "512x512"
    response_format: Literal["url", "b64_json"] = "url"
    negative_prompt: str = Field(default=DEFAULT_NEGATIVE_PROMPT, max_length=2000)
    steps: int = Field(default=30, ge=8, le=50)
    cfg: float = Field(default=5.0, ge=1.0, le=6.0)
    seed: int | None = None

    @field_validator("n")
    @classmethod
    def one_image_per_request(cls, value: int) -> int:
        """Keep requests serial and predictable on a fanless laptop.

        Args:
            value: Requested image count.

        Returns:
            The validated image count.

        Raises:
            ValueError: If more than one image is requested.
        """
        if value != 1:
            raise ValueError("Anima-MLX currently supports n=1")
        return value


def _parse_size(size: str) -> tuple[int, int]:
    """Validate and parse an image size.

    Args:
        size: Width and height in WIDTHxHEIGHT form.

    Returns:
        A width and height tuple.

    Raises:
        ValueError: If dimensions are outside the supported range.
    """
    try:
        width, height = (int(part) for part in size.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("size must use WIDTHxHEIGHT, for example 512x512") from exc
    if not (256 <= width <= 1024 and 256 <= height <= 1024):
        raise ValueError("width and height must be between 256 and 1024")
    if width % 16 or height % 16:
        raise ValueError("width and height must be divisible by 16")
    return width, height


def _authorize(authorization: str | None) -> None:
    """Check the optional bearer token configured for the service.

    Args:
        authorization: Incoming Authorization header.

    Raises:
        HTTPException: If the bearer token is missing or invalid.
    """
    api_key = os.getenv("ANIMA_API_KEY", "").strip()
    if api_key and not secrets.compare_digest(authorization or "", f"Bearer {api_key}"):
        raise HTTPException(status_code=401, detail="invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load one shared pipeline and tokenizer for the process lifetime.

    Args:
        app: FastAPI application instance.

    Yields:
        Control while the application is serving requests.
    """
    repo_id = os.getenv("ANIMA_MODEL_REPO", "xocialize/anima-mlx")
    quant = os.getenv("ANIMA_QUANT", "int4")
    if quant not in {"int4", "bf16"}:
        raise RuntimeError("ANIMA_QUANT must be int4 or bf16")
    app.state.quant = quant
    app.state.repo_id = repo_id
    app.state.output_dir = Path(
        os.getenv("ANIMA_OUTPUT_DIR", str(Path.cwd() / "outputs"))
    ).resolve()
    app.state.output_dir.mkdir(parents=True, exist_ok=True)
    app.state.generate_lock = asyncio.Lock()
    logger.info(
        "Loading %s (%s); first startup may download model files", repo_id, quant
    )
    app.state.pipeline = await asyncio.to_thread(
        AnimaPipeline.from_pretrained, repo_id, quant
    )
    app.state.tokenizer = await asyncio.to_thread(AnimaTokenizer)
    logger.info("Anima-MLX is ready")
    yield


app = FastAPI(
    title="Anima-MLX API",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health")
async def health(authorization: str | None = Header(default=None)) -> dict:
    """Return service readiness and model configuration."""
    _authorize(authorization)
    return {
        "status": "ok",
        "model": app.state.repo_id,
        "quant": app.state.quant,
        "busy": app.state.generate_lock.locked(),
    }


@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None)) -> dict:
    """Return the OpenAI-compatible model list."""
    _authorize(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": "anima-mlx",
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/images/generations")
async def generate_image(
    payload: ImageGenerationRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict:
    """Generate one image and return a URL or base64 payload.

    Args:
        payload: Image prompt and generation controls.
        request: Current HTTP request, used to form the image URL.
        authorization: Optional bearer token.

    Returns:
        An OpenAI-compatible Images response.

    Raises:
        HTTPException: If authentication or size validation fails.
    """
    _authorize(authorization)
    try:
        width, height = _parse_size(payload.size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload.model not in {"anima-mlx", app.state.repo_id}:
        raise HTTPException(status_code=404, detail="unknown model")

    seed = payload.seed if payload.seed is not None else secrets.randbelow(2**31)
    started = time.time()
    logger.info(
        "Queued generation size=%s steps=%d cfg=%.1f seed=%d",
        payload.size,
        payload.steps,
        payload.cfg,
        seed,
    )
    async with app.state.generate_lock:

        def run_generation() -> tuple[np.ndarray, float]:
            tokenizer = app.state.tokenizer
            pipeline = app.state.pipeline
            mx.reset_peak_memory()
            qids, tids, weights = tokenizer.encode_weighted(payload.prompt)
            cond = pipeline.encode_context(qids, tids, weights)
            nqids, ntids, nweights = tokenizer.encode_weighted(payload.negative_prompt)
            uncond = pipeline.encode_context(nqids, ntids, nweights)
            image = pipeline.generate(
                cond,
                uncond,
                height=height,
                width=width,
                steps=payload.steps,
                cfg=payload.cfg,
                seed=seed,
            )
            mx.eval(image)
            array = (
                (np.asarray(image[0], np.float32) * 255.0)
                .round()
                .clip(0, 255)
                .astype(np.uint8)
            )
            peak_memory_gb = round(mx.get_peak_memory() / 1e9, 2)
            mx.clear_cache()
            return array, peak_memory_gb

        array, peak_memory_gb = await asyncio.to_thread(run_generation)

    from PIL import Image

    filename = f"anima_{int(time.time() * 1000)}_{seed}.png"
    path = app.state.output_dir / filename
    await asyncio.to_thread(Image.fromarray(array).save, path)
    logger.info(
        "Generated %s in %.2fs (peak %.2f GB)",
        filename,
        time.time() - started,
        peak_memory_gb,
    )
    result: dict[str, str | int] = {"seed": seed}
    if payload.response_format == "b64_json":
        result["b64_json"] = base64.b64encode(path.read_bytes()).decode("ascii")
    else:
        public_base = os.getenv("ANIMA_PUBLIC_BASE_URL", "").rstrip("/")
        if not public_base:
            public_base = str(request.base_url).rstrip("/")
        result["url"] = f"{public_base}/images/{filename}"
    return {
        "created": int(started),
        "data": [result],
        "usage": {
            "generation_seconds": round(time.time() - started, 2),
            "peak_memory_gb": peak_memory_gb,
        },
    }


@app.get("/images/{filename}")
async def image(filename: str) -> FileResponse:
    """Serve a generated PNG without requiring API credentials.

    Args:
        filename: Generated image filename.

    Returns:
        The requested PNG response.

    Raises:
        HTTPException: If the image does not exist.
    """
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.endswith(".png"):
        raise HTTPException(status_code=404, detail="image not found")
    path = app.state.output_dir / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path, media_type="image/png")


def main() -> None:
    """Run the API server using environment-based host and port settings."""
    import uvicorn

    uvicorn.run(
        "anima_mlx.api:app",
        host=os.getenv("ANIMA_HOST", "0.0.0.0"),
        port=int(os.getenv("ANIMA_PORT", "10240")),
        workers=1,
    )
