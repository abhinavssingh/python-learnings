"""Generate images locally with Ollama prompt refinement and Diffusers."""

import os
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch
from diffusers import DiffusionPipeline
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import init  # noqa: E402,F401

from lib.utility.genai.config import GenAIConfig  # noqa: E402
from lib.utility.genai.factory import get_llm  # noqa: E402

IMAGE_SIZES = ("384x384", "512x512", "512x768", "768x512")
QUALITY_STEPS = {"Draft": 8, "Standard": 15, "High": 22}
DEFAULT_IMAGE_MODEL = "segmind/tiny-sd"
PROMPT_REFINEMENT_SYSTEM = (
    "You write concise prompts for a local text-to-image model. Rewrite the user's "
    "design idea with a clear subject, composition, visual style, lighting, and color "
    "palette. Do not invent readable words, brand names, or logos unless the user "
    "explicitly asks for them. Return only the final image prompt."
)


def refine_prompt(prompt: str, llm) -> str:
    """Refine an image description with the configured local Ollama chat model."""
    result = llm.generate(
        prompt.strip(),
        system=PROMPT_REFINEMENT_SYSTEM,
        max_tokens=180,
        temperature=0.2,
    ).text.strip()
    if not result:
        raise RuntimeError("Ollama returned an empty refined prompt.")
    return result


@lru_cache(maxsize=2)
def load_image_pipeline(model_id: str):
    """Load and cache a text-to-image model for CPU inference."""
    try:
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load local image model '{model_id}'. Check the model ID, "
            "internet access for the first download, and available disk space. "
            f"Details: {exc}"
        ) from exc

    pipeline = pipeline.to("cpu")
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    return pipeline


def generate_image(
    prompt: str,
    size: str = "384x384",
    quality: str = "Standard",
    *,
    llm=None,
    pipeline=None,
    model_id: str | None = None,
    seed: int | None = None,
    return_prompt: bool = False,
):
    """Refine a prompt locally, generate an image with Diffusers, and return PIL RGB."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Enter a description for the image you want to generate.")
    if size not in IMAGE_SIZES:
        raise ValueError(f"Unsupported image size: {size}")
    if quality not in QUALITY_STEPS:
        raise ValueError(f"Unsupported image quality: {quality}")

    if llm is None:
        llm = get_llm(GenAIConfig.from_env())
    refined = refine_prompt(prompt, llm)
    model_id = model_id or os.getenv("GENAI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
    pipeline = pipeline or load_image_pipeline(model_id)
    width, height = (int(value) for value in size.split("x"))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed if seed is not None else int.from_bytes(os.urandom(8), "big"))

    try:
        result = pipeline(
            prompt=refined,
            width=width,
            height=height,
            num_inference_steps=QUALITY_STEPS[quality],
            guidance_scale=7.0,
            generator=generator,
        )
        if not result.images:
            raise RuntimeError("The local image model returned no image.")
        image = result.images[0].convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            f"Local image generation failed with '{model_id}': {exc}"
        ) from exc

    return (image, refined) if return_prompt else image


def create_app() -> gr.Blocks:
    config = GenAIConfig.from_env()
    llm = get_llm(config)
    model_id = os.getenv("GENAI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)

    def generate(prompt: str, size: str, quality: str):
        return generate_image(
            prompt,
            size,
            quality,
            llm=llm,
            model_id=model_id,
            return_prompt=True,
        )

    with gr.Blocks(title="Local AI Banner and Poster Generator") as app:
        gr.Markdown(
            "# Local AI Banner and Poster Generator\n"
            "Ollama refines the prompt; Diffusers generates the image on this PC "
            f"using CPU inference. Text model: `{config.chat_model}` · Image model: `{model_id}`."
        )
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Image idea",
                    placeholder="Describe the subject, visual style, colors, and composition.",
                    lines=4,
                    max_lines=8,
                )
                size = gr.Dropdown(
                    choices=list(IMAGE_SIZES),
                    value="384x384",
                    label="Image dimensions",
                )
                quality = gr.Dropdown(
                    choices=list(QUALITY_STEPS),
                    value="Standard",
                    label="Generation quality",
                )
                generate_button = gr.Button("Generate image", variant="primary")
                refined_prompt = gr.Textbox(
                    label="Ollama-refined image prompt",
                    interactive=False,
                    lines=3,
                )
            with gr.Column(scale=1):
                output = gr.Image(label="Generated design", type="pil")

        generate_button.click(
            fn=generate,
            inputs=[prompt, size, quality],
            outputs=[output, refined_prompt],
        )

    return app


def main() -> None:
    host = os.getenv("GENAI_GRADIO_HOST", "127.0.0.1")
    port = int(os.getenv("GENAI_GRADIO_PORT", "7860"))
    create_app().launch(server_name=host, server_port=port, show_error=True)


if __name__ == "__main__":
    main()
