# Local AI Banner and Poster Generator

This Gradio project uses the configured **local Ollama chat model** to refine
the prompt and **Hugging Face Diffusers** to generate the image on this PC. It
does not require OpenAI, ComfyUI, or a separate image server.

## Setup

Install the image-generation dependencies in the repository's Python
environment, then ensure Ollama is running with the configured chat model:

```powershell
.\.venv\Scripts\python.exe -m pip install -r Module-5\GENAI\projects\requirements-imagegen.txt
ollama pull qwen3:8b
.\.venv\Scripts\python.exe Module-5\GENAI\projects\image_generation.py
```

Open `http://127.0.0.1:7860`, enter an idea, choose dimensions and generation
quality, and click **Generate image**. The app displays the Ollama-refined
prompt and the generated PIL image.

## Local inference notes

- `GENAI_CHAT_MODEL` and `OLLAMA_HOST` configure prompt refinement through
  `GenAIConfig` (defaults: `qwen3:8b` and `http://localhost:11434`).
- `GENAI_IMAGE_MODEL` selects the Diffusers model; the default is
  `segmind/tiny-sd`, a small model chosen for CPU-only use.
- `GENAI_GRADIO_HOST` and `GENAI_GRADIO_PORT` configure the web UI.
- The model is downloaded from Hugging Face on first use and cached locally.
  First generation can take several minutes on CPU.

This project environment has Python 3.14, Intel integrated graphics (no CUDA),
31.5 GB system RAM, and about 13 GB available at the time of setup. PyTorch and
Diffusers currently publish compatible Windows/Python wheels. Inference uses
CPU float32; integrated Intel graphics are not used by this pipeline. Close
memory-heavy applications if model loading fails.

Empty prompts and unsupported options are rejected before inference. Model
download, Ollama, and generation errors are surfaced in the UI. The
`generate_image()` function accepts injectable Ollama and pipeline instances
for tests that do not download a model or generate an image.
