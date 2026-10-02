# Applied GenAI and AI Projects

This folder contains runnable applications and analysis demos that apply ideas from the GenAI learning path to practical tasks. Read each project's companion guide before running it: model downloads, datasets, optional dependencies, and output paths differ.

## Projects

> The autonomous-driving project files were moved to the capstone section. Use the guides in [../capstone/README.md](../capstone/README.md) and [../capstone/AUTONOMOUS_DRIVING_README.md](../capstone/AUTONOMOUS_DRIVING_README.md) instead of the older project paths.

### HR policy assistant

`hr_assistant.py` is a local Gradio conversational RAG application for the HR policy PDF. It retrieves policy passages and retains a short conversation window.

```powershell
python Module-5/GENAI/projects/hr_assistant.py
```

Install the GenAI requirements and pull the local chat and embedding models as described in [HR_ASSISTANT_README.md](HR_ASSISTANT_README.md). Open `http://127.0.0.1:7860` after startup. First launch indexes the PDF; later launches can reuse the saved index.

### Local image generation

`image_generation.py` provides a Gradio interface that refines a prompt with the local Ollama chat model and generates an image with Hugging Face Diffusers.

```powershell
python Module-5/GENAI/projects/image_generation.py
```

Install `requirements-imagegen.txt` and review [IMAGE_GENERATION_README.md](IMAGE_GENERATION_README.md) before launching. The image model downloads on first use and CPU generation may take several minutes. This project does not require OpenAI or a separate image server.

### Supporting materials

- `vehicle_detection.ipynb` and `tesla_safety_analysis.ipynb` are notebook-based exploration materials for the autonomous-driving theme.
- `image_generation_demo.ipynb` accompanies the image-generation project.
- `HR_Assistant_REQUIREMENTS.md` and `IMAGE_GEN_REQUIREMENTS.md` preserve project requirement specifications.
- `requirements-autonomous.txt` and `requirements-imagegen.txt` are project-specific dependency lists; the HR assistant uses the parent GenAI requirements.

## A learner's checklist

1. Read the relevant project README and requirement document.
2. Confirm datasets and model prerequisites before execution.
3. Run the smallest/default workflow and inspect both console output and saved report.
4. Trace input, transformation, inference, and output stages.
5. Test invalid or unavailable inputs and note the application's error behavior.
6. Distinguish a working demo from evidence of model quality. In particular, pretrained detections are not trained results, and an LLM-generated policy answer should be checked against its retrieved source passages.

Run commands from the repository root. Output and environment settings are documented in each linked project guide and the [parent GenAI guide](../README.md).
