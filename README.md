# Animated Wallpaper Generation
This repository contains code to deploy apps with Modal to run AI models in ComfyUI for creating animated wallpapers.

## Dependencies
Follow Modal's Getting Started Guide (https://modal.com/docs/guide).

## Usage

### 1. Tokens
Add a `tokens.json` file with your HuggingFace and Civitai tokens in the root project directory following this format:
```
{
    "HF_TOKEN": "<token>",
    "CIVITAI_TOKEN": "<token>"
}
```

### 2. Running the App
`APP=<app> modal serve main.py`

`<app>` is one of `ace-step`, `flux`, `krita`, `qwen`, or `wan`.

For `ace-step`, `flux`, `qwen`, and `wan`, open ComfyUI in the browser using the web function UI URL.

For `krita`, use the web function UI URL as the server URL for the Krita AI Diffusion plugin (https://github.com/Acly/krita-ai-diffusion).

## Examples
https://github.com/user-attachments/assets/1c3cc3f8-69a2-4bb4-82b9-4bc597f6eaf6

The example above also used Real-ESRGAN (https://github.com/xinntao/Real-ESRGAN) and Flowframes (https://github.com/n00mkrad/flowframes).
