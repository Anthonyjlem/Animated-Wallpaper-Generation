"""
Run models in ComfyUI in the browser

Usage:
`APP=<app> modal serve main.py` and open ComfyUI in the browser
    <app> is one of ace-step, flux, krita, qwen, or wan
For ace-step, flux, qwen, and wan, open ComfyUI in the browser using the web function UI URL.
For krita, use the web function UI URL as the server URL for the plugin.
"""

import os
import subprocess

import modal

from app_builders.ace_step_comfy_app_builder import ACEStepComfyAppBuilder
from app_builders.flux_comfy_app_builder import FluxComfyAppBuilder
from app_builders.krita_comfy_app_builder import KritaComfyAppBuilder
from app_builders.qwen_comfy_app_builder import QwenComfyAppBuilder
from app_builders.wan_comfy_app_builder import WanComfyAppBuilder


REDOWNLOAD_MODELS = False  # set to True to force all of the models to be re-downloaded; useful to force download new models

APP_REGISTRY = {
    "ace-step": ACEStepComfyAppBuilder,
    "flux": FluxComfyAppBuilder,
    "krita": KritaComfyAppBuilder,
    "qwen": QwenComfyAppBuilder,
    "wan": WanComfyAppBuilder,
}


try:
    app_name = os.environ["APP"]
except:
    raise ValueError("Environment variable `APP` was not set")

app_builder = APP_REGISTRY[app_name.lower().strip()]()
app = app_builder.build_app()
app_builder.print_output_volume_usage()
@app.function(
    max_containers=1,  # limit interactive session to 1 container
    gpu=app_builder.gpu,
    volumes=app_builder.get_volumes(),
)
@modal.concurrent(max_inputs=10)  # required for UI startup process which runs several API calls concurrently
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)