"""
Class for building ComfyUI apps to run Flux in the browser
"""


from app_builders.comfy_app_builder import ComfyAppBuilder
import comfy_utils


class FluxComfyAppBuilder(ComfyAppBuilder):
    """
    Class for building ComfyUI apps to run Flux in the browser

    Example usage:
    ```
    app_builder = FluxComfyAppBuilder()
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
    ```
    """
    def __init__(self):
        """
        Initializes an instance of ACEStepComfyAppBuilder
        """
        super().__init__()
        self.gpu = "T4"
        self._app_name = "flux-comfyui"
        self._output_vol_name = "flux-comfyui-output"  # name of output volume

    ##############################################################################
    #                             PUBLIC METHODS                                 #
    ##############################################################################

    ##############################################################################
    #                             PRIVATE METHODS                                #
    ##############################################################################
    def _hf_download(self, tokens={}):
        """
        Download models from HuggingFace

        Args:
            tokens (dict{string: string}): Tokens for downloading models from HuggingFace and Civitai
        """
        super()._hf_download(tokens=tokens)
        repo = "comfyanonymous/flux_text_encoders"
        save_dir = f"{self._comfy_models_dir}/text_encoders"
        comfy_utils.download_hf_file(repo, "t5xxl_fp8_e4m3fn_scaled.safetensors", self._cache_dir, save_dir)
        comfy_utils.download_hf_file(repo, "clip_l.safetensors", self._cache_dir, save_dir)
        comfy_utils.download_hf_file("Comfy-Org/Lumina_Image_2.0_Repackaged",
                                     "split_files/vae/ae.safetensors",
                                     self._cache_dir,
                                     f"{self._comfy_models_dir}/vae",
                                     "ae.safetensors")
        comfy_utils.download_hf_file("black-forest-labs/FLUX.1-schnell",
                                     "flux1-schnell.safetensors",
                                     self._cache_dir,
                                     f"{self._comfy_models_dir}/unet",
                                     token=tokens["HF_TOKEN"])