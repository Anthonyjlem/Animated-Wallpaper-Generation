"""
Class for building ComfyUI apps to run a checkpoint of Wan2.1 Image-to-Video for live wallpapers in the browser
"""


from app_builders.comfy_app_builder import ComfyAppBuilder
import comfy_utils


class WanComfyAppBuilder(ComfyAppBuilder):
    """
    Class for building ComfyUI apps to run a checkpoint of Wan2.1 Image-to-Video for live wallpapers in the browser

    Example usage:
    ```
    app_builder = WanComfyAppBuilder()
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
        self.gpu = "L40S"
        self._app_name = "wan-comfyui"
        self._output_vol_name = "wan-comfyui-output"  # name of output volume

    ##############################################################################
    #                             PUBLIC METHODS                                 #
    ##############################################################################

    ##############################################################################
    #                             PRIVATE METHODS                                #
    ##############################################################################
    def _install_comfy_nodes(self, image):
        """
        Downloads comfy nodes
    
        We use `comfy-cli` to download custom nodes. Use the [ComfyUI Registry](https://registry.comfy.org/) to find the specific
        custom node name to use with this command. See [this post](https://modal.com/blog/comfyui-custom-nodes) for more examples
        on how to install popular custom nodes like ComfyUI Impact Pack and ComfyUI IPAdapter Plus. Do the following to install a
        custom node from GitHub:

        ```
        image = image.add_local_dir(
            "<path to repo>",
            f"{COMFY_DIR}/custom_nodes/<node name>",
            copy=True
         )
         image = image.pip_install_from_requirements("<path to repo>/requirements.txt")
         ```

        Args:
            image (modal.Image): The image to install comfy nodes in
    
        Returns:
            image (modal.Image): The image with the comfy nodes installed
        """
        image = super()._install_comfy_nodes(image)
        image = image.run_commands(
            "comfy node install --fast-deps ComfyUI-GGUF",
            "comfy node install --fast-deps ComfyUI-WanStartEndFramesNative",
        )
        return image

    def _hf_download(self, tokens={}):
        """
        Download models from HuggingFace

        Args:
            tokens (dict{string: string}): Tokens for downloading models from HuggingFace and Civitai
        """
        super()._hf_download(tokens=tokens)

        repo = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
    
        filename = "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        save_filename = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        comfy_utils.download_hf_file(repo, filename, self._cache_dir, f"{self._comfy_models_dir}/text_encoders", save_filename)
        filename = "split_files/vae/wan_2.1_vae.safetensors"
        save_filename = "wan_2.1_vae.safetensors"
        comfy_utils.download_hf_file(repo, filename, self._cache_dir, f"{self._comfy_models_dir}/vae", save_filename)
        filename = "split_files/clip_vision/clip_vision_h.safetensors"
        save_filename = "clip_vision_h.safetensors"
        comfy_utils.download_hf_file(repo, filename, self._cache_dir, f"{self._comfy_models_dir}/clip_vision", save_filename)
        url = f"https://civitai.com/api/download/models/1873761?type=Model&format=GGUF&size=full&fp=fp32&token={tokens["CIVITAI_TOKEN"]}"
        comfy_utils.download_wget_file(url,
                                       "liveWallpaperFast_i2v14B720P.gguf",
                                       self._cache_dir,
                                       f"{self._comfy_models_dir}/diffusion_models")