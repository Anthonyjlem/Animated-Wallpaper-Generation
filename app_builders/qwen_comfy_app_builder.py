"""
Class for building ComfyUI apps to run Qwen2.5-VL in the browser
"""


import subprocess

from app_builders.comfy_app_builder import ComfyAppBuilder
import comfy_utils


class QwenComfyAppBuilder(ComfyAppBuilder):
    """
    Class for building ComfyUI apps to run Qwen in the browser

    Example usage:
    ```
    app_builder = QwenComfyAppBuilder()
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
        self.gpu = "A100-80GB"
        self._app_name = "qwen-comfyui"
        self._output_vol_name = "qwen-comfyui-output"  # name of output volume

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
                "comfy node install --fast-deps ComfyUI-Qwen-VL",
                "comfy node install --fast-deps comfyui-custom-scripts",
        )
        return image

    def _hf_download(self, tokens={}):
        """
        Download models from HuggingFace

        Args:
            tokens (dict{string: string}): Tokens for downloading models from HuggingFace and Civitai
        """
        super()._hf_download(tokens=tokens)

        save_dir = f"{self._comfy_models_dir}/Qwen/Qwen-VL/Qwen2.5-VL-32B-Instruct"
    
        subprocess.run(
            f"mkdir -p {save_dir}",
            shell=True,
            check=True,
        )
    
        repo = "Qwen/Qwen2.5-VL-32B-Instruct"
        for i in range(18):
            comfy_utils.download_hf_file(repo, "model-000"+str(i+1).zfill(2)+"-of-00018.safetensors", self._cache_dir, save_dir)
        comfy_utils.download_hf_file(repo, "config.json", self._cache_dir, save_dir)
        comfy_utils.download_hf_file(repo, "tokenizer.json", self._cache_dir, save_dir)
        comfy_utils.download_hf_file(repo, "vocab.json", self._cache_dir, save_dir)
        comfy_utils.download_hf_file(repo, "merges.txt", self._cache_dir, save_dir)
        comfy_utils.download_hf_file(repo, "chat_template.json", self._cache_dir, save_dir)
        comfy_utils.download_hf_file(repo, "preprocessor_config.json", self._cache_dir, save_dir)
        comfy_utils.download_hf_file(repo, "generation_config.json", self._cache_dir, save_dir)
        comfy_utils.download_hf_file(repo, "tokenizer_config.json", self._cache_dir, save_dir)
        comfy_utils.download_hf_file(repo, "model.safetensors.index.json", self._cache_dir, save_dir)