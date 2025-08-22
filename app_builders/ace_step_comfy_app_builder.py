"""
Class for building ComfyUI apps to run ACE-Step in the browser
"""


import subprocess

from app_builders.comfy_app_builder import ComfyAppBuilder
import comfy_utils


class ACEStepComfyAppBuilder(ComfyAppBuilder):
    """
    Class for building ComfyUI apps to run ACE-Step in the browser

    Example usage:
    ```
    app_builder = ACEStepComfyAppBuilder()
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
        self._app_name = "ace-step-comfyui"
        self._output_vol_name = "ace-step-comfyui-output"  # name of output volume

    ##############################################################################
    #                             PUBLIC METHODS                                 #
    ##############################################################################

    ##############################################################################
    #                             PRIVATE METHODS                                #
    ##############################################################################
    def _build_image(self):
        """
        Installs ComfyUI and other required dependencies.
    
        We use [comfy-cli](https://github.com/Comfy-Org/comfy-cli) to install ComfyUI and its dependencies.
    
        Returns:
            image (modal.Image): The built image
        """
        image = super()._build_image()
        # Dependencies for comfy node audiotools
        image = image.apt_install("sox")
        image = image.apt_install("ffmpeg")
        image = image.apt_install("libportaudio2")
        image = image.pip_install("sounddevice")
        image = image.pip_install("easydict")
        image = image.pip_install("torch-complex")
        return image

    def _post_install_dep(self, image):
        """
        Installs dependencies
    
        Installs dependencies that must be installed after the comfy nodes are installed to ensure the image has the correct
        versions
    
        Args:
            image (modal.Image): The image to install the dependencies in
    
        Returns:
            image (modal.Image): The image with the dependencies installed
        """
        image = super()._post_install_dep(image)
        # For audiotools; must happen after all numpy installations (last one is during ACE-Step comfy node)
        return image.pip_install("numpy==2.2")

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
            "comfy node install --fast-deps ace-step",
            "comfy node install --fast-deps audiotools",
            "comfy node install --fast-deps ComfyUI-Qwen3",
        )
        return image

    def _hf_download(self, tokens={}):
        """
        Download models from HuggingFace

        Args:
            tokens (dict{string: string}): Tokens for downloading models from HuggingFace and Civitai
        """
        super()._hf_download(tokens=tokens)
        self._ace_step_download()
        self._qwen_download()

    def _ace_step_download(self):
        """
        Downloads ACE-Step into the cache and creates a symbolic link to it in the correct directory for ComfyUI to use
        """
    
        save_dir = f"{self._comfy_models_dir}/TTS"
        subprocess.run(
            f"mkdir -p {save_dir}",
            shell=True,
            check=True,
        )
        comfy_utils.download_hf_snapshot("ACE-Step/ACE-Step-v1-3.5B",
                                         "ACE-Step-v1-3.5B",
                                         self._cache_dir,
                                         save_dir,
                                         allow_patterns=["*.json", "*.safetensors"])

    def _qwen_download(self):
        """
        Downloads Qwen3 into the cache and creates a symbolic link to it in the correct directory for ComfyUI to use
        """
    
        save_dir = f"{self._comfy_models_dir}/Qwen/Qwen/Qwen3-14B"
    
        subprocess.run(
            f"mkdir -p {save_dir}",
            shell=True,
            check=True,
        )
    
        repo = "Qwen/Qwen3-14B"
        for i in range(8):
            comfy_utils.download_hf_file(repo, "model-000"+str(i+1).zfill(2)+"-of-00008.safetensors", self._cache_dir, save_dir)
        comfy_utils.download_hf_file(repo, "config.json", self._cache_dir, save_dir)
        comfy_utils.download_hf_file(repo, "tokenizer.json", self._cache_dir, save_dir)
        comfy_utils.download_hf_file(repo, "vocab.json", self._cache_dir, save_dir)
        comfy_utils.download_hf_file(repo, "merges.txt", self._cache_dir, save_dir)
        comfy_utils.download_hf_file(repo, "generation_config.json", self._cache_dir, save_dir)
        comfy_utils.download_hf_file(repo, "tokenizer_config.json", self._cache_dir, save_dir)
        comfy_utils.download_hf_file(repo, "model.safetensors.index.json", self._cache_dir, save_dir)