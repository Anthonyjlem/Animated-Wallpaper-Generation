"""
Class for building ComfyUI apps to run models for the Generative AI plugin for Krita (https://github.com/Acly/krita-ai-diffusion)
"""


import subprocess

from app_builders.comfy_app_builder import ComfyAppBuilder
import comfy_utils


class KritaComfyAppBuilder(ComfyAppBuilder):
    """
    Class for building ComfyUI apps to run models for the Generative AI plugin for Krita
    (https://github.com/Acly/krita-ai-diffusion)

    Example usage:
    ```
    app_builder = KritaComfyAppBuilder()
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
        self._app_name = "krita-comfyui"
        self._output_vol_name = "krita-comfyui-output"  # name of output volume

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
        image = image.apt_install("libgl1")  # for controlnet nodes 
        image = image.apt_install("libglib2.0-0")  # for controlnet nodes 
        return image

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
            "comfy node install --fast-deps comfyui_controlnet_aux",
            "comfy node install --fast-deps comfyui_ipadapter_plus",
            "comfy node install --fast-deps comfyui-inpaint-nodes",
            "comfy node install --fast-deps comfyui-tooling-nodes",
        )
        return image

    def _hf_download(self, tokens={}):
        """
        Download models from HuggingFace

        Args:
            tokens (dict{string: string}): Tokens for downloading models from HuggingFace and Civitai
        """
        super()._hf_download(tokens=tokens)
        self._illustrious_download()
        self._clip_vision_download()
        self._upscale_download()
        self._inpaint_download(tokens)
        self._controlnet_download()
        self._ipadapter_download()

    def _illustrious_download(self):
        """
        Download Illustrious XL v2.0
        """
        comfy_utils.download_hf_file("OnomaAIResearch/Illustrious-XL-v2.0",
                                     "Illustrious-XL-v2.0.safetensors",
                                     self._cache_dir,
                                     f"{self._comfy_models_dir}/checkpoints")

    def _clip_vision_download(self):
        """
        Download CLIP ViT-H
        """
        comfy_utils.download_hf_file("h94/IP-Adapter",
                                     "models/image_encoder/model.safetensors",
                                     self._cache_dir,
                                     f"{self._comfy_models_dir}/clip_vision",
                                     "clip-vision_vit-h.safetensors")

    def _upscale_download(self):
        """
        Download upscaling models
        """
        upscale_models_dir = f"{self._comfy_models_dir}/upscale_models"
    
        url = "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/4x-NMKD-YandereNeo.pth"
        comfy_utils.download_wget_file(url, "4x-NMKD-YandereNeo.pth", self._cache_dir, upscale_models_dir)
        repo = "Acly/Omni-SR"
        comfy_utils.download_hf_file(repo, "OmniSR_X2_DIV2K.safetensors", self._cache_dir, upscale_models_dir)
        comfy_utils.download_hf_file(repo, "OmniSR_X3_DIV2K.safetensors", self._cache_dir, upscale_models_dir)
        comfy_utils.download_hf_file(repo, "OmniSR_X4_DIV2K.safetensors", self._cache_dir, upscale_models_dir)
        repo = "Acly/hat"
        comfy_utils.download_hf_file(repo, "HAT_SRx4_ImageNet-pretrain.pth", self._cache_dir, upscale_models_dir)
        comfy_utils.download_hf_file(repo, "Real_HAT_GAN_sharper.pth", self._cache_dir, upscale_models_dir)

    def _inpaint_download(self, tokens):
        """
        Download AnimagineXL v3.1 Inpainting

        Args:
            tokens (dict{string: string}): Tokens for downloading models from HuggingFace and Civitai
        """
        save_dir = f"{self._comfy_models_dir}/inpaint"
        subprocess.run(
            f"mkdir -p {save_dir}",
            shell=True,
            check=True,
        )
        url = f"https://civitai.com/api/download/models/480117?type=Model&format=SafeTensor&size=pruned&fp=fp16&token={tokens["CIVITAI_TOKEN"]}"
        comfy_utils.download_wget_file(url, "animaginexl_v31Inpainting.safetensors", self._cache_dir, f"{save_dir}")

    def _controlnet_download(self):
        """
        Download ControlNets
        """
        controlnet_dir = f"{self._comfy_models_dir}/controlnet"
    
        repo = "Eugeoter/noob-sdxl-controlnet-scribble_pidinet"
        filename = "diffusion_pytorch_model.fp16.safetensors"
        save_filename = "noob-sdxl-controlnet-scribble_pidinet.fp16.safetensors"
        comfy_utils.download_hf_file(repo, filename, self._cache_dir, controlnet_dir, save_filename)
        repo = "Eugeoter/noob-sdxl-controlnet-lineart_anime"
        filename = "diffusion_pytorch_model.fp16.safetensors"
        save_filename = "noob-sdxl-controlnet-lineart_anime.fp16.safetensors"
        comfy_utils.download_hf_file(repo, filename, self._cache_dir, controlnet_dir, save_filename)
        repo = "Eugeoter/noob-sdxl-controlnet-softedge_hed"
        filename = "diffusion_pytorch_model.fp16.safetensors"
        save_filename = "noob-sdxl-controlnet-softedge_hed.fp16.safetensors"
        comfy_utils.download_hf_file(repo, filename, self._cache_dir, controlnet_dir, save_filename)
        repo = "Eugeoter/noob-sdxl-controlnet-canny"
        filename = "noob_sdxl_controlnet_canny.fp16.safetensors"
        comfy_utils.download_hf_file(repo, filename, self._cache_dir, controlnet_dir)
        repo = "Eugeoter/noob-sdxl-controlnet-depth_midas-v1-1"
        filename = "diffusion_pytorch_model.fp16.safetensors"
        save_filename = "noob-sdxl-controlnet-depth_midas-v1-1.fp16.safetensors"
        comfy_utils.download_hf_file(repo, filename, self._cache_dir, controlnet_dir, save_filename)
        repo = "Eugeoter/noob-sdxl-controlnet-normal"
        filename = "diffusion_pytorch_model.fp16.safetensors"
        save_filename = "noob-sdxl-controlnet-normal.fp16.safetensors"
        comfy_utils.download_hf_file(repo, filename, self._cache_dir, controlnet_dir, save_filename)
        repo = "windsingai/Illustrious-XL-openpose-test"
        filename = "openpose_s6000.safetensors"
        comfy_utils.download_hf_file(repo, filename, self._cache_dir, controlnet_dir)
        repo = "Eugeoter/noob-sdxl-controlnet-tile"
        filename = "diffusion_pytorch_model.fp16.safetensors"
        save_filename = "noob-sdxl-controlnet-tile.fp16.safetensors"
        comfy_utils.download_hf_file(repo, filename, self._cache_dir, controlnet_dir, save_filename)


    def _ipadapter_download(self):
        """
        Download IP-Adapters
        """
        save_dir = f"{self._comfy_models_dir}/ipadapter"
        subprocess.run(
            f"mkdir -p {save_dir}",
            shell=True,
            check=True,
        )
        comfy_utils.download_hf_file("h94/IP-Adapter",
                                     "sdxl_models/image_encoder/model.safetensors",
                                     self._cache_dir,
                                     f"{self._comfy_models_dir}/clip_vision",
                                     "clip-vision_vit-g.safetensors")
        comfy_utils.download_hf_file("kataragi/Noob_ipadapter", "ip_adapter_Noobtest_800000.bin", self._cache_dir, f"{save_dir}")