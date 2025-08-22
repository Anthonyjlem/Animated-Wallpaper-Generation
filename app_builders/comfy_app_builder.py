"""
Base class for building ComfyUI apps to run in the browser
"""


import json

import modal


class ComfyAppBuilder:
    """
    Base class for building ComfyUI apps to run in the browser

    Example usage:
    ```
    app_builder = ComfyAppBuilder()
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
        Initializes an instance of ComfyAppBuilder
        """
        self.gpu = "T4"

        self._app_name = "comfyui"
        self._output_vol_name = "comfyui-output"  # name of output volume

        self._comfy_dir = "/root/comfy/ComfyUI"
        self._comfy_models_dir = f"{self._comfy_dir}/models"
        self._comfy_output_dir = f"{self._comfy_dir}/output"
        self._cache_dir = "/cache"  # mount volume here and download models into this volume

        self._image = None
        self._volumes = {}

    ##############################################################################
    #                             PUBLIC METHODS                                 #
    ##############################################################################
    def build_app(self, redownload_models=False):
        """
        Builds a Modal App for running ComfyUI in the browser

        Args:
            redownload_models (bool): Set to True to force all of the models to be re-downloaded; useful to force download new
                models

        Returns:
            app (modal.App): The app
        """
        self._build_image_and_volumes(redownload_models=redownload_models)
        app = modal.App(name=self._app_name, image=self._image)
        return app

    def get_volumes(self, redownload_models=False):
        """
        Gets the volumes associated with the image running in the app

        Args:
            redownload_models (bool): Set to True to force all of the models to be re-downloaded; useful to force download new
                models

        Returns:
            self._volumes (dict{string: modal.Volume}): The volumes
        """
        if not self._volumes:
            self._build_image_and_volumes(redownload_models=redownload_models)
        return self._volumes

    def print_output_volume_usage(self):
        """
        Prints messages instructing the user how to get and delete output generations
        """
        print(f"`modal volume get {self._output_vol_name} <file>` to download output generations")
        print(f"`modal volume delete {self._output_vol_name} -y` to delete the output volume")

    ##############################################################################
    #                             PRIVATE METHODS                                #
    ##############################################################################
    def _build_image_and_volumes(self, redownload_models=False):
        """
        Builds the image and volumes for the app

        Args:
            redownload_models (bool): Set to True to force all of the models to be re-downloaded; useful to force download new
                models
        """
        image = self._build_image()
        image = self._install_comfy_nodes(image)
        image = self._post_install_dep(image)
        image = self._copy_files(image)
        image, cache_vol = self._download_models(image, redownload_models=redownload_models)
        image, output_vol = self._create_output_vol(image)
        self._image = image
        self._volumes = {self._cache_dir: cache_vol, self._comfy_output_dir: output_vol}

    def _build_image(self):
        """
        Installs ComfyUI and other required dependencies.
    
        We use [comfy-cli](https://github.com/Comfy-Org/comfy-cli) to install ComfyUI and its dependencies.
    
        Returns:
            image (modal.Image): The built image
        """
        image = modal.Image.debian_slim(python_version="3.11")
        image = image.apt_install("git")  # install git to clone ComfyUI
        image = image.apt_install("wget")  # install wget to download model weights from civitai
        image = image.pip_install("fastapi[standard]==0.115.4")  # install web dependencies
        image = image.pip_install("comfy-cli")  # install comfy-cli
        # Install huggingface_hub with hf_transfer support to speed up model downloads
        image = image.pip_install("huggingface_hub[hf_transfer]")
        # Use comfy-cli to install ComfyUI and its dependencies
        image = image.run_commands("comfy --skip-prompt install --fast-deps --nvidia")
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
        return image

    def _copy_files(self, image):
        """
        Copies local files into the image
    
        Args:
            image (modal.Image): The image to copy files into
    
        Returns:
            image (modal.Image): The image with the copied files
        """
        image = image.add_local_python_source("comfy_utils", copy=True)
        return image.add_local_file("tokens.json", "/root/tokens.json", copy=True)

    def _hf_download(self, tokens={}):
        """
        Download models from HuggingFace

        Args:
            tokens (dict{string: string}): Tokens for downloading models from HuggingFace and Civitai
        """
        pass

    def _download_models(self, image, redownload_models=False):
        """
        Download models and create a cache volume
    
        By persisting the cache to a Volume, you avoid re-downloading the models every time you rebuild your image.
    
        Args:
            image (modal.Image): The image to download models into and create a cache volume for
            redownload_models (bool): Set to True to force all of the models to be re-downloaded; useful to force download new
                models
    
        Returns:
            image (modal.Image): The image with the downloaded models and cache volume
            vol (modal.Volume): The cache volume
        """
        with open("tokens.json", "r") as f:
            tokens = json.load(f)  # HuggingFace and Civitai tokens for downloading models
        vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
        image = image.env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
        image = image.run_function(
            self._hf_download,
            # Persist the HF cache to a Modal Volume so future runs don't need to re-download models
            volumes={self._cache_dir: vol},
            force_build=redownload_models,
            kwargs={"tokens":tokens},
            )
        return image, vol

    def _create_output_vol(self, image):
        """
        Creates an output volume and deletes the output directory in the image
    
        Sets up the output directory following https://gist.github.com/Jinnai/5e3292393413c70b3cedebe0fdb1108d
    
        Args:
            image (modal.Image): The image to create an output volume for
    
        Returns:
            image (modal.Image): The image with its ComfyUI output directory deleted
            vol (modal.Volume): The output volume
        """
        output_vol = modal.Volume.from_name(self._output_vol_name, create_if_missing=True)
        image = image.run_commands(f"rm -rf {self._comfy_output_dir}")  # needs to be empty for Volume mount to work
        return image, output_vol