import subprocess


def download_hf_file(repo, filename, cache_dir, save_dir, save_filename=None, token=None):
    """
    Downloads a file from HuggingFace and creates a symbolic link to it in the specified directory

    Args:
        repo (string): The HuggingFace repository
        filename (string): The path to the file to download in the HuggingFace repository
        cache_dir (stirng): The directory to download the file into
        save_dir (string): The directory to create the symbolic link in that points to the file downloaded into the cache directory
        save_filename (string, optional): The name of the symbolic link. Defaults to `filename`.
        token (string, optional): HuggingFace token to use for downloading the file
    """
    from huggingface_hub import hf_hub_download

    print(f"Downloading {repo}/{filename}")
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        cache_dir=cache_dir,
        token=token,
    )
    save_filename = filename if save_filename is None else save_filename
    subprocess.run(
        f"ln -s {path} {save_dir}/{save_filename}",
        shell=True,
        check=True,
    )


def download_hf_snapshot(repo, dir_name, cache_dir, save_dir, allow_patterns=[], ignore_patterns=[]):
    """
    Downloads a snapshot of a HuggingFace repository and creates a symbolic link to it in the specified directory

    Args:
        repo (string): The HuggingFace repository
        dir_name (string): The name of the symbolic link. Defaults to `filename`.
        cache_dir (stirng): The directory to download the snapshot into
        save_dir (string): The directory to create the symbolic link in that points to the snapshot downloaded into the cache
            directory
        allow_patterns (list[string]): All files in the repository matching the allowed patterns will be downlaoded. Defaults to
            all files being downloaded.
        ignore_patterns (list[string]): All files in the repository matching the ignore patterns will not be downlaoded. Defaults
            to no files being ignored.
    """
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo}")
    path = snapshot_download(
        repo_id=repo,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        cache_dir=cache_dir,
    )
    subprocess.run(
        f"ln -s {path} {save_dir}/{dir_name}",
        shell=True,
        check=True,
    )


def download_wget_file(url, filename, cache_dir, save_dir):
    """
    Downloads a file and creates a symbolic link to it in the specified directory

    Args:
        url (string): The URL of the file to download
        filename (string): The name you choose for the file
        cache_dir (stirng): The directory to download the file into
        save_dir (string): The directory to create the symbolic link in that points to the file downloaded into the cache directory
    """
    print(f"Downloading {url}")
    path = f"{cache_dir}/{filename}"
    subprocess.run(
        f"wget -q -O {path} {url}",
        shell=True,
        check=True,
    )
    subprocess.run(
        f"ln -s {path} {save_dir}/{filename}",
        shell=True,
        check=True,
    )