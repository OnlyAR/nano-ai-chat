import click
from modelscope import snapshot_download
from pathlib import Path
from loguru import logger


@click.command()
@click.option("-m", "--model", help="remote model path", required=True)
@click.option("-o", "--output", help="local path to save model", default="model", show_default=True)
def download(model: str, output: str = "model"):
    local_path = Path(output)
    local_path.mkdir(parents=True, exist_ok=True)
    model_dir = snapshot_download(model, cache_dir=local_path)
    logger.info(f"Model downloaded to {model_dir}")


if __name__ == '__main__':
    download()
