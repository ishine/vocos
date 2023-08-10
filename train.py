import torch
from pytorch_lightning.cli import LightningCLI


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True

    # Automatic instantiation
    cli = LightningCLI(run=False)
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
