import torch
from pytorch_lightning.cli import LightningCLI


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True

    # Hack for Resume
    ckpt_path = None
    # ckpt_path = "logs/lightning_logs/version_XXX/checkpoints/last.ckpt"

    # Automatic instantiation
    cli = LightningCLI(run=False)
    if ckpt_path is None:
        print("Training from scratch.")
        cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
    else:
        print(f"Resume training from {ckpt_path}.")
        cli.trainer.fit(model=cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
