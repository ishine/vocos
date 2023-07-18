from pytorch_lightning.cli import LightningCLI


if __name__ == "__main__":
    cli = LightningCLI(run=False)
    # cli.trainer._checkpoint_connector.restore()
    # cli.trainer.default_root_dir = ''
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule, ckpt_path="logs/lightning_logs/version_26/checkpoints/last.ckpt")
