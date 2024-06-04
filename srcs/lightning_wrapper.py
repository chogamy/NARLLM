import lightning as L

# from transformers import AutoModel, AutoTokenizer


class LightningWrapper(L.LightningModule):
    def __init__(self, model) -> None:
        super().__init__()

        self.model = model

    def forward(self):
        assert 0

    def training_step(self):
        assert 0

    def configure_optimizers(self):
        assert 0
