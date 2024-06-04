import lightning as L
from transformers import AutoModel, AutoTokenizer


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


def get_model(args):
    model = AutoModel.from_pretrained(args.model)
    model = LightningWrapper(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
