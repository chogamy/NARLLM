import torch
from torch.nn import CrossEntropyLoss
import lightning as L


class LightningWrapper(L.LightningModule):
    def __init__(self, model, tokenizer, metric) -> None:
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.loss = CrossEntropyLoss()
        self.metric = metric

    def forward(self, batch):
        outputs = self.model(**batch)
        return outputs

    def training_step(self, batch, batch_id):
        length_batch = {}
        legnth_target = {}
        outputs = self.model(length_batch)
        loss1 = self.loss(outputs, legnth_target)

        target_batch = {}
        target_target = {}
        outputs = self.model(target_batch)
        loss2 = self.loss(outputs, target_target)

        loss = loss1 + loss2

        self.log_dict({"loss": loss}, prog_bar=True)

        return loss

    def nar_predict(self, batch):
        assert 0

    @torch.no_grad()
    def validation_step(self, batch, batch_id):
        model_input = {}
        target = {}

        outputs = self.model.nar_predict(model_input)

        result = self.metric(outputs, target)

        self.log_dict({})

    @torch.no_grad()
    def test_step(self, batch, batch_id):
        model_input = {}
        target = {}

        outputs = self.model.nar_predict(model_input)

        result = self.metric(outputs, target)

        self.log_dict({})

    def on_validation_epoch_end(self):
        self.eval()
        self.log_dict()

        self.metric.save()

    def on_test_epoch_end(self):
        self.eval()
        self.log_dict()

        self.metric.save()

    def configure_optimizers(self):
        assert 0
