import torch
from torch import optim, nn
from torch.nn import functional as F
import lightning as L


class LightningWrapper(L.LightningModule):
    def __init__(self, model, config, tokenizer, metric) -> None:
        super().__init__()

        self.model = model
        self.length_classifier = nn.Linear(config.hidden_size, 200)
        self.tokenizer = tokenizer

        self.metric = metric

    def forward(self, batch):
        outputs = self.model(**batch)
        return outputs

    def training_step(self, batch, batch_id):
        length_batch = {}
        legnth_target = {}
        outputs = self.model(length_batch)
        loss1 = F.cross_entropy(outputs, legnth_target)

        target_batch = {}
        target_target = {}
        outputs = self.model(target_batch)
        loss2 = F.cross_entropy(outputs, target_target)

        loss = loss1 + loss2

        self.log_dict({"loss": loss}, prog_bar=True)

        return loss

    def nar_predict(self, batch):

        b, l = batch["input_ids"].shape
        device = batch["input_ids"].device

        outputs = self.model(**batch, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]

        # find last token ids
        last_token_ids = torch.sum(batch["attention_mask"], dim=1) - 1
        batch_ids = torch.arange(b, device=last_token_ids.device)

        last_token_hidden = hidden[batch_ids, last_token_ids, :]

        # predict target length
        length_logits = self.length_classifier(last_token_hidden)
        lengths = torch.argmax(length_logits, dim=-1)

        # make mask tokens
        mask_input_ids = self.tokenizer(
            [f"{str(length)}{length*self.tokenizer.eos_token}" for length in lengths],
            padding=False,
            truncation=False,
        )["input_ids"]
        # TODO
        # {str(length)}, {length*self.tokenizer.eos_token} respectively!
        # to extract only predicted tokens for masked part

        start_end_ids = []

        # insert mask tokens
        for i in range(b):
            start = last_token_ids[i] + 1
            end = start + len(mask_input_ids[i])
            mask = torch.tensor(mask_input_ids[i], device=device)

            if end > l:
                new_input_ids = torch.cat([batch["input_ids"][i][:start], mask])
                new_input_ids = new_input_ids[-l:]
                batch["input_ids"][i] = new_input_ids
                batch["attention_mask"][i][:] = 1
                start_end_ids.append((end - 1, l))
            else:
                batch["input_ids"][i][start:end] = mask
                batch["attention_mask"][i][start:end] = 1
                start_end_ids.append((start, end))

        outputs = self.model(**batch)

        sequence_ids = torch.argmax(torch.softmax(outputs.logits, dim=-1), dim=-1)
        sequences = [
            self.tokenizer.decode(
                seq_id[start_end_id[0] : start_end_id[1]], skip_special_tokens=True
            )
            for seq_id, start_end_id in zip(sequence_ids, start_end_ids)
        ]

        outputs = {"logits": outputs.logits, "sequences": sequences}

        return outputs

    @torch.no_grad()
    def validation_step(self, batch, batch_id):
        model_input = {}
        for k, v in batch.items():
            if "length_" in k:
                model_input[k.replace("length_", "")] = v

        target = model_input.pop("target")

        outputs = self.nar_predict(model_input)

        assert 0

        result = self.metric(outputs, target)

        self.log_dict({})

    @torch.no_grad()
    def test_step(self, batch, batch_id):
        model_input = {}
        target = {}

        outputs = self.nar_predict(model_input)

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
        # optimizer
        # lr_scheduler
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
