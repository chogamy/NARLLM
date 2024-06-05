import lightning as L
from torch.utils.data import DataLoader

from .preprocess import PREPROCESS
from .load import LOAD


class DataModule(L.LightningDataModule):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args

        self.prompt = "task1: predict the length of a target \ntask2: predict a target for the length"

    def setup(self, stage: str, tokenizer) -> None:
        if stage == "fit":
            self.train = LOAD[self.args.data]("train")
            self.valid = LOAD[self.args.data]("validation")

            self.train = self.train.map(
                PREPROCESS[self.args.data],
                remove_columns=[],
                fn_kwargs={"tokenizer": tokenizer, "prompt": self.prompt},
                load_from_cache_file=True,
                desc="Pre-processing",
                batched=True,
            )

            self.valid = self.valid.map(
                PREPROCESS[self.args.data],
                remove_columns=[],
                fn_kwargs={"tokenizer": tokenizer, "prompt": self.prompt},
                load_from_cache_file=True,
                desc="Pre-processing",
                batched=True,
            )

        if stage == "test":
            self.test = LOAD[self.args.data]("test")

            self.test = self.test.map(
                PREPROCESS[self.args.data],
                remove_columns=[],
                fn_kwargs={"tokenizer": tokenizer, "prompt": self.prompt},
                load_from_cache_file=False,
                desc="Pre-processing",
                batched=True,
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.args.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.args.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.args.batch_size)

    def predict_dataloader(self):
        # what's the difference between this and test?
        return DataLoader(self.test, batch_size=self.args.batch_size)
