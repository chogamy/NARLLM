import os

from datasets import load_from_disk
import lightning as L
from torch.utils.data import DataLoader

from .preprocess import PREPROCESS
from .load import LOAD


class DataModule(L.LightningDataModule):
    def __init__(self, args, tokenizer) -> None:
        super().__init__()

        self.args = args
        self.tokenizer = tokenizer
        self.prompt = "task1: predict the length of a target \ntask2: predict a target for the length"

    def setup(self, stage) -> None:
        if stage == "fit":
            dir = os.path.join(os.getcwd(), "data", self.args.data, "cache")

            remove_columns = [
                "your persona:",
                "partner's persona:",
                "dialogue",
                "target",
            ]

            # train set
            if os.path.exists(os.path.join(dir, "train")):
                self.train = load_from_disk(os.path.join(dir, "train"))
            else:
                os.makedirs(os.path.join(dir, "train"), exist_ok=True)
                self.train = LOAD[self.args.data]("train")

                self.train = self.train.map(
                    PREPROCESS[self.args.data],
                    remove_columns=remove_columns,
                    fn_kwargs={"tokenizer": self.tokenizer, "prompt": self.prompt},
                    load_from_cache_file=True,
                    keep_in_memory=True,
                    desc="Pre-processing",
                    batched=True,
                )

                self.train.save_to_disk(os.path.join(dir, "train"))

            # valid set
            if os.path.exists(os.path.join(dir, "valid")):
                self.valid = load_from_disk(os.path.join(dir, "valid"))
            else:
                os.makedirs(os.path.join(dir, "valid"), exist_ok=True)
                self.valid = LOAD[self.args.data]("validation")

                self.valid = self.valid.map(
                    PREPROCESS[self.args.data],
                    remove_columns=remove_columns,
                    fn_kwargs={"tokenizer": self.tokenizer, "prompt": self.prompt},
                    load_from_cache_file=True,
                    keep_in_memory=True,
                    desc="Pre-processing",
                    batched=True,
                )

                self.valid.save_to_disk(os.path.join(dir, "valid"))

        if stage == "test":
            self.test = LOAD[self.args.data]("test")

            self.test = self.test.map(
                PREPROCESS[self.args.data],
                remove_columns=[],
                fn_kwargs={"tokenizer": self.tokenizer, "prompt": self.prompt},
                load_from_cache_file=False,
                desc="Pre-processing",
                batched=True,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.args.trainer_args["limit_train_batches"]
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid, batch_size=self.args.trainer_args["limit_val_batches"]
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.args.trainer_args["limit_test_batches"]
        )

    def predict_dataloader(self):
        # what's the difference between this and test?
        return DataLoader(
            self.test, batch_size=self.args.trainer_args["limit_predict_batches"]
        )
