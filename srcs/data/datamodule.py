import lightning as L
from torch.utils.data import DataLoader

from .preprocess import PREPROCESS


class DataModule(L.LightningDataModule):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args

        self.sys_prompt = None

        # self.input_prompt = "1. Predict the target sequence length for input of the given task \ntask: {}\ninput: {} \ntarget length: {}"
        # self.target_prompt = "2. Generate target for predicted length\ntarget: {}"
        # self.context_prompt = ""

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train = PREPROCESS[self.args.data]("train")
            self.valid = PREPROCESS[self.args.data]("validation")

        if stage == "test":
            self.test = PREPROCESS[self.args.data]("test")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.args.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.args.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.args.batch_size)

    def predict_dataloader(self):
        # what's the difference between this and test?
        return DataLoader(self.test, batch_size=self.args.batch_size)


def get_datamodule(args):
    dm = DataModule(args)
    dm.prepare_data()
    dm.setup(args.mode)

    return dm
