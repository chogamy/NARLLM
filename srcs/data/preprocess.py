from datasets import load_dataset


def conv_ai_2(split):
    # where is split?
    dataset = load_dataset("conv_ai_2", split=split)
    # preprocess here
    return dataset


PREPROCESS = {"conv_ai_2": conv_ai_2}
