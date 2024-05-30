import os
import copy

from datasets import load_dataset, Dataset


def ConvAI2(split):
    if split == "train":
        path = os.path.join(
            os.getcwd(), "data", "ConvAI2", "train_both_original_no_cands.txt"
        )
    if split in ["validation", "test"]:
        path = os.path.join(
            os.getcwd(), "data", "ConvAI2", "valid_both_original_no_cands.txt"
        )

    with open(path, "r") as f:
        lines = [line for line in f]

    windows = []
    window = None
    for line in lines:
        if line.split(" ", 1)[0] == "1":
            if window is not None:
                windows.append(window)

            window = {"your persona:": [], "partner's persona:": [], "dialogue": []}

        else:
            if "your persona:" in line:
                window["your persona:"].append(line.split("your persona:")[1].strip())
            elif "partner's persona:" in line:
                window["partner's persona:"].append(
                    line.split("partner's persona:")[1].strip()
                )
            else:
                text = line.split(" ", 1)[1]
                text1, text2 = text.split("\t")

                text1 = f"you: {text1}".strip()
                text2 = f"partner: {text2}".strip()
                window["dialogue"].append(text1)
                window["dialogue"].append(text2)
    if window is not None:
        windows.append(window)
        window = {"your persona:": [], "partner's persona:": [], "dialogue": []}
    # strip -> no \n on right
    # is OK?

    # check aug data
    # 제대로 안된거같다
    aug_windows = []
    for window in windows:
        base_window = {k: v for k, v in window.items()}
        base_window["dialogue"] = base_window["dialogue"][:1]

        rests = window["dialogue"][1:]

        for rest in rests:
            base_window["target"] = rest

            aug_window = copy.deepcopy(base_window)

            aug_windows.append(aug_window)
            base_window["dialogue"].append(rest)

    dataset = Dataset.from_list(aug_windows)

    return dataset


LOAD = {"ConvAI2": ConvAI2}
