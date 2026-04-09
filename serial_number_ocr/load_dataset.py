from datasets import load_dataset


def load_ocr_datasets():
    dataset = load_dataset(
        "wendlerc/CaptionedSynthText",
        split="train",
        streaming=True,
    )
    return dataset
