from datasets import load_dataset


def load_ocr_datasets():
    synth = load_dataset("wendlerc/CaptionedSynthText")
    icdar = load_dataset("dlxjj/ICDAR2015")
    return synth, icdar
