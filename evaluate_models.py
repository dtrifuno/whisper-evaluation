import re

from evaluate import load
import pandas as pd

def strip_punctuation(text):
    return re.sub("[.,?!]", "", text).replace("-", " ").lower()

data = pd.read_csv("transcripts.csv", dtype=str, na_filter=False)
for column in data:
    data[column + "_stripped"] = data[column].map(strip_punctuation)

wer_metric = load("wer")
cer_metric = load("cer")
MODELS = ["small", "medium", "large-v2"]

results = {
        "model_name": [],
        "WER": [],
        "WER (w/o punct)": [],
        "CER": [],
        "CER (w/o punct)": []
    }

for model in MODELS:
    results["model_name"].append(model)
    wer = wer_metric.compute(references=data["stenogram"], predictions=data[model])
    results["WER"].append(wer)
    stripped_wer = wer_metric.compute(references=data["stenogram_stripped"], predictions=data[model + "_stripped"])
    results["WER (w/o punct)"].append(stripped_wer)
    cer = cer_metric.compute(references=data["stenogram"], predictions=data[model])
    results["CER"].append(cer)
    stripped_cer = cer_metric.compute(references=data["stenogram_stripped"], predictions=data[model + "_stripped"])
    results["CER (w/o punct)"].append(stripped_cer)

results = pd.DataFrame(results)
print(results)
