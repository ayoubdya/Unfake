import os
from datasets import load_dataset

dataset = "ErfanMoosaviMonazzah/fake-news-detection-dataset-English"
name = dataset.split("/")[-1]

ds = load_dataset(dataset)
os.makedirs("data", exist_ok=True)

if isinstance(ds, dict):
  for split in ds.keys():
    ds[split].to_csv(f"data/{split}_{name}.csv", index=False)
else:
  ds.to_csv(f"data/{name}.csv", index=False)
