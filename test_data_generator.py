import json
import pandas as pd
path = "data/mini_dataset.json"

review_texts = []

with open(path, "r") as f:
    data = json.loads(f.read())
    for item in data:
        review_texts.append(item["reviewText"])

print(review_texts)

review_texts_df = pd.DataFrame(columns=["sentence"],data=review_texts)
review_texts_df.to_csv("data/test_data.csv")