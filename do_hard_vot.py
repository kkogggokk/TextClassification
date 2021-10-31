import numpy as np
import os
from tqdm import tqdm
import pandas as pd

root = "/private/000_kdigit/000_teamp/ensemble"
csv_paths = [os.path.join(root, path) for path in os.listdir(root) if '.csv' in path]
preds = None
for cp in tqdm(csv_paths):
    test_df = pd.read_csv(cp)
    if preds is None:
        preds = test_df['Category']
    else:
        preds += test_df['Category']
preds /= len(csv_paths)
preds = np.array((preds >= 0.5), dtype=np.int)

best = pd.read_csv("/private/000_kdigit/000_teamp/000_source/submission12.csv")
bpred = best['Category'].to_numpy()
print((preds != bpred).sum())
test_df['Category'] = preds
test_df.to_csv('ensembled.csv', index=False)
