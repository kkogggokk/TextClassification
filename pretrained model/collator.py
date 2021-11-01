import torch

class Collator():

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, sample):
        text = [s['text'] for s in sample]
        label = [s['label'] for s in sample]

        encode = self.tokenizer(text, padding=True,truncation=True, return_tensors="pt", max_length=self.max_length) 

        result_dict = {'ids': encode['ids'], 'attention_mask': encode['attention_mask'], 'labels': torch.tensor(label, dtype=torch.long)}

        return result_dict