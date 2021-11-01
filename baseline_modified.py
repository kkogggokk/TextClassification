import os
import pdb
import argparse
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from tqdm import tqdm, trange

from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    BartForSequenceClassification,
    BartTokenizer,
    AutoTokenizer,
    AutoModel,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pickle
import random


def make_id_file(task, tokenizer):
    def make_data_strings(file_name):
        data_strings = []
        with open(os.path.join('/private/000_kdigit/000_teamp/001_dataset/', file_name), 'r', encoding='utf-8') as f:
            id_file_data = [tokenizer.encode(line.lower()) for line in tqdm(f.readlines())]
        for item in tqdm(id_file_data):
            data_strings.append(' '.join([str(k) for k in item]))
        return data_strings
    print('it will take some times...')
    train_pos = make_data_strings('sentiment.train.1')
    train_neg = make_data_strings('sentiment.train.0')
    dev_pos = make_data_strings('sentiment.dev.1')
    dev_neg = make_data_strings('sentiment.dev.0')
    print('make id file finished!')
    return train_pos, train_neg, dev_pos, dev_neg


model_arch = "bert"
if model_arch == "bert":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
elif model_arch == "sci_bert":
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
elif model_arch == "bart":
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartForSequenceClassification.from_pretrained('facebook/bart-large')
elif model_arch == "roberta": 
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')

def cast_to_int(sample):
    return [int(word_id) for word_id in sample]

with open(os.path.join('/private/000_kdigit/000_teamp/001_dataset/', "sentiment.dev.1"), 'r', encoding='utf-8') as f:
    dev_sents = [line for line in tqdm(f.readlines())]
with open(os.path.join('/private/000_kdigit/000_teamp/001_dataset/', "sentiment.dev.0"), 'r', encoding='utf-8') as f:
    dev_sents += [line for line in tqdm(f.readlines())]
dev_sents = np.array(dev_sents)

if os.path.isfile(f"{model_arch}_train_pos.pickle"):
    with open(f'{model_arch}_train_pos.pickle', 'rb') as f:
        train_pos = pickle.load(f)
    with open(f'{model_arch}_train_neg.pickle', 'rb') as f:
        train_neg = pickle.load(f)
    with open(f'{model_arch}_valid_pos.pickle', 'rb') as f:
        dev_pos = pickle.load(f)
    with open(f'{model_arch}_valid_neg.pickle', 'rb') as f:
        dev_neg = pickle.load(f)
else:
    train_pos, train_neg, dev_pos, dev_neg = make_id_file('yelp', tokenizer)
    with open(f"{model_arch}_train_pos.pickle", "wb") as f:
        pickle.dump(train_pos, f, pickle.HIGHEST_PROTOCOL)
    with open(f"{model_arch}_train_neg.pickle", "wb") as f:
        pickle.dump(train_neg, f, pickle.HIGHEST_PROTOCOL)
    with open(f"{model_arch}_valid_pos.pickle", "wb") as f:
        pickle.dump(dev_pos, f, pickle.HIGHEST_PROTOCOL)
    with open(f"{model_arch}_valid_neg.pickle", "wb") as f:
        pickle.dump(dev_neg, f, pickle.HIGHEST_PROTOCOL)


class SentimentDataset(object):
    def __init__(self, tokenizer, pos, neg):
        self.tokenizer = tokenizer
        self.data = []
        self.label = []

        for pos_sent in pos:
            self.data += [self._cast_to_int(pos_sent.strip().split())]
            self.label += [[1]]
        for neg_sent in neg:
            self.data += [self._cast_to_int(neg_sent.strip().split())]
            self.label += [[0]]

    def _cast_to_int(self, sample):
        return [int(word_id) for word_id in sample]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return np.array(sample), np.array(self.label[index])

train_dataset = SentimentDataset(tokenizer, train_pos, train_neg)
dev_dataset = SentimentDataset(tokenizer, dev_pos, dev_neg)

for i, item in enumerate(train_dataset):
    print(item)
    if i == 10:
        break

def collate_fn_sentiment(samples):
    input_ids, labels = zip(*samples)
    max_len = max(len(input_id) for input_id in input_ids)
    max_len = min(15, max_len)
    input_ids = [list(reversed(list(reversed(input_id))[:max_len])) for input_id in input_ids]
    sorted_indices = np.argsort([len(input_id) for input_id in input_ids])[::-1]

    input_ids = pad_sequence([torch.tensor(input_ids[index]) for index in sorted_indices],
                             batch_first=True)
    attention_mask = torch.tensor(
        [[1] * len(input_ids[index]) + [0] * (max_len - len(input_ids[index])) for index in
         sorted_indices])
    token_type_ids = torch.tensor([[0] * len(input_ids[index]) for index in sorted_indices])
    position_ids = torch.tensor([list(range(len(input_ids[index]))) for index in sorted_indices])
    labels = torch.tensor(np.stack(labels, axis=0)[sorted_indices])

    return input_ids, attention_mask, token_type_ids, position_ids, labels

def collate_fn_sentiment_valid(samples):
    input_ids, labels = zip(*samples)
    max_len = max(len(input_id) for input_id in input_ids)
    max_len = min(15, max_len)
    input_ids = [list(reversed(list(reversed(input_id))[:max_len])) for input_id in input_ids]
    unsorted_indices = [i for i in range(len(input_ids))]

    input_ids = pad_sequence([torch.tensor(input_ids[index]) for index in unsorted_indices],
                             batch_first=True)
    attention_mask = torch.tensor(
        [[1] * len(input_ids[index]) + [0] * (max_len - len(input_ids[index])) for index in
         unsorted_indices])
    token_type_ids = torch.tensor([[0] * len(input_ids[index]) for index in unsorted_indices])
    position_ids = torch.tensor([list(range(len(input_ids[index]))) for index in unsorted_indices])
    labels = torch.tensor(np.stack(labels, axis=0)[unsorted_indices])
    
    return input_ids, attention_mask, token_type_ids, position_ids, labels

train_batch_size=64
eval_batch_size=64

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=train_batch_size,
                                           shuffle=True, collate_fn=collate_fn_sentiment,
                                           pin_memory=True, num_workers=0)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=eval_batch_size,
                                         shuffle=False, collate_fn=collate_fn_sentiment_valid,
                                         num_workers=0)

# random seed
random_seed=42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

model.train()
train_epoch = 3
learning_rate = 2e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=len(train_loader) * train_epoch, pct_start=0.1)

def compute_acc(predictions, target_labels):
    return (np.array(predictions) == np.array(target_labels)).mean()

writer = SummaryWriter()
best_acc = 0
best_weights = None
failed_sents = []
failed_ans = []
for epoch in range(train_epoch):
    with tqdm(train_loader, unit="batch") as tepoch:
        for iteration, (input_ids, attention_mask, token_type_ids, position_ids, labels) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            position_ids = position_ids.to(device)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()

            output = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids,
                           labels=labels)

            loss = output.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            step = iteration + len(train_loader) * epoch
            writer.add_scalar("Train/Loss", loss.item(), step)
            tepoch.set_postfix(loss=loss.item())
            if iteration != 0 and iteration % int(len(train_loader) / 5) == 0:
                # Evaluate the model five times per epoch
                with torch.no_grad():
                    model.eval()
                    valid_losses = []
                    predictions = []
                    target_labels = []
                    for input_ids, attention_mask, token_type_ids, position_ids, labels in tqdm(dev_loader,
                                                                                                desc='Eval',
                                                                                                position=1,
                                                                                                leave=None):
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        token_type_ids = token_type_ids.to(device)
                        position_ids = position_ids.to(device)
                        labels = labels.to(device, dtype=torch.long)

                        output = model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       labels=labels)

                        logits = output.logits
                        loss = output.loss
                        valid_losses.append(loss.item())

                        batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
                        batch_labels = [int(example) for example in labels]

                        predictions += batch_predictions
                        target_labels += batch_labels

                acc = compute_acc(predictions, target_labels)
                failed = (np.array(predictions) != np.array(target_labels))
                valid_loss = sum(valid_losses) / len(valid_losses)
                writer.add_scalar("Valid/Loss", valid_loss, step)
                writer.add_scalar("Valid/Acc", acc, step)
                if acc > best_acc:
                    best_acc = acc
                    best_weights = model.state_dict()
                    failed_sents = dev_sents[failed]
                    failed_ans = np.array(target_labels)[failed]
with open("failed", 'w') as f:
    for fs, fa in zip(failed_sents, failed_ans):
        f.write(f"{fs}   {fa}")
model.load_state_dict(best_weights)
torch.save(best_weights, "best_model.pth")

import pandas as pd
test_df = pd.read_csv('/private/000_kdigit/000_teamp/001_dataset/test_no_label.csv')
test_dataset = test_df['Id']

def make_id_file_test(tokenizer, test_dataset):
    data_strings = []
    id_file_data = [tokenizer.encode(sent.lower()) for sent in test_dataset]
    for item in id_file_data:
        data_strings.append(' '.join([str(k) for k in item]))
    return data_strings

test = make_id_file_test(tokenizer, test_dataset)

class SentimentTestDataset(object):
    def __init__(self, tokenizer, test):
        self.tokenizer = tokenizer
        self.data = []

        for sent in test:
            self.data += [self._cast_to_int(sent.strip().split())]

    def _cast_to_int(self, sample):
        return [int(word_id) for word_id in sample]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return np.array(sample)

test_dataset = SentimentTestDataset(tokenizer, test)

def collate_fn_sentiment_test(samples):
    input_ids = samples
    max_len = max(len(input_id) for input_id in input_ids)
    max_len = min(15, max_len)
    input_ids = [list(reversed(list(reversed(input_id))[:max_len])) for input_id in input_ids]
    # sorted_indices = np.argsort([len(input_id) for input_id in input_ids])[::-1]
    unsorted_indices = [i for i in range(len(input_ids))]

    input_ids = pad_sequence([torch.tensor(input_ids[index]) for index in unsorted_indices],
                             batch_first=True)
    attention_mask = torch.tensor(
        [[1] * len(input_ids[index]) + [0] * (max_len - len(input_ids[index])) for index in
         unsorted_indices])
    token_type_ids = torch.tensor([[0] * len(input_ids[index]) for index in unsorted_indices])
    position_ids = torch.tensor([list(range(len(input_ids[index]))) for index in unsorted_indices])

    return input_ids, attention_mask, token_type_ids, position_ids

test_batch_size = 32
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                          shuffle=False, collate_fn=collate_fn_sentiment_test,
                                          num_workers=0)


with torch.no_grad():
    model.eval()
    predictions = []
    for input_ids, attention_mask, token_type_ids, position_ids in tqdm(test_loader,
                                                                        desc='Test',
                                                                        position=1,
                                                                        leave=None):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        position_ids = position_ids.to(device)

        output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids,
                       position_ids=position_ids)

        logits = output.logits
        batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
        predictions += batch_predictions

test_df['Category'] = predictions
test_df.to_csv('submission.csv', index=False)
