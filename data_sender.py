import os
import sys

import pandas as pd
from torch.utils.data import DataLoader

import decompose_col_val
from EmbedModel import EmbedModel
from dataset import collate_fn, MergedMatchingDataset
from dataset import MergedMatchingDataset

def _make_example(left_val, right_val, type='l', label=0):
    if type == 'l':
        neighbor = right_val
        neighbor_ids = [neighbor[0]]
        center_example = left_val
        neighbor_examples = [neighbor]
        neighbor_masks = [1] * len(neighbor_ids)
    else:
        raise NotImplementedError
    labels = [label]

    example = {
        "type": type,
        "center": center_example,
        "neighbors": neighbor_examples,
        "neighbors_mask": neighbor_masks,
        "labels": labels,
    }

    return example

def _read_csv(path):
    columns = pd.read_csv(path).columns
    type = {}
    for name in columns:
        if name == 'id':
            continue
        type[name] = str
    data = pd.read_csv(path, dtype=type)
    data = data.fillna(" ")
    return data

# tableA_path = 'data/amazon_google/tableA.csv'
# tableB_path = 'data/amazon_google/tableB.csv'
# test_path = 'data/amazon_google/test.csv'
# train_path = 'data/amazon_google/train.csv'
# val_path = 'data/amazon_google/valid.csv'


def main(input_path):
    ditto_input = input_path#'data/amazon_google/test_ditto.txt'
    with open(ditto_input, encoding="utf8") as file:
        candidates = [line.rstrip() for line in file]

    list_candidates = []

    for i in candidates:
        df, label = decompose_col_val.decompose_srt_to_full_df(i)
        left_col = [i for i in df.columns.values if "left" in i]
        right_col = [i for i in df.columns.values if "right" in i]

        l_values = df[left_col].values.tolist()[0]#get all values
        l_values.insert(0, -1)#add an id (simbolic) to follow the input
        r_values = df[right_col].values.tolist()[0]#get all values
        r_values.insert(0, -1)#add an id (simbolic) to follow the input

        output = _make_example(left_val=l_values, right_val=r_values, label=label)
        list_candidates.append(output)
    return list_candidates

# tableA = _read_csv(tableA_path)
# tableB = _read_csv(tableB_path)
# useful_field_num = len(tableA.columns) - 1
#
# test_dataset = MergedMatchingDataset(test_path, tableA, tableB, other_path=[train_path, val_path])
# test_iter = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn, shuffle=False)
#
# for j, batch in enumerate(test_iter):
#     print("J", j)
#     print("data: ", batch)


if __name__ == '__main__':
    main('data/amazon_google/test_ditto.txt')