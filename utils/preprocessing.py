import os
import pandas as pd
from tqdm import tqdm
import numpy as np

def convert_flow_fix_size(seq, max_length, padding):
    seq = seq[:max_length]
    t_arr = seq.tolist() + [padding]*(max_length-len(seq))
    return t_arr


def convert_to_df(df, x_key, y_key, method=None, max_length=30):
    ids = np.arange(len(df))
    xs = df[x_key]
    ys = df[y_key]
    if method is not None:
        xs = xs.apply(lambda x: method(x, max_length))
    result = pd.DataFrame({
        "id": ids,
        "x": xs,
        "y": ys,
    })
    return result


def pl_bert_preprocessing(x, max_length):
    padding = -2500
    x = convert_flow_fix_size(x, max_length, padding)
    x = [i + 1500 + 1000 for i in x]
    return x


def sketch_preprocessing(x, max_length):
    padding = 1500
    x = x + 1500
    return convert_flow_fix_size(x, max_length, padding)


def get_preprocessed_df(df_path, x_label, y_label, max_length, dataset_path):
    df = pd.read_parquet(df_path)
    origin_filename = df_path.split("/")[-1].split(".")[0]
    preprocessed_df = convert_to_df(df, x_label, y_label, max_length=max_length)
    preprocessed_df.to_parquet(os.path.join(dataset_path, f"preprocessed_{origin_filename}.parquet"))
    return preprocessed_df