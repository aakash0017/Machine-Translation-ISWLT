import torch
import numpy as np
import boto3
import os
import random
# from tqdm.notebook import tqdm
from tqdm import tqdm
import wandb
import logging
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
import os
import pandas as pd

import boto3
session = boto3.Session(
    aws_access_key_id='AKIA4QB2WTN57SCTNAGG',
    aws_secret_access_key='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',
)
s3 = session.resource('s3')
s3.meta.client.download_file(Bucket='mtacl', Key='fr_en_test_data', Filename='fr_en_test.csv')
  
test = pd.read_csv('fr_en_test.csv')

#Remove any possible duplicates
test = test.drop_duplicates(subset=["en", "fr"])

test = test[["en", "fr"]]
test.columns = ["input_text", "target_text"]

test['input_text'] = test.progress_apply(
    lambda row: add_verbosity(row['input_text'], row['target_text']),
    axis=1
)
test = test.astype(str)

def add_verbosity(input, target, test=True):
  ts_ratio = len(target.split(' '))/len(input.split(' '))
  if test:
    prefix = "normal"
  else:
    if ts_ratio < 0.95:
      prefix = "short"
    elif ts_ratio >= 0.95 and ts_ratio <= 1.10:
      prefix = "normal"
    else:
      prefix = "long"
  return prefix + " " + input

model_args = T5Args()
model_args.max_length = 100
model_args.length_penalty = 2.5
model_args.repetition_penalty = 1.5
model_args.num_beams = 5

model = T5Model("mt5", "all/checkpoint-15870-epoch-1", args=model_args)

preds = model.predict(test.input_text.values.tolist())
test["preds"] = preds
test.to_csv("fr_en_test_prediction.csv")

session = boto3.Session(
    aws_access_key_id='AKIA4QB2WTN57SCTNAGG',
    aws_secret_access_key='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',
)
s3 = session.resource('s3')
s3.meta.client.upload_file(Bucket='mtacl', Key='fr_en_test_prediction', Filename='fr_en_test_prediction.csv')
