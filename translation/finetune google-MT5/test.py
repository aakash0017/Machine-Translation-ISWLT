# import libraries & dependenciesimport torch
from sys import prefix
import numpy as np
import boto3
import os
import boto3
import random
from tqdm import tqdm
import wandb
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
import os
import pandas as pd

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(7)
tqdm.pandas()

# for logging loss to wandb.ai
access_key = "" # enter wandb secret_accces_key
wandb.login(key=access_key)

# download training data from s3
session = boto3.Session(
    aws_access_key_id='AKIA4QB2WTN57SCTNAGG',
    aws_secret_access_key='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',
)
s3 = session.resource('s3')
s3.meta.client.download_file(Bucket='mtacl', Key='fr_en_train_data', Filename='fr_en_test.csv')

# for logging loss to wandb.ai
access_key = "" # enter wandb secret_accces_key
wandb.login(key=access_key)

# download training data from s3
session = boto3.Session(
    aws_access_key_id='AKIA4QB2WTN57SCTNAGG',
    aws_secret_access_key='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',
)
s3 = session.resource('s3')
s3.meta.client.download_file(Bucket='mtacl', Key='fr_en_train_data', Filename='fr_en_test.csv')

# read train data as pandas DataFrame
test_df = pd.read_csv('fr_en_test.csv')
test_df = test_df[["en", "fr"]]
test_df.columns = ["input_text", "target_text"]

def add_verbosity(input):
  """
  input: list of source & target sequences
  output: processed source sequence based on the calculated length ratios 
  """
  prefix = "normal"
  return prefix + " " + input

# preprocess test data
test_df['input_text'] = test_df.progress_apply(
    lambda row: add_verbosity(row['input_text']),
    axis=1
)

# model checkpoints
model_name = "enimai/mt5-mustc-fr"

model_args = T5Args()
model_args.max_length = 100
model_args.length_penalty = 2.5
model_args.repetition_penalty = 1.5
model_args.num_beams = 5

# intialize model
model = T5Model("mt5", model_name, args=model_args)

# predict output
preds = model.predict(test_df.input_text.values.tolist())

test_df["preds"] = preds
# save generated predictions
test_df.to_csv("test-output.csv")