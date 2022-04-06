import torch
import numpy as np
import boto3
import os
import boto3
import random
from tqdm import tqdm
import wandb
import logging
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

# read train data as Pandas DataFrame https://iwslt-data.s3.ap-south-1.amazonaws.com/fr_en_train_data
train_df = pd.read_csv('fr_en_train.csv')
train_df = train_df[["en", "fr"]]
train_df.columns = ["input_text", "target_text"]

def add_verbosity(input, target):
  """
  input: list of source & target sequences
  output: processed source sequence based on the calculated length ratios 
  """
  ts_ratio = len(target)/len(input)
  if ts_ratio < 0.95:
    prefix = "short"
  elif ts_ratio >= 0.95 and ts_ratio <= 1.10:
    prefix = "normal"
  else:
    prefix = "long"
  return prefix + " " + input

# preprocess train data
train_df['input_text'] = train_df.progress_apply(
    lambda row: add_verbosity(row['input_text'], row['target_text']),
    axis=1
)

# sample train-validation (95-5)% split
eval_df = train_df.sample(frac=0.05).astype(str)
train_df = train_df.drop(index=eval_df.index).astype(str)

# log errors and warnings
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# training arguments
model_args = T5Args()
model_args.max_seq_length = 100
model_args.train_batch_size = 10
model_args.eval_batch_size = 10
model_args.num_train_epochs = 8
model_args.scheduler = "cosine_schedule_with_warmup"
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 10000
model_args.learning_rate = 0.0003
model_args.optimizer = 'Adafactor'
model_args.use_multiprocessing = False
model_args.fp16 = False
model_args.save_steps = -1
model_args.save_eval_checkpoints = False
model_args.no_cache = True
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.save_model_every_epoch = False
model_args.preprocess_inputs = False
model_args.use_early_stopping = True
model_args.num_return_sequences = 1
model_args.do_lower_case = True
model_args.output_dir = "all/"
model_args.best_model_dir = "all/mt5_translation_checkpoints"
model_args.wandb_project = "verbosity based MT5 en-fr fine-tune"

# Initialize T5Model
model = T5Model("mt5", "google/mt5-base", args=model_args)

# Train the model
model.train_model(train_df, eval_data=eval_df)

# Optional: Evaluate the model. We'll test it properly anyway.
results = model.eval_model(eval_df, verbose=True)