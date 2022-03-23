# -*- coding: utf-8 -*-
"""fine tune mt5 en-fr .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ERnW0wN4b7I0UyoeBpDvxSh0WwON0Z-v
"""

# !pip install simpletransformers boto3 wandb --quiet

# !pip install datasets --quiet

import torch
import numpy as np
import boto3
import os
import random
# from tqdm.notebook import tqdm
from tqdm import tqdm
import datasets
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

# For logging loss
wandb.login(key="c7deb1bb77ce9433eb246d460385f363659145a8")

session = boto3.Session(
    aws_access_key_id='AKIA4QB2WTN57SCTNAGG',
    aws_secret_access_key='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',
)
s3 = session.resource('s3')
s3.meta.client.download_file(Bucket='mtacl', Key='paws_fr_train', Filename='paws_fr_train.csv')

train = pd.read_csv('paws_fr_train.csv')
train.dropna(inplace=True)
train.rename(columns = {
    'sentence1': 'input_text',
    'sentence2': 'target_text'
}, inplace=True)

# output_list = []
# for instance in tqdm(paw_x_fr['train'], total=paw_x_fr['train'].shape[0]):
#   output_list.append(instance)

# output_df = pd.DataFrame(output_list)
# output_df.drop(['id', 'label'], axis=1, inplace=True)
# output_df.columns = ["input_text", "target_text"]
# output_df.to_csv("paws_fr_train", index=False)

def add_verbosity(input, target, test=False):
  ts_ratio = len(target)/len(input)
  if test:
    prefix = "par normal"
  else:
    if ts_ratio < 0.95:
      prefix = "par short"
    elif ts_ratio >= 0.95 and ts_ratio <= 1.10:
      prefix = "par normal"
    else:
      prefix = "par long"
  return prefix + " " + input

train['input_text'] = train.progress_apply(
    lambda row: add_verbosity(row['input_text'], row['target_text']),
    axis=1
)

#Train 95% / Validation 5% Split
validation = train.sample(frac=0.05).astype(str)
train = train.drop(index=validation.index).astype(str)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


train_df = train
eval_df = validation

train_df["prefix"] = ""
eval_df["prefix"] = ""

model_args = T5Args()
# model_args.max_seq_length = 100
model_args.train_batch_size = 32
model_args.eval_batch_size = 16
model_args.num_train_epochs = 5
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
model_args.save_model_every_epoch = True
model_args.preprocess_inputs = False
model_args.use_early_stopping = True
model_args.num_return_sequences = 1
model_args.n_gpu = 4
model_args.do_lower_case = True
model_args.output_dir = "para_model/"
model_args.best_model_dir = "para_model/best_model"
model_args.wandb_project = "prompt based MT5 fr paraphrase fine-tune 2"

model = T5Model("mt5", "checkpoint-7935-epoch-1", args=model_args)
#model.model.load_state_dict(torch.load("../input/semifinalyoruba/outputs/best_model/pytorch_model.bin"))

# Train the model
model.train_model(train_df, eval_data=eval_df)

# Optional: Evaluate the model. We'll test it properly anyway.
# results = model.eval_model(eval_df, verbose=True)

# !unzip /content/mt5_checkpoitns_fr.zip
# model_args = T5Args()
# model_args.max_length = 100
# model_args.length_penalty = 2.5
# model_args.repetition_penalty = 1.5
# model_args.num_beams = 5

# model = T5Model("mt5", "all/best_model", args=model_args)
# preds = model.predict(validation.input_text.values.tolist())
# validation["preds"] = preds