from lib2to3.pgen2.tokenize import tokenize
import datasets
from transformers import AutoTokenizer
from s3fs import S3FileSystem

def add_verbosity(inputs: list, targets: list, test=False):
  processed_inputs = []
  for input, target in zip(inputs, targets):
    if test:
      input = 'normal' + ' ' + input
    else:
      ts_ratio = len(target)/len(input)
      if ts_ratio < 0.97:
        prefix = 'short'
        input = prefix + ' ' + input
      elif ts_ratio >= 0.97 or ts_ratio <= 1.05:
        prefix = 'normal'
        input = prefix + ' ' + input
      else:
        prefix = 'long'
        input = prefix + ' ' + input
    processed_inputs.append(input)
  return processed_inputs

def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  inputs = [ex[source_lang] for ex in batch["translation"]]
  targets = [ex[target_lang] for ex in batch["translation"]]
  inputs = add_verbosity(inputs, targets)
  model_inputs = tokenizer(inputs, max_length=encoder_max_length, padding=False, truncation=True)
  # Setup the tokenizer for targets
  with tokenizer.as_target_tokenizer():
      labels = tokenizer(targets, max_length=decoder_max_length, padding=False, truncation=True)
  model_inputs["decoder_input_ids"] = labels.input_ids
  model_inputs["decoder_attention_mask"] = labels.attention_mask
  model_inputs["labels"] = labels.input_ids.copy()
  
  return model_inputs

def process_data_to_model_inputs_test(batch):
  # tokenize the inputs and labels
  inputs = [ex[source_lang] for ex in batch["translation"]]
  targets = [ex[target_lang] for ex in batch["translation"]]
  inputs = add_verbosity(inputs, targets)
  model_inputs = tokenizer(inputs, max_length=encoder_max_length, padding=False, truncation=True)
  # Setup the tokenizer for targets
  with tokenizer.as_target_tokenizer():
      labels = tokenizer(targets, max_length=decoder_max_length, padding=False, truncation=True)
  model_inputs["decoder_input_ids"] = labels.input_ids
  model_inputs["decoder_attention_mask"] = labels.attention_mask
  model_inputs["labels"] = labels.input_ids.copy()

  return model_inputs

if __name__ == "__main__":
    encoder_max_length=128
    decoder_max_length=128
    source_lang = "en"
    target_lang = "de"

    print("Importing Dataset")
    dataset = datasets.load_dataset('wmt16', 'de-en')
    model_checkpoint = "xlm-roberta-base"
    print("Initializing XLM-Roberta Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    print("Tokenizing Validation Data")
    column_names = dataset['validation'].column_names
    tokenized_val_data = dataset['validation'].map(
        process_data_to_model_inputs,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on validation dataset",
    )

    print("Tokenizing Test Data")
    column_names = dataset['test'].column_names
    tokenized_test_data = dataset['test'].map(
        process_data_to_model_inputs_test,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on test dataset",
    )

    print("Tokenizing Train Data")
    column_names = dataset['train'].column_names
    tokenized_train_data = dataset['train'].map(
        process_data_to_model_inputs,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on train dataset",
    )
    s3 = S3FileSystem(key='AKIA4QB2WTN57SCTNAGG', secret='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')

    tokenized_train_data.save_to_disk('s3://mtacl/tokenized_data_enc_dec_xlmr/train', fs=s3)
    tokenized_val_data.save_to_disk('s3://mtacl/tokenized_data_enc_dec_xlmr/validation', fs=s3)
    tokenized_test_data.save_to_disk('s3://mtacl/tokenized_data_enc_dec_xlmr/test', fs=s3)