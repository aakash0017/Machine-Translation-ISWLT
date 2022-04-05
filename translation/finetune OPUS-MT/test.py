# import libraries & dependencies
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, MarianMTModel, MarianTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from tqdm import tqdm 

# data processing
raw_datasets = load_dataset("enimai/must_c_french")
metric = load_metric("sacrebleu")

# dataset description
print(raw_datasets)

def add_verbosity(input_list, target_list):
  """
  input: list of source & target sequences
  output: processed source sequences with pre-appended "normal" prompt
  """
  processed_input = []
  for input, target in zip(input_list, target_list):
    prefix = "normal"
    input = prefix + " " + input
    processed_input.append(input)
  return processed_input

# preprocess MUST-C dataset
max_input_length = 128 
max_target_length = 128
source_lang = "en"
target_lang = "fr"
def preprocess_function(examples):
    inputs = examples["en"]
    targets = examples["fr"]
    inputs = add_verbosity(inputs, targets) # append appropriate prompts 
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# prediction on already fine-tuned model
model_checkpoint = "enimai/OPUS-mt-en-fr-finetuned-MUST-C"

# intialize tokenizer and fine-tuned model 
tokenizer = MarianTokenizer.from_pretrained(model_checkpoint)
model = MarianMTModel.from_pretrained(model_checkpoint)

# apply preprocessing function to raw-datasets
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# initialize DataLoader
test_dataloader = DataLoader(tokenized_datasets, batch_size=1, num_workers=0)

# generate model prediction
predictions = []
for batch in tqdm(test_dataloader, total=tokenized_datasets.shape[0]):
  translated = model.generate(**tokenizer(batch['en'], return_tensors="pt", padding=True))
  predictions.extend([tokenizer.decode(t, skip_special_tokens=True) for t in translated])