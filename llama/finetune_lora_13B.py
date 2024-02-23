import torch

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random as rn
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, LlamaTokenizerFast, Trainer, TrainingArguments,EarlyStoppingCallback, AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorWithPadding
import pathlib
from peft import LoraConfig, LoftQConfig, prepare_model_for_kbit_training, get_peft_model

from transformers.trainer_callback import EarlyStoppingCallback
from datasets import load_metric
import os
from torch.optim import AdamW

print(torch.version.cuda, flush=True)
print(torch.cuda.is_available(), flush=True)
print("Total : ", torch.cuda.device_count(), flush=True)
print(torch.cuda.max_memory_allocated(), flush=True)

os.environ['WANDB_API_KEY'] = 'e819d741a0c770cb527b6ca091ee5d1b25a8222e'
os.environ['TRANSFORMERS_CACHE'] = '/home/bmohapat/data/.cache/huggingface/transformers'
os.environ['TORCH_EXTENSIONS_DIR'] = '/home/bmohapat/data/.cache/torch_extensions'
# Set the DS_SKIP_CUDA_CHECK environment variable to "1" (to skip CUDA check)
os.environ["DS_SKIP_CUDA_CHECK"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:4096'

SEED = 3407
rn.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = 'cuda'

train_df = pd.read_csv("../../data/meetup_d1_template_train_val/train_meetup_d1_template_short_image_descriptions.csv")
val_df = pd.read_csv("../../data/meetup_d1_template_train_val/val_meetup_d1_template_short_image_descriptions.csv")

print("train dataset shape", train_df.shape)
print("validation dataset shape", val_df.shape)
train_df.reset_index()
val_df.reset_index()

# Define custom dataset class
class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, outputs):
        self.tokenizer = tokenizer
        self.texts = texts
        self.outputs = outputs
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        if index in self.cached_data_dict:
            return self.cached_data_dict[index]
        
        text = self.texts[index]
        output = self.outputs[index]
        tokenized_text = self.tokenizer(
            text + " " + output,
            padding="max_length", 
            truncation=True,
            max_length=2048,
            return_tensors='pt'
        )

#         tokenized_output = self.tokenizer.encode_plus(
#             output,
#             padding='max_length',
#             truncation=True,
#             max_length=64,
#             return_tensors='pt'
#         )
        
        ret = {
            'input_ids': tokenized_text['input_ids'].squeeze(),
            'attention_mask': tokenized_text['attention_mask'].squeeze(),
            'decoder_input_ids': tokenized_text['input_ids'].clone().squeeze(),
            'labels': tokenized_text['input_ids'].clone().squeeze()
        }
        
        self.cached_data_dict[index] = ret
        
        return ret
    
# Define the training arguments
training_args = TrainingArguments(
    output_dir='/home/bmohapat/data/meetup_testing/llama13_lora/',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    warmup_steps=8,
    weight_decay=0.01,
    logging_dir='../../logs',
    logging_steps=1,
    evaluation_strategy='epoch',
    save_strategy='steps',
    save_steps=20,
    save_total_limit=10,
    learning_rate=2e-5,
    warmup_ratio=0.04,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    fp16=True,
    # bf16=True if torch.cuda.is_bf16_supported() else False,
    # fp16=False if torch.cuda.is_bf16_supported() else True,
    deepspeed='/home/bmohapat/github/LLM-Grounding-Study/code/Llama/deepspeed_config.json',
)


print("Loading the model...")
# tokenizer = LlamaTokenizer.from_pretrained("/home/bmohapat/pyllama_data/hf_weights/7B")
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-13b",
                padding_side="right",
                truncation_side="left")

model = AutoModelForCausalLM.from_pretrained("/data/almanach/user/wantoun/scratch/models/huggyllama-llama-13b/")
model.config.use_cache = False
print(model.config)

# Get LORA configurations and wrap the model in it for efficient finetuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.1,
    bias="lora_only",
    modules_to_save=["decode_head"],
)

print_trainable_parameters(f"Trainable parameters before LORA : {model}", flush=True)
# model_prepared = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
lora_model = get_peft_model(model, lora_config)
print_trainable_parameters(f"Trainable parameters after LORA : {lora_model}", flush=True)

# Add pad token if necessary
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.unk_token})

print("Creating the dataset...")
# Create train and validation datasets
train_dataset = TextDataset(tokenizer, train_df['inputs'].values.tolist(), train_df['outputs'].values.tolist())
val_dataset = TextDataset(tokenizer, val_df['inputs'].values.tolist(), val_df['outputs'].values.tolist())

print("created dataset")

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
data_collator_with_padding = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# Create the Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator_with_padding
)

# Should show false if we are using deepspeed
print("place_model_on_device", flush=True)
print(trainer.place_model_on_device, flush=True)
print("Max memory allocated cuda", torch.cuda.max_memory_allocated(), flush=True)

# Train the model
if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
# trainer.save_state()
# trainer.save_model()

model_id = "llama-lora-13b"
lora_model.save_pretrained(model_id)


# Test the trained model
test_results = trainer.evaluate(eval_dataset=val_dataset)
print("Test Results:", test_results)
