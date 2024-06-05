import os
from typing import Union
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
from itertools import islice
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
from tqdm import tqdm

# WANDB
import wandb
wandb.init()
# wandb_outputs_table = wandb.Table(columns=["input_text", "student_output_text", "teacher_output_text"])

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\033[93m\033[1mDevice is: {device}\033[0m")

# Step 1: Load the teacher model (TinyLlama as a LMHeadModel)
teacher_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# teacher_model_path = "mistralai/Mistral-7B-v0.3"
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path).to(device)
teacher_model.eval()


# Step 2: Define the student model (a smaller transformer/MAMBA model with a LM head)

# for sanity check, here is a TinyLlama student model that is identical to teacher but without the pre-trained weights and half the hidden size
sanity_student_config = AutoConfig.from_pretrained(teacher_model_path)
sanity_student_config.num_hidden_layers = sanity_student_config.num_hidden_layers // 4


# MAMBA student model
def get_mamba_model(path: str = None):
    if path:
         return MambaLMHeadModel.from_pretrained(path, device=device, dtype=teacher_dtype)
    config_data = {
        "d_model": 2560,
        "n_layer": teacher_model.config.num_hidden_layers // 4, # 22 in case of TinyLlama-1.1B
        "vocab_size": teacher_model.config.vocab_size,
        "ssm_cfg": {},
        "rms_norm": True,
        "residual_in_fp32": True,
        "fused_add_norm": True,
        "pad_vocab_size_multiple": 8
    }
    config = MambaConfig(**config_data)
    param = next(teacher_model.parameters())
    teacher_dtype = param.dtype
    mamba_student_model = MambaLMHeadModel(config,
            initializer_cfg=None,
            device=device,
            dtype=teacher_dtype,
            )
    return mamba_student_model

vocab_size = teacher_model.config.vocab_size



# Step 3: Knowledge Distillation

temperature = 2.0  # Temperature for softmax computation
alpha = 0.8  # The weight of the distillation loss


HF_PADDING_IGNORE = -100



def init_dataloader(batch_size: int, max_length: int, partition: str = "train"):

    dataset_path = "wikitext-2-v1"
    dataset = load_dataset("wikitext", dataset_path, streaming=True) #

    
    # Load the teacher tokenizer
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, use_fast=True)

    # Tokenize the dataset
    def tokenize_function(examples):
        return teacher_tokenizer(examples["text"], truncation=True,
                                 padding=False,
                                #   padding="max_length", max_length=max_length, return_tensors="pt"
                                  )

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=teacher_tokenizer,
        mlm=False,  # Set to True if using Masked Language Modeling
        pad_to_multiple_of=8  # Optional, can pad to the nearest multiple of 8 for efficiency
    )
    
    # Create the data loader
    data_loader = DataLoader(tokenized_datasets[partition], batch_size=batch_size, collate_fn=data_collator)
    return data_loader, teacher_tokenizer.pad_token_id


def print_model_parameters(model_name: str, model: Union[AutoModelForCausalLM, MambaLMHeadModel]):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}")
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

def logits_to_tokens(logits):
    """Convert logits to token ids."""
    return torch.argmax(logits, dim=-1)

def distill_knowledge(teacher_model: AutoModelForCausalLM, student_model: MambaLMHeadModel, optimizer: torch.optim.Optimizer, batch_size: int, max_length: int, limit: int=1000, epochs: int=5, load_chkpt: bool=False, model_path: str=None):
    if load_chkpt:
        student_model.load_state_dict(torch.load(model_path))

    first_batch = True
    log_interval = 200
    
    # print the number of parameters in both models
    print_model_parameters(teacher_model_path, teacher_model)
    print_model_parameters("MAMBA Student Model", student_model)
    for epoch in range(epochs):

        running_loss = 0
        running_distillation_loss = 0
        running_cross_entropy_loss = 0

        dataloader, pad_token_id = init_dataloader(batch_size, max_length)
        for batch_idx, batch in tqdm(enumerate(islice(dataloader, limit))):
            batched_input_ids = batch['input_ids'].to(device)
            
            inputs = batched_input_ids[:, :-1].contiguous().to(device)
            labels = batched_input_ids[:, 1:].contiguous().to(device)
            labels[labels == pad_token_id] = HF_PADDING_IGNORE

            batched_attention_mask = batch['attention_mask'].to(device)
            attention_mask = batched_attention_mask[:, :-1].contiguous().to(device)
            
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=inputs,
                                                 attention_mask=attention_mask
                                                ).logits.to(device)
            if isinstance(student_model, MambaLMHeadModel):
                student_outputs = student_model(input_ids=inputs,
                                                ).logits.to(device)
            else:
                student_outputs = student_model(input_ids=inputs,
                                                attention_mask=attention_mask
                                                ).logits.to(device) 

            if first_batch:
                print(f"Student logits shape: {student_outputs.shape}")
                print(f"Teacher logits shape: {teacher_outputs.shape}")
                first_batch = False

            # Compute the distillation loss based on https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
            distillation_loss = nn.KLDivLoss(reduction="batchmean")(
                torch.log_softmax(student_outputs / temperature, dim=-1),
                torch.softmax(teacher_outputs / temperature, dim=-1),
            ) * (temperature ** 2)

            student_label_loss = nn.CrossEntropyLoss(ignore_index=HF_PADDING_IGNORE)(student_outputs.view(-1, student_outputs.size(-1)), labels.view(-1))
            
            loss = alpha * distillation_loss + (1 - alpha) * student_label_loss
            running_loss += loss.item()
            running_distillation_loss += distillation_loss.item()
            running_cross_entropy_loss += student_label_loss.item()

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=0.5)
            
            optimizer.step()

            if batch_idx % log_interval == 0:
                wandb.log({"epoch": epoch, "runningl_loss": running_loss / log_interval})
                wandb.log({"epoch": epoch, "running_distillation_loss": running_distillation_loss / log_interval})
                wandb.log({"epoch": epoch, "runnning_cross_entropy_loss": running_cross_entropy_loss / log_interval})
                running_loss = 0
                running_distillation_loss = 0
                running_cross_entropy_loss = 0
                
                
                
        if os.path.exists("./checkpoints") == False:
            os.mkdir("./checkpoints")
        torch.save(student_model.state_dict(), f"./checkpoints/student_chkpt_epoch_{epoch}.pt")

        # evaluate the student model
        evaluate(student_model)


# Training Loop
def train(limit: int = 1000, batch_size: int = 4, max_length: int = 128, epochs: int = 5,
           learning_rate: float = 5e-5, load_chkpt: bool=False, load_hf_model: bool=False, model_path: str=None, is_mamba: bool=False):   
    # assert that if either load_chkpt or load_hf_model is True but not both
    assert not (load_chkpt and load_hf_model), "Both load_chkpt and load_hf_model cannot be True at the same time"

    teacher_model.eval()
    if load_hf_model:
        if not is_mamba:
            student_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        else:
            student_model = get_mamba_model(path=model_path)
    else:
        if not is_mamba:
            student_model = AutoModelForCausalLM.from_config(sanity_student_config).to(device)
        else:
            student_model = get_mamba_model()

    student_model.train()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    distill_knowledge(teacher_model, student_model, optimizer, batch_size, max_length, limit=limit, epochs=epochs,
                       load_chkpt=load_chkpt, model_path=model_path)
    # save the student model 
    student_model.save_pretrained(f"student_model_full_trained_epoch_{epochs}_lr_{learning_rate}_mxln_{max_length}")


# Evaluate the student model
def evaluate(model_or_path: Union[str, AutoModelForCausalLM, MambaLMHeadModel]):

    # evaluate the student model using the test dataset
    if isinstance(model_or_path, str):
        student_model = AutoModelForCausalLM.from_pretrained(model_or_path).to(device)
    else:
        student_model = model_or_path
    
    student_model.eval()
    dataloader, pad_token_id = init_dataloader(4, 128, "test")
    
    # evalua using the test dataset
    running_loss = 0
    log_interval = 100
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        batched_input_ids = batch['input_ids'].to(device)
        inputs = batched_input_ids[:, :-1].contiguous().to(device)
        labels = batched_input_ids[:, 1:].contiguous().to(device)
        labels[labels == pad_token_id] = HF_PADDING_IGNORE

        batched_attention_mask = batch['attention_mask'].to(device)
        attention_mask = batched_attention_mask[:, :-1].contiguous().to(device)

        if isinstance(student_model, MambaLMHeadModel):
            student_outputs = student_model(input_ids=inputs,
                                        ).logits.to(device)
        else:
            student_outputs = student_model(input_ids=inputs,
                                            attention_mask=attention_mask
                                            ).logits.to(device)

        student_label_loss = nn.CrossEntropyLoss(ignore_index=HF_PADDING_IGNORE)(student_outputs.view(-1, student_outputs.size(-1)), labels.view(-1))
        running_loss += student_label_loss.item()

        if batch_idx % log_interval == 0:
            
            wandb.log({"test_loss": running_loss / log_interval})
            running_loss = 0

        

    wandb.log({"average_test_loss": total_loss / total_batches})
    wandb.log({"average_test_perplexity": torch.exp(total_loss / total_batches).item()})
