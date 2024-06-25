import os
from typing import Union
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling, get_scheduler, MambaForCausalLM
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from itertools import islice
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
from tqdm.auto import tqdm
import numpy as np
import argparse
from memory import MemoryTrace
from kl_div_loss import KLDivLoss
from uld_loss import ULDLoss
from pprint import pprint
import time
# WANDB
import wandb



class EmbeddingProjectionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.projection(x)
    



accelerator = Accelerator(log_with="wandb")
hf_mamba_path = "state-spaces/mamba-790m-hf"
teacher_model_path = "meta-llama/Meta-Llama-3-8B"
tiny_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# teacher_model_path = tiny_model_path #TODO

# teacher_model_path = "mistralai/Mistral-7B-v0.3"

def get_teacher_model(path: str):
    model = AutoModelForCausalLM.from_pretrained(path) 
    print_model_parameters("Teacher", model)
    accelerator.print(model.config)
    return model


def get_sanity_student_model(path: str=None):
    if path:
        model = AutoModelForCausalLM.from_pretrained(path)
    else:
        # reduce the number of parameters in the student model
        config = AutoConfig.from_pretrained(tiny_model_path)
        teacher_model_config = AutoConfig.from_pretrained(teacher_model_path)
        config.eos_token_id = teacher_model_config.eos_token_id
        config.bos_token_id = teacher_model_config.bos_token_id
        config.vocab_size = teacher_model_config.vocab_size
        config.num_hidden_layers = config.num_hidden_layers
        model = AutoModelForCausalLM.from_config(config)
    # print memory foorprint and number of parameters
    # adapt TinyLlama-1.1B to the teacher model
    if accelerator.is_main_process: 
        print_model_parameters("Sanity Student", model)
        pprint(model.config)
    return model



# MAMBA student model
def get_mamba_model(path: str = None):
    teacher_model = get_teacher_model(teacher_model_path)
    param = next(teacher_model.parameters())
    teacher_dtype = param.dtype
    if path:
        mamba_student_model = MambaForCausalLM.from_pretrained(path)
        config = mamba_student_model.config
    else:
        config = AutoConfig.from_pretrained(hf_mamba_path)
        config.vocab_size = teacher_model.config.vocab_size
        config.torch_dtype = teacher_dtype
        mamba_student_model = MambaForCausalLM(config)
    if accelerator.is_main_process: 
        print_model_parameters("MAMBA", mamba_student_model)
        pprint(config)
    return mamba_student_model



# Step 3: Knowledge Distillation

temperature = 2.0  # Temperature for softmax computation
alpha = 0.5  # The weight of the distillation loss


HF_PADDING_IGNORE = -100



def init_dataloader(batch_size: int, max_length: int, partition: str = "train", student_tokenizer_path: str = None):

    dataset_path = "wikitext-2-v1"

    if student_tokenizer_path:
        dataset = load_dataset("wikitext", dataset_path)
        student_tokenizer = AutoTokenizer.from_pretrained(student_tokenizer_path, use_fast=True)
        student_tokenizer.pad_token = student_tokenizer.eos_token
        def student_tokenize_function(examples):
            return student_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        
        student_tokenized_datasets = dataset.map(student_tokenize_function, batched=True, remove_columns=["text"])
        student_data_collator = DataCollatorForLanguageModeling(
            tokenizer=student_tokenizer,
            mlm=False,  # Set to True if using Masked Language Modeling
            pad_to_multiple_of=8  # Optional, can pad to the nearest multiple of 8 for efficiency
        )
        student_data_loader = DataLoader(student_tokenized_datasets[partition], batch_size=batch_size, collate_fn=student_data_collator)
    
    dataset = load_dataset("wikitext", dataset_path)
    # Load the teacher tokenizer
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, use_fast=True)
    # add padding token to the tokenizer
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    # Tokenize the dataset
    def teacher_tokenize_function(examples):
        return teacher_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

    teacher_tokenized_datasets = dataset.map(teacher_tokenize_function, batched=True, remove_columns=["text"])

    teacher_data_collator = DataCollatorForLanguageModeling(
        tokenizer=teacher_tokenizer,
        mlm=False,  # Set to True if using Masked Language Modeling
        pad_to_multiple_of=8  # Optional, can pad to the nearest multiple of 8 for efficiency
    )
    
    # Create the data loader
    teacher_data_loader = DataLoader(teacher_tokenized_datasets[partition], batch_size=batch_size, collate_fn=teacher_data_collator)

    if student_tokenizer_path:
        return teacher_data_loader, student_data_loader, teacher_tokenizer.pad_token_id
    
    return teacher_data_loader, teacher_tokenizer.pad_token_id


def print_model_parameters(model_name: str, model: Union[AutoModelForCausalLM, MambaLMHeadModel]):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Model: {model_name}")
    accelerator.print(f"Total Parameters: {total_params}")
    accelerator.print(f"Trainable Parameters: {trainable_params}")
    accelerator.print(f"Total Memory Footprint: {total_params * 4 / 1024 / 1024} MB")

def logits_to_tokens(logits):
    """Convert logits to token ids."""
    return torch.argmax(logits, dim=-1)

def distill_knowledge(teacher_model: AutoModelForCausalLM, student_model: Union[MambaLMHeadModel, AutoModelForCausalLM], optimizer: torch.optim.Optimizer,
                       batch_size: int, max_length: int,
                         limit: int=1000, epochs: int=5, load_chkpt: bool=False, model_path: str=None, accumulation_steps: int = 1):


    if load_chkpt:
        student_model.load_state_dict(torch.load(model_path))

    first_batch = True
    log_interval = 100
    
    running_loss = 0
    running_distillation_loss = 0
    running_cross_entropy_loss = 0

    if model_path is None:
        loss_fn = KLDivLoss(reduction='mean', temperature=temperature, ignore_idx=HF_PADDING_IGNORE, distillation_loss_weight=alpha)
        accelerator.print("Using KL Divergence Loss")
    else:
        loss_fn = ULDLoss(distillation_weight=alpha, crossentropy_weight=1-alpha, ignore_idx=HF_PADDING_IGNORE, teacher_temperature=temperature, student_temperature=temperature, skip_student_eos=True, skip_teacher_eos=True)
        accelerator.print("Using ULD Loss")

    dataloader = init_dataloader(batch_size, max_length, "train", student_tokenizer_path=model_path)
    if  model_path:
        teacher_train_dataloader, student_train_dataloader, pad_token_id = dataloader
    else:
        teacher_train_dataloader, pad_token_id = dataloader

    eval_dataloader, _ = init_dataloader(batch_size, max_length, "test")
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=10, num_training_steps=epochs * len(teacher_train_dataloader))

    teacher_train_dataloader, student_train_dataloader, eval_dataloader, student_model, teacher_model, optimizer = accelerator.prepare(teacher_train_dataloader, student_train_dataloader, eval_dataloader, student_model, teacher_model, optimizer)

    other_dataloader = teacher_train_dataloader if not model_path else student_train_dataloader
    steps_per_epoch = len(teacher_train_dataloader)
    for epoch in range(epochs):
        with MemoryTrace() as mem_trace:
            progress_bar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=steps_per_epoch//accumulation_steps,
                                dynamic_ncols=True, disable=not accelerator.is_local_main_process)
            for batch_idx, (teacher_batch, student_batch)  in enumerate(islice(zip(teacher_train_dataloader, other_dataloader), limit)):
                teacher_batched_input_ids = teacher_batch['input_ids']
                student_batched_input_ids = student_batch['input_ids']
                
                teacher_inputs = teacher_batched_input_ids[:, :-1].contiguous()
                student_inputs = student_batched_input_ids[:, :-1].contiguous()

                student_labels = student_batched_input_ids[:, 1:].contiguous()
                student_labels[student_labels == pad_token_id] = HF_PADDING_IGNORE

                teacher_labels = teacher_batched_input_ids[:, 1:].contiguous()
                teacher_labels[teacher_labels == pad_token_id] = HF_PADDING_IGNORE

                batched_attention_mask = teacher_batch['attention_mask']
                attention_mask = batched_attention_mask[:, :-1].contiguous()
                
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids=teacher_inputs,
                                                    attention_mask=attention_mask
                                                    )
                if isinstance(student_model, MambaLMHeadModel):
                    raise NotImplementedError("non HF Mamba model not supported")
                else:
                    student_outputs = student_model(input_ids=student_inputs,
                                                    attention_mask=attention_mask,
                                                    labels=student_labels)

                if first_batch:
                    accelerator.print(f"Student logits shape: {student_outputs.logits.shape}")
                    accelerator.print(f"Teacher logits shape: {teacher_outputs.logits.shape}")
                    first_batch = False
                
                
                loss, student_label_loss, distillation_loss = loss_fn(student_outputs, teacher_outputs, student_labels, teacher_labels)
                student_label_loss = student_label_loss.mean() / accumulation_steps
                distillation_loss = distillation_loss.mean() / accumulation_steps
                loss = loss.mean() / accumulation_steps
                running_loss += loss.detach().float()
                running_distillation_loss += distillation_loss.detach().float()
                running_cross_entropy_loss += student_label_loss.detach().float()
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.zero_grad()
                    accelerator.backward(loss)
                    # Gradient clipping
                    accelerator.clip_grad_norm_(student_model.parameters(), max_norm=0.5)
                    optimizer.step()
                    # log once even though using accelerate
                    
                    accelerator.log({"epoch": epoch, "running_loss": running_loss.item()})
                    accelerator.log({"epoch": epoch, "running_distillation_loss": running_distillation_loss.item()})
                    accelerator.log({"epoch": epoch, "running_cross_entropy_loss": running_cross_entropy_loss.item()})
                    accelerator.log({"epoch": epoch, "learning_rate": optimizer.param_groups[0]['lr']})


                    running_loss = 0
                    running_distillation_loss = 0
                    running_cross_entropy_loss = 0
                    progress_bar.update()

                # evaluate the student model every 4 log intervals
                if batch_idx % log_interval == 0:
                    # evaluate the student model
                    evaluate(student_model, eval_dataloader=eval_dataloader, is_student=True, pad_token_id=pad_token_id)
                    student_model.train()
                
                lr_scheduler.step()
                
                progress_bar.set_description(f"{accelerator.process_index} Training Epoch: {epoch+1}/{epochs} | Step: {batch_idx}/{steps_per_epoch} | Average Training Loss: {loss.mean().item():.5f}")
            
            progress_bar.close()
            print(f"Epoch: {epoch} completed")
        
        print(f"process index: {accelerator.process_index}", mem_trace)
        
        if os.path.exists("./checkpoints") == False:
            os.mkdir("./checkpoints")
        if accelerator.is_main_process:
            torch.save(student_model.state_dict(), f"./checkpoints/student_chkpt_epoch_{epoch}_type_{'mamba' if isinstance(student_model, MambaLMHeadModel) else 'transformer'}_max_length_{max_length}.pt")
    
    if accelerator.is_main_process:
        # evaluate the teacher model
        accelerator.wait_for_everyone()
        evaluate(teacher_model, eval_dataloader=eval_dataloader, is_student=False, pad_token_id=pad_token_id)
        evaluate(student_model, eval_dataloader=eval_dataloader, is_student=True, pad_token_id=pad_token_id)

    accelerator.end_training()


        


# Training Loop
def train(limit: int = 1000, batch_size: int = 4, max_length: int = 128, epochs: int = 5,
           learning_rate: float = 5e-5, load_chkpt: bool=False, load_hf_model: bool=False, model_path: str=None, is_mamba: bool=False, accumulation_steps: int = 1):   
    # assert that if either load_chkpt or load_hf_model is True but not both
    assert not (load_chkpt and load_hf_model), "Both load_chkpt and load_hf_model cannot be True at the same time"
    
    teacher_model = get_teacher_model(teacher_model_path)
    
    
    teacher_model.eval()

    if load_hf_model:
        if not is_mamba:
            student_model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            student_model = get_mamba_model(path=model_path)
    else:
        if not is_mamba:
            student_model = get_sanity_student_model()
        else:
            student_model = get_mamba_model()
    
    student_model.train()

    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    
    distill_knowledge(teacher_model, student_model, optimizer, batch_size, max_length, limit=limit, epochs=epochs,
                       load_chkpt=load_chkpt, model_path=model_path, accumulation_steps=accumulation_steps)
    # save the student model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(student_model)
    unwrapped_model.save_pretrained(f"distr_full_trained_epoch_{epochs}_lr_{learning_rate}_is_mamba_{is_mamba}_max_length_{max_length}")


# Evaluate the student model
def evaluate(model_or_path: Union[str, AutoModelForCausalLM, MambaLMHeadModel, MambaForCausalLM], eval_dataloader: DataLoader = None, pad_token_id: int = None, is_student: bool = True):
    accelerator.wait_for_everyone()
    # evaluate the student model using the test dataset
    if isinstance(model_or_path, str):
        student_model = AutoModelForCausalLM.from_pretrained(model_or_path)
    else:
        student_model = model_or_path
    
    student_model.eval()
    if eval_dataloader is None:
        dataloader, pad_token_id = init_dataloader(8, 128, "test")
    else:
        dataloader, pad_token_id = eval_dataloader, pad_token_id
    
    # evalua using the test dataset
    running_loss = 0
    counter = 0
    
    start = time.perf_counter()
    for batch in tqdm(dataloader):
        counter += 1
        batched_input_ids = batch['input_ids']
        inputs = batched_input_ids[:, :-1].contiguous()
        labels = batched_input_ids[:, 1:].contiguous()
        labels[labels == pad_token_id] = HF_PADDING_IGNORE

        batched_attention_mask = batch['attention_mask']
        attention_mask = batched_attention_mask[:, :-1].contiguous()
        with torch.no_grad():

            if isinstance(student_model, MambaLMHeadModel):
                student_outputs = student_model(input_ids=inputs,
                                            ).logits
            else:
                student_outputs = student_model(input_ids=inputs,
                                                attention_mask=attention_mask,
                                                labels=labels
                                                ).logits

        student_label_loss = nn.CrossEntropyLoss(ignore_index=HF_PADDING_IGNORE)(student_outputs.view(-1, student_outputs.size(-1)), labels.view(-1))
        running_loss += student_label_loss.item()
    duration = time.perf_counter() - start
    prefix = "student_" if is_student else "teacher_"
    accelerator.log({f"{prefix}test_loss": running_loss / counter})
    perplexity = np.exp(running_loss / counter)
    accelerator.log({f"{prefix}test_perplexity": perplexity})
    accelerator.log({f"{prefix}test_duration": duration})
    prefix = "Student" if is_student else "Teacher"
    print(f"{prefix} Test Loss: {(running_loss / counter):.5f} | Test Perplexity: {perplexity:.5f} | Duration: {duration:.5f} seconds")
    
    

    
    

        

# command line run for training with parsing arguments
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--load_chkpt", action="store_true")
    parser.add_argument("--load_hf_model", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--is_mamba", action="store_true", default=False)
    parser.add_argument("--accumulation_steps", type=int, default=1)

    args = parser.parse_args()
    log_config_dict = {
                "limit": str(args.limit),
                "batch_size": str(args.batch_size),
                "max_length": str(args.max_length),
                "epochs": str(args.epochs),
                "learning_rate": str(args.learning_rate),
                "model_path": str(args.model_path),
                "is_mamba": str(args.is_mamba),
                "accumulation_steps": str(args.accumulation_steps),
                
        }
    accelerator.init_trackers(
        project_name="ACC-MAMBA-KD-ULD",
        config=log_config_dict,
        init_kwargs={"wandb": {"name": f"distillation-uld-kd-mamba-{args.epochs}-epochs-{args.max_length}-max-length-{args.batch_size}-batch-size-{args.learning_rate}-lr-{args.is_mamba}-is-mamba-{args.accumulation_steps}-accumulation-steps-{teacher_model_path}-teacher-model-{args.model_path}-student-model"}}
    )

    train(limit=args.limit, batch_size=args.batch_size, max_length=args.max_length, epochs=args.epochs,
          learning_rate=args.learning_rate, load_chkpt=args.load_chkpt, load_hf_model=args.load_hf_model,
          model_path=args.model_path, is_mamba=args.is_mamba, accumulation_steps=args.accumulation_steps)

    # example command line run:
    # python evals/distillation.py --limit 1000000000000 --batch_size 16 --max_length 256 --epochs 5 --learning_rate 1e-3 --is_mamba 
    # python evals/distillation.py --limit 1000000000000 --batch_size 16 --max_length 256 --epochs 5 --learning_rate 1e-3 --load_chkpt --model_path ./checkpoints/student_chkpt_epoch_0_type_mamba_max_length_256.pt --is_mamba 
    # python evals/distillation.py --limit 1000000000000 --batch_size 8 --max_length 128 --epochs 3 --learning_rate 1e-3 --load_hf_model --model_path meta-llama/Meta-Llama-3-8B 
    # python evals/distillation.py --limit 1000000000000 --batch_size 8 --max_length 128 --epochs 3 --learning_rate 1e-3 --load_hf_model --model_path /cs/labs/roys/w552295/bamba/full_trained_epoch_2_lr_0.001_is_mamba_True_max_length_128  --is_mamba
    # python evals/distillation.py --limit 1000000000000 --batch_size 2 --max_length 128 --epochs 3 --learning_rate 1e-3 --load_hf_model --model_path state-spaces/mamba-790m-hf --accumulation_steps 16 

