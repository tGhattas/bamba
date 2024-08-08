import os
import random
from typing import Optional, Union
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling, MambaForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from legacy_DK import distill_knowledge
try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:
    class MambaLMHeadModel: pass
import argparse

from pprint import pprint

import time
from hf_trainer import KDTrainer
import wandb
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel
from accelerate import PartialState
import numpy as np
import lm_eval
from lm_eval.loggers import WandbLogger


logger = None
accelerator = None
log_config_dict = None
hf_mamba_path = "state-spaces/mamba-790m-hf"
teacher_model_path = "meta-llama/Meta-Llama-3-8B"
tiny_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
pythia_14m_model_path = "EleutherAI/pythia-14m"
pythia_1B_model_path = "EleutherAI/pythia-1b"
pythia_28B_model_path = "EleutherAI/pythia-2.8b"
pythia_69B_model_path = "EleutherAI/pythia-6.9b"
teacher_model_path = pythia_28B_model_path 

# teacher_model_path = "mistralai/Mistral-7B-v0.3"

def get_teacher_model(path: str, peft_config_path: Optional[str] = None, peft: bool = False):
    if peft:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map={'':PartialState().process_index})
        if peft_config_path:
            model = PeftModel.from_pretrained(model, peft_config_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(path)
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
    
    return model


# MAMBA student model
def get_mamba_model(path: str = None, gpu: int = None, set_teacher_embedding_size: bool = False, load_model_weights: bool = False):
    device = f'cuda{f":{gpu}" if gpu else ""}' if torch.cuda.is_available() else 'mps'
    teacher_model = get_teacher_model(teacher_model_path)
    param = next(teacher_model.parameters())
    if path and load_model_weights:
        mamba_student_model = smart_to(MambaForCausalLM.from_pretrained(path), device)
        config = mamba_student_model.config
        if set_teacher_embedding_size:
            config.vocab_size = teacher_model.config.vocab_size
    else:
        print("---------------------------------Loading Mamba model from scratch---------------------------------")
        config = AutoConfig.from_pretrained(path if path is not None else hf_mamba_path)
        config.vocab_size = teacher_model.config.vocab_size
        mamba_student_model = smart_to(MambaForCausalLM(config), device)
    return mamba_student_model


HF_PADDING_IGNORE = -100


def get_dataset(batch_size: int, max_length: int, partition: str = "train", minimize_dataset: bool = False, return_dataloader: bool = True, dataset_path: str = None):

    dataset_path = "wikitext-2-v1" if dataset_path is None else dataset_path
    if 'wikitext' in dataset_path:
        dataset = load_dataset("wikitext", dataset_path, streaming=False, split=partition)
    else:
        dataset = load_dataset(dataset_path, streaming=False, split=partition)
    # Load the teacher tokenizer
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, use_fast=True)
    # add padding token to the tokenizer
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    # Tokenize the dataset
    def teacher_tokenize_function(examples):
        return teacher_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    num_of_gpus = max(torch.cuda.device_count(), 1)
    
    if os.path.exists(tokenized_dataset_path := f"./tokenized_datsets/{dataset_path}-{partition}-tokenized-mini-{minimize_dataset}"):
        teacher_tokenized_datasets = load_from_disk(tokenized_dataset_path)
        print("---------------------------Tokenized dataset loaded from disk---------------------------")
    else:   
        teacher_tokenized_datasets = dataset.map(teacher_tokenize_function, batched=True, num_proc=10,
                                                remove_columns=["text"], batch_size=batch_size * num_of_gpus)
        teacher_tokenized_datasets.save_to_disk(tokenized_dataset_path)
    teacher_data_collator = DataCollatorForLanguageModeling(
        tokenizer=teacher_tokenizer,
        mlm=False,  # Set to True if using Masked Language Modeling
        pad_to_multiple_of=8  # Optional, can pad to the nearest multiple of 8 for efficiency
    )
    tokenized_data = teacher_tokenized_datasets if not minimize_dataset else teacher_tokenized_datasets.take(10)
    # Create the data loader
    teacher_data = DataLoader(tokenized_data, batch_size=batch_size, collate_fn=teacher_data_collator) if return_dataloader else tokenized_data

    return (teacher_data, teacher_tokenizer.pad_token_id, teacher_data_collator)



def print_model_parameters(model_name: str, model: Union[AutoModelForCausalLM, MambaLMHeadModel], accelerator: Optional[Accelerator] = None):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    printF = pprint if accelerator is None else accelerator.print
    printF(f"Model: {model_name}")
    printF(f"Total Parameters: {total_params}")
    printF(f"Trainable Parameters: {trainable_params}")
    printF(f"Total Memory Footprint: {total_params * 4 / 1024 / 1024} MB")


def logits_to_tokens(logits):
    """Convert logits to token ids."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def finetune_teacher(unique_id: str, batch_size: int, max_length: int, minimize_dataset:bool, epochs:int, lr: float, optimizer: str, mixed_precision: bool, tf32: bool, peft: bool, accumulation_steps: int,
                    teacher_model_path: str = teacher_model_path, wandb_name: str = "", dataset_path: str = None, evaluate_only: bool = False, load_hf_model: bool = False):
    # fine tune teacher model using hf trainer
    train_dataset, _, teacher_data_collator = get_dataset(batch_size, max_length, "train", minimize_dataset=minimize_dataset, return_dataloader=False, dataset_path=dataset_path)
    test_dataset, _, _ = get_dataset(batch_size, max_length, "validation", minimize_dataset=minimize_dataset, return_dataloader=False, dataset_path=dataset_path)
    if peft:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=64,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        assert load_hf_model, "PEFT requires a pre-trained model"
        model = AutoModelForCausalLM.from_pretrained(teacher_model_path, quantization_config=bnb_config,
                                                    torch_dtype=torch.bfloat16, device_map={'':PartialState().process_index})
    else:
        if load_hf_model:
            model = AutoModelForCausalLM.from_pretrained(teacher_model_path)
        else:
            print("---------------------------------Loading Teacher model from scratch---------------------------------")
            config = AutoConfig.from_pretrained(teacher_model_path) 
            model = AutoModelForCausalLM.from_config(config)

        

    fix_mamba_config(model)
    name = f"u{unique_id}_finetuned_{wandb_name}_{epochs}_ep_{teacher_model_path}_optm{optimizer}_mp{mixed_precision}_{dataset_path}".replace('.','').replace('/','')
    training_args = SFTConfig(
        output_dir=f"./ft-{unique_id}-results",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        eval_steps=100 if not minimize_dataset else 10,
        save_steps=3000,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=lr,
        report_to="wandb",  # Enable logging to wandb
        gradient_accumulation_steps=accumulation_steps,
        remove_unused_columns=False,
        fp16=mixed_precision,
        tf32=tf32,
        optim=optimizer,
        gradient_checkpointing=not peft,
        lr_scheduler_type="cosine",
        run_name=name,
        load_best_model_at_end=True,
        max_seq_length=max_length,
        eval_on_start=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=teacher_data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config if peft else None,
    )

    if not evaluate_only:
        # Train the model
        trainer.train()
        trainer.save_model(name)
    # Evaluate the model
    eval_results = trainer.evaluate()
    # add perplexity
    eval_results["eval_perplexity"] = np.exp(eval_results["eval_loss"])
    print("Evaluation results:", eval_results)

    return name
    



def hf_train(unique_id: str, teacher_model: AutoModelForCausalLM, student_model: Union[MambaLMHeadModel, AutoModelForCausalLM],
            minimize_dataset: bool, batch_size: int, max_length: int, epochs: int, model_path: str, accumulation_steps: int,
            alpha: float, temperature: float, learning_rate: float, mixed_precision: bool, optimizer: str, tf32: bool, teacher_model_path: str = teacher_model_path,
            wandb_name: str = "", dataset_path: str = None, scaling_factor:float = None):
    # student_model.pad_token = student_tokenizer.eos_token
    student_model.resize_token_embeddings(teacher_model.config.vocab_size)
    train_dataset, _, teacher_data_collator = get_dataset(batch_size, max_length, "train", minimize_dataset=minimize_dataset, return_dataloader=False, dataset_path=dataset_path)
    test_dataset, _, _ = get_dataset(batch_size, max_length, "validation", minimize_dataset=minimize_dataset, return_dataloader=False, dataset_path=dataset_path)
    name = f"u{unique_id}_hf_train_{wandb_name}_{epochs}_epochs_{model_path}_optim{optimizer}_mp{mixed_precision}_{dataset_path}".replace('.','').replace('/','')
    # student_model.config.use_cache = False
    training_args = SFTConfig(
        output_dir=f"./hf-{unique_id}-results",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        eval_steps=100 if not minimize_dataset else 10,
        save_steps=3000,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=learning_rate,
        report_to="wandb",  # Enable logging to wandb
        gradient_accumulation_steps=accumulation_steps,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        optim=optimizer,
        gradient_checkpointing=True,
        fp16=mixed_precision,
        tf32=tf32,
        run_name=name,
        load_best_model_at_end=True,
        eval_on_start=True,
    )
    trainer = KDTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        temperature=temperature,
        alfa=alpha,
        args=training_args,
        data_collator=teacher_data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        scaling_factor=scaling_factor,
    )
    global accelerator
    accelerator = trainer.accelerator
    print_model_parameters(teacher_model_path, teacher_model, accelerator = trainer.accelerator)
    print_model_parameters(model_path, student_model, accelerator = trainer.accelerator)    
    # Train the model
    trainer.train()
    # save the model
    trainer.save_model(name)
    # Evaluate the model
    post_eval_results = trainer.evaluate()
    # add perplexity
    post_eval_results["eval_perplexity"] = np.exp(post_eval_results["eval_loss"])
    printF = pprint if accelerator is None else accelerator.print
    printF("Post-training evaluation results:", post_eval_results)
    eval_lm_harness(name)


def fix_mamba_config(model):
    model.config.keys_to_ignore_at_inference = getattr(model.config, "keys_to_ignore_at_inference", [])
    model.config.keys_to_ignore_at_inference.append("cache_params")


# Training Loop
def train(limit: int = 1000, batch_size: int = 4, max_length: int = 128, epochs: int = 5,
        learning_rate: float = 5e-5, load_chkpt: bool=False, load_hf_model: bool=False, model_path: str=None,
        is_mamba: bool=False, gpu: int = None, accumulation_steps: int = 1, use_modified_tokenizer: bool = False,
        use_teacher_tokenizer: bool = False, teacher_model_path: str = teacher_model_path,
        minimize_dataset: bool = False, unique_id: str = '', alpha: float = 0.5, temperature: float = 2.0, hf_trainer: bool = False,
        optimizer=None, mixed_precision: bool = False, tf32: bool = False, peft: bool = False, peft_config_path: str = None, wandb_name: str = "",
        dataset_path: str = None, scaling_factor: float = None):   
    # assert that if either load_chkpt or load_hf_model is True but not both
    assert not (load_chkpt and load_hf_model), "Both load_chkpt and load_hf_model cannot be True at the same time"
    device = f'cuda{f":{gpu}" if gpu else ""}' if torch.cuda.is_available() else 'mps'
    teacher_model = get_teacher_model(teacher_model_path, peft_config_path, peft=peft)
    smart_to(teacher_model, device)
    
    teacher_model.eval()
    if load_hf_model:
        if not is_mamba:
            student_model = smart_to(AutoModelForCausalLM.from_pretrained(model_path), device)
        else:
            student_model = get_mamba_model(path=model_path, gpu=gpu, load_model_weights=load_hf_model)
    else:
        if not is_mamba:
            student_model = smart_to(get_sanity_student_model(), device)
        else:
            student_model = get_mamba_model(path=model_path, gpu=gpu, load_model_weights=load_hf_model)
    student_model.train()

    
    if hf_trainer:
        fix_mamba_config(student_model)
        save_path = hf_train(unique_id, teacher_model, student_model, minimize_dataset, batch_size, max_length, epochs, model_path,
                accumulation_steps, alpha, temperature, learning_rate, mixed_precision, optimizer, tf32, teacher_model_path,
                wandb_name, dataset_path, scaling_factor=scaling_factor)
        eval_lm_harness(save_path)
    else:
        optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
        distill_knowledge(teacher_model, student_model, optimizer, batch_size, max_length, limit=limit, epochs=epochs,
                        load_chkpt=load_chkpt, model_path=model_path, gpu=gpu, accumulation_steps=accumulation_steps,
                        modified_tokenizer=use_modified_tokenizer, use_teacher_tokenizer=use_teacher_tokenizer, teacher_model_path=teacher_model_path,
                        minimize_dataset=minimize_dataset, unique_id=unique_id, alpha=alpha, temperature=temperature, accelerator=accelerator, logger=logger)


def smart_to(model, device="cuda" if torch.cuda.is_available() else "mps"):
    if accelerator is None:
        return model.to(device)
    return model


def init_accelerate(hf: bool = True):
    global accelerator
    accelerator = Accelerator() if not hf else True
    if hf:
        print("Accelerate set to True")
    else:
        print(f"Using device: {accelerator.device}")


def init_logger(logger_):
    global logger
    logger = logger_


def eval_lm_harness(name: str):
    # eval using eval lm harness
    # accelerate launch --main_process_port 29531 -m lm_eval --model hf --model_args tokenizer=EleutherAI/pythia-1b,pretrained=/cs/labs/roys/w552295/bamba/u7084_WED-scale10-tmp6-alf9-pythia69Teacher__hf_train_WED-scale10-tmp6-alf9-pythia69Teacher_1_epochs_state-spacesmamba-370m-hf_optimadamw_bnb_8bit_mpFalse_JeanKaddourminipile --tasks lambada_openai,hellaswag,arc_challenge,piqa,arc_easy --batch_size 10 --output_path outputs/u7084
    if accelerator.is_local_main_process:
        path_prefix = "/cs/labs/roys/w552295/bamba/" if torch.cuda.is_available() else "./"
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"tokenizer=EleutherAI/pythia-1b,pretrained={path_prefix}{name}",
            tasks="lambada_openai,hellaswag,arc_challenge,piqa,arc_easy",
            log_samples=True,
        )

        wandb_logger = WandbLogger()
        wandb_logger.post_init(results)
        wandb_logger.log_eval_result()
        wandb_logger.log_eval_samples(results["samples"])  # if log_samples


# command line run for training with parsing arguments
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10000000000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--load_chkpt", action="store_true")
    parser.add_argument("--load_hf_model", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--is_mamba", action="store_true", default=True)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--use_modified_tokenizer", action="store_true", default=False)
    parser.add_argument("--use_teacher_tokenizer", action="store_true", default=False)
    parser.add_argument("--teacher_model_path", type=str, default=teacher_model_path)
    parser.add_argument("--wandb_name", type=str, default='')
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--use_accelerate", action="store_true", default=False)
    parser.add_argument("--minimize_dataset", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--finetune_teacher", action="store_true", default=False)
    parser.add_argument("--hf_trainer", action="store_true", default=True)
    parser.add_argument("--optimizer", type=str, default="adamw_hf")
    parser.add_argument("--mixed_precision", action="store_true", default=False)
    parser.add_argument("--tf32", action="store_true", default=False)
    parser.add_argument("--peft", action="store_true", default=False)
    parser.add_argument("--peft_config_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--scaling_factor", type=float, default=None)
    args = parser.parse_args()
    

    unique_run_id = str(random.randint(0, 9999))
    name_prefix = f"{unique_run_id}_{args.wandb_name}_"
    log_config_dict = {
                "limit": str(args.limit),
                "batch_size": str(args.batch_size),
                "max_length": str(args.max_length),
                "epochs": str(args.epochs),
                "learning_rate": str(args.learning_rate),
                "model_path": str(args.model_path),
                "is_mamba": str(args.is_mamba),
                "accumulation_steps": str(args.accumulation_steps),
                "mixed_precision": str(args.mixed_precision),
                "tf32": str(args.tf32),
        }
    
    wandb_name = f"mp{args.mixed_precision}-{args.epochs}-epochs-{args.max_length}-maxLen-alfa{args.alpha}-tmp{args.temperature}-{args.batch_size}-batchsize-{args.learning_rate}-lr-{args.is_mamba}-isMamba-{args.accumulation_steps}-accum-steps-{teacher_model_path}-teacher-model-{args.model_path}-student-model".replace('.','').replace('/','')
    if not torch.cuda.is_available():
        os.environ["WANDB_PROJECT"] = "LOCAL_RUN"
    else:
        os.environ["WANDB_PROJECT"] = "hf-MAMBA"
        os.environ["WANDB_NAME"] = wandb_name
    if args.use_accelerate and not args.hf_trainer:
        init_accelerate(args.hf_trainer)
        if args.resume:
            os.environ["WANDB_RESUME"] = "must"
            assert args.wandb_id is not None, "Please provide a wandb_id to resume"
            os.environ["WANDB_RUN_ID"] = args.wandb_id
        
        if not args.hf_trainer:
            accelerator.print("-----Accelerate Initialized-----")
            accelerator.init_trackers(
                project_name="HF-ACC",
                config=log_config_dict,
                init_kwargs={"wandb": {"name": wandb_name}},
            )
            init_logger(accelerator)
    elif args.use_accelerate:
            init_accelerate(args.hf_trainer)
    else:
        wandb.init(
            project="MAMABA",
            config=log_config_dict,
            name=None if args.resume else wandb_name,
            resume=args.resume,
            id=args.wandb_id
        )
        init_logger(wandb)

    if args.finetune_teacher:
        finetune_teacher(unique_id=name_prefix, batch_size=args.batch_size, max_length=args.max_length, minimize_dataset=args.minimize_dataset, epochs=args.epochs, lr=args.learning_rate, optimizer=args.optimizer, teacher_model_path=args.teacher_model_path, mixed_precision=args.mixed_precision, tf32=args.tf32, peft=args.peft, accumulation_steps=args.accumulation_steps, wandb_name=args.wandb_name, dataset_path=args.dataset_path, load_hf_model=args.load_hf_model)
    else:
        train(limit=args.limit, batch_size=args.batch_size, max_length=args.max_length, epochs=args.epochs,
            learning_rate=args.learning_rate, load_chkpt=args.load_chkpt, load_hf_model=args.load_hf_model,
            model_path=args.model_path, is_mamba=args.is_mamba, gpu=args.gpu, accumulation_steps=args.accumulation_steps,
            use_modified_tokenizer=args.use_modified_tokenizer, use_teacher_tokenizer=args.use_teacher_tokenizer,
            teacher_model_path=args.teacher_model_path, minimize_dataset=args.minimize_dataset, unique_id=name_prefix, alpha=args.alpha,
            temperature=args.temperature, hf_trainer=args.hf_trainer, optimizer=args.optimizer, mixed_precision=args.mixed_precision,
            tf32=args.tf32, peft_config_path=args.peft_config_path, peft=args.peft, wandb_name=args.wandb_name, dataset_path=args.dataset_path,
            scaling_factor=args.scaling_factor)

    # example command line run:
    # python evals/distillation.py --limit 1000000000000 --batch_size 8 --max_length 128 --epochs 3 --learning_rate 1e-3 --load_hf_model --model_path /cs/labs/roys/w552295/bamba/full_trained_epoch_2_lr_0.001_is_mamba_True_max_length_128  --is_mamba
    # python evals/distillation.py --limit 100 --batch_size 2 --max_length 128 --epochs 3 --learning_rate 1e-3 --load_hf_model --model_path state-spaces/mamba-370m-hf --accumulation_steps 16 --is_mamba




