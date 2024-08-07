from itertools import islice
from typing import Union

from numpy import isnan
from memory import MemoryTrace
from kl_div_loss import KLDivLoss
from uld_loss import ULDLoss
from modified_tokenizer import ModifiedMambaTokenizerFactory
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, MambaForCausalLM
import torch
from torch import nn
from pprint import pprint



def distill_knowledge(teacher_model: AutoModelForCausalLM, student_model: AutoModelForCausalLM, optimizer: torch.optim.Optimizer,
                    batch_size: int, max_length: int,
                    limit: int=1000, epochs: int=5, load_chkpt: bool=False, model_path: str=None, gpu: int = None, accumulation_steps: int = 1,
                    modified_tokenizer: bool = False, use_teacher_tokenizer: bool = False, teacher_model_path: str = None, minimize_dataset: bool = False, unique_id: str = '', alpha: float = 0.5, temperature: float = 2.0, accelerator=None, logger=None):
    device = f'cuda{f":{gpu}" if gpu else ""}' if torch.cuda.is_available() else 'mps'
    printF = pprint if accelerator is None else accelerator.print

    def smart_to(model, device="cuda" if torch.cuda.is_available() else "mps"):
        if accelerator is None:
            return model.to(device)
        return model

    if load_chkpt:
        student_model.load_state_dict(torch.load(model_path))

    first_batch = True
    log_interval = 10
    
    running_loss = 0
    running_distillation_loss = 0
    running_cross_entropy_loss = 0
    
    student_underlying_model = student_model
    teacher_underlying_model = teacher_model

    assert not (modified_tokenizer and use_teacher_tokenizer), "Both modified_tokenizer and use_teacher_tokenizer cannot be True at the same time"
    if modified_tokenizer:
        student_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, use_fast=True)
        tokenizer_factory = ModifiedMambaTokenizerFactory(student_tokenizer=student_tokenizer, teacher_tokenizer=teacher_tokenizer)
        student_tokenizer = tokenizer_factory.get_modified_tokenizer()
        student_underlying_model.resize_token_embeddings(len(student_tokenizer))
        dataloader = init_dataloader(batch_size, max_length, "train", student_tokenizer=student_tokenizer, minimize_dataset=minimize_dataset)
        printF(f"Student Model Vocab Size: {student_tokenizer.vocab_size}")
        printF(f"Teacher Model Vocab Size: {teacher_tokenizer.vocab_size}")
    elif use_teacher_tokenizer:
        student_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, use_fast=True)
        student_tokenizer.pad_token = student_tokenizer.eos_token
        student_underlying_model.resize_token_embeddings(teacher_underlying_model.config.vocab_size)
        dataloader = init_dataloader(batch_size, max_length, "train", student_tokenizer=student_tokenizer, minimize_dataset=minimize_dataset)
        printF("Using Teacher Tokenizer for student model")
    else:
        dataloader = init_dataloader(batch_size, max_length, "train", student_tokenizer=model_path, minimize_dataset=minimize_dataset)

    teacher_vocab_size = teacher_underlying_model.config.vocab_size
    student_vocab_size = student_underlying_model.config.vocab_size
    if teacher_vocab_size == student_vocab_size:
        loss_fn = KLDivLoss(reduction='mean', temperature=temperature, ignore_idx=HF_PADDING_IGNORE, distillation_loss_weight=alpha, using_acc=False)
        printF("-"*50)
        printF("Using KL Divergence Loss")
        printF("-"*50)
    else:
        loss_fn = ULDLoss(distillation_weight=alpha, crossentropy_weight=1-alpha, ignore_idx=HF_PADDING_IGNORE, teacher_temperature=temperature, student_temperature=temperature, skip_student_eos=True, skip_teacher_eos=True)
        printF("-"*50)
        printF("Using ULD Loss")
        printF("-"*50)
    

    if  model_path or modified_tokenizer or use_teacher_tokenizer:
        teacher_train_dataloader, student_train_dataloader, pad_token_id = dataloader
    else:
        teacher_train_dataloader, pad_token_id = dataloader

    eval_dataloader, _ = init_dataloader(batch_size, max_length, "test", minimize_dataset=minimize_dataset)
    lr_scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=int(0.05 * epochs * len(teacher_train_dataloader)), num_training_steps=epochs * len(teacher_train_dataloader))

    if accelerator is not None:
        teacher_train_dataloader, student_train_dataloader, eval_dataloader, student_model, teacher_model, optimizer, lr_scheduler = accelerator.prepare(teacher_train_dataloader, student_train_dataloader, eval_dataloader, student_model, teacher_model, optimizer, lr_scheduler)


    other_dataloader = teacher_train_dataloader if not model_path else student_train_dataloader
    steps_per_epoch = len(teacher_train_dataloader)

    if (accelerator is not None) or accelerator is None:
        printF("PRE TRAINING EVALS")
        # evaluate the teacher model
        evaluate(teacher_model, eval_dataloader=eval_dataloader, is_student=False, pad_token_id=pad_token_id)
        evaluate(student_model, eval_dataloader=eval_dataloader, is_student=True, pad_token_id=pad_token_id)

    for epoch in range(epochs):
        with MemoryTrace() as mem_trace:
            progress_bar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=steps_per_epoch//accumulation_steps,
                                dynamic_ncols=True, disable=(accelerator is not None and not accelerator.is_local_main_process))
            ignored_count = 0
            for batch_idx, (teacher_batch, student_batch)  in enumerate(islice(zip(teacher_train_dataloader, other_dataloader), limit)):
                teacher_batched_input_ids = smart_to(teacher_batch['input_ids'], device)
                student_batched_input_ids = smart_to(student_batch['input_ids'], device)
                
                teacher_inputs = smart_to(teacher_batched_input_ids[:, :-1].contiguous(), device)
                student_inputs = smart_to(student_batched_input_ids[:, :-1].contiguous(), device)

                student_labels = smart_to(student_batched_input_ids[:, 1:].contiguous(), device)
                student_labels[student_labels == pad_token_id] = HF_PADDING_IGNORE

                teacher_labels = smart_to(teacher_batched_input_ids[:, 1:].contiguous(), device)
                teacher_labels[teacher_labels == pad_token_id] = HF_PADDING_IGNORE

                batched_attention_mask = smart_to(teacher_batch['attention_mask'], device)
                attention_mask = smart_to(batched_attention_mask[:, :-1].contiguous(), device)
                
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids=teacher_inputs,
                                                    attention_mask=attention_mask
                                                    )

                student_outputs = student_model(input_ids=student_inputs,
                                                attention_mask=attention_mask,
                                                labels=student_labels)

                if first_batch:
                    printF = pprint if accelerator is None else accelerator.print
                    printF(f"Student logits shape: {student_outputs.logits.shape}")
                    printF(f"Teacher logits shape: {teacher_outputs.logits.shape}")
                    first_batch = False
                
                
                if (teacher_labels == pad_token_id).all() or (student_labels == pad_token_id).all():
                    ignored_count += 1
                    continue
                
                loss, student_label_loss, distillation_loss = loss_fn(student_outputs, teacher_outputs, student_labels, teacher_labels)
                student_label_loss = student_label_loss.mean() / accumulation_steps
                distillation_loss = distillation_loss.mean() / accumulation_steps
                loss = loss.mean() / accumulation_steps
                running_loss += loss.detach().float()
                running_distillation_loss += distillation_loss.detach().float()
                running_cross_entropy_loss += student_label_loss.detach().float()
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.zero_grad()
                    if accelerator is not None:
                        accelerator.backward(loss)
                    else:
                        loss.backward()
                    # Gradient clipping
                    (torch.nn.utils if accelerator is None else  accelerator).clip_grad_norm_(student_model.parameters(), max_norm=0.5)
                    optimizer.step()
                    logger.log({"running_loss": running_loss.item()})
                    logger.log({"running_distillation_loss": running_distillation_loss.item()})
                    logger.log({"running_cross_entropy_loss": running_cross_entropy_loss.item()})
                    logger.log({"learning_rate": optimizer.param_groups[0]['lr']})

                    running_loss = 0
                    running_distillation_loss = 0
                    running_cross_entropy_loss = 0
                    progress_bar.update()

                # evaluate the student model every 4 log intervals
                if batch_idx % log_interval == 0:
                    # evaluate the student model
                    evaluate(student_model, eval_dataloader=eval_dataloader, is_student=True, pad_token_id=pad_token_id, gpu=gpu)
                    student_model.train()
                
                lr_scheduler.step()
                # print till the 4th after the decimal point
                progress_bar.set_description(f"Training Epoch: {epoch+1}/{epochs} | Step: {batch_idx}/{steps_per_epoch} | Average Training Loss: {loss.mean().item():.5f}")
            
            progress_bar.close()
            print(f"Epoch: {epoch} completed")
            print(f"Ignored {ignored_count} batches")
        
        print(mem_trace) if accelerator is None else print(f"process index: {accelerator.process_index}", mem_trace)
        
        # if os.path.exists("./checkpoints") == False:
        #     os.mkdir("./checkpoints")
        # if (accelerator is not None and accelerator.is_main_process) or accelerator is None:
        #     torch.save(student_model.state_dict(), f"./checkpoints/u{unique_id}_student_chkpt_epoch_{epoch}_type_{'mamba' if isinstance(student_model, MambaLMHeadModel) else 'transformer'}_max_length_{max_length}.pt")
    
    
    if (accelerator is not None) or accelerator is None:
        printF("POST TRAINING EVALS")
        evaluate(teacher_model, eval_dataloader=eval_dataloader, is_student=False, pad_token_id=pad_token_id, gpu=gpu)
        evaluate(student_model, eval_dataloader=eval_dataloader, is_student=True, pad_token_id=pad_token_id, gpu=gpu)



# Evaluate the student model
def evaluate(model_or_path: Union[str, AutoModelForCausalLM, MambaForCausalLM], gpu: int = None, eval_dataloader= None, pad_token_id: int = None, is_student: bool = True, logger = None):
    device = f'cuda{f":{gpu}" if gpu else ""}' if torch.cuda.is_available() else 'mps'
    if accelerator is not None:
        accelerator.wait_for_everyone()
    # evaluate the student model using the test dataset
    if isinstance(model_or_path, str):
        student_model = smart_to(AutoModelForCausalLM.from_pretrained(model_or_path), device)
    else:
        student_model = model_or_path
    
    student_model.eval()
    if eval_dataloader is None:
        dataloader, pad_token_id = init_dataloader(8, 128, "test")
    else:
        dataloader, pad_token_id = eval_dataloader, pad_token_id
    
    # evalua using the test dataset
    running_loss = 0
    counter = 1
    ignored_count = 0
    # student_model_eval = student_model.module if isinstance(student_model, DataParallel) else student_model
    start = time.perf_counter()
    for batch in tqdm(dataloader):
        
        batched_input_ids = smart_to(batch['input_ids'], device)
        inputs = smart_to(batched_input_ids[:, :-1].contiguous(), device)
        labels = smart_to(batched_input_ids[:, 1:].contiguous(), device)
        # skip the batch if all the labels are pad tokens
        if (labels == pad_token_id).all():
            ignored_count += 1
            continue

        labels[labels == pad_token_id] = HF_PADDING_IGNORE

        batched_attention_mask = smart_to(batch['attention_mask'], device)
        attention_mask = smart_to(batched_attention_mask[:, :-1].contiguous(), device)
        with torch.no_grad():

            if isinstance(student_model, MambaLMHeadModel):
                student_outputs = smart_to(student_model(input_ids=inputs,
                                            ).logits, device)
            else:
                student_outputs = smart_to(student_model(input_ids=inputs,
                                                attention_mask=attention_mask,
                                                labels=labels
                                                ).logits, device)

        student_label_loss = nn.CrossEntropyLoss(ignore_index=HF_PADDING_IGNORE)(student_outputs.view(-1, student_outputs.size(-1)), labels.view(-1))

        running_loss += student_label_loss.item()
        if isnan(running_loss):
            print("NaN loss detected")
            print(f"student_outputs: {student_outputs}")
            print(f"labels.view(-1): {labels.view(-1)}")
            print(f"inputs: {inputs}")
            print(f"origin of labels: {batched_input_ids}")
            raise ValueError("NaN loss detected")
        counter += 1
            
    duration = time.perf_counter() - start
    prefix = "student_" if is_student else "teacher_"
    logger.log({f"{prefix}test_loss": running_loss / counter})
    perplexity = np.exp(running_loss / counter)
    logger.log({f"{prefix}test_perplexity": perplexity})
    logger.log({f"{prefix}test_duration": duration})
    prefix = "Student" if is_student else "Teacher"
    print(f"{prefix} Test Loss: {(running_loss / counter):.5f} | Test Perplexity: {perplexity:.5f} | Duration: {duration:.5f} seconds")
    print(f"Ignored {ignored_count} batches")
    