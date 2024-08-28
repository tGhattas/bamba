# from datasets import load_dataset
# from transformers import AutoTokenizer

# # Load your dataset
# dataset = load_dataset('JeanKaddour/minipile', split='train')

# # Load the EleutherAI/pythia-1.4b tokenizer
# tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-1.4b')

# # Tokenize the entire dataset
# def tokenize_function(example):
#     return tokenizer(example['text'], return_special_tokens_mask=True)

# # Apply the tokenizer to the dataset
# tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# # Count the number of tokens
# num_tokens = sum(len(tokens) for tokens in tokenized_dataset['input_ids'])

# print(f"Total number of tokens in the dataset: {num_tokens}")



from transformers import MambaForCausalLM, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model

model = AutoModelForCausalLM.from_pretrained('/cs/labs/roys/w552295/bamba/mamba-790m-hf-resized-50432')
# model.resize_token_embeddings(50432)
# model.save_pretrained('mamba-790m-hf-resized-50432')                          
# 
pm = PeftModel.from_pretrained(model, '/cs/labs/roys/w552295/bamba/u7022_lora-tmp2-alf9-mamba790-minipile__hf_train_lora-tmp2-alf9-mamba790-minipile_1_epochs_state-spacesmamba-790m-hf_optimadamw_hf_mpFalse_JeanKaddourminipile-PEFT')
pm = pm.merge_and_unload()
print(pm)
pm.save_pretrained('mamba-790m-hf-LORA-merged-r16')
