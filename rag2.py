import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("cyberagent/Llama-3.1-70B-Japanese-Instruct-2407", device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("cyberagent/Llama-3.1-70B-Japanese-Instruct-2407")

