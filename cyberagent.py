
import os
os.environ['HF_HOME'] = '/data/cache/huggingface'
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model = AutoModelForCausalLM.from_pretrained("cyberagent/Llama-3.1-70B-Japanese-Instruct-2407", device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("cyberagent/Llama-3.1-70B-Japanese-Instruct-2407")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

messages = [
    #{"role": "user", "content": "AIによって私たちの暮らしはどのように変わりますか？"}
    #{"role": "user", "content": "松江高専直野寮は過ごしやすいですか？"}
    {"role": "user", "content": "松江高専直野寮は過ごしやすいですか？"}
]

input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
output_ids = model.generate(input_ids,
                            max_new_tokens=1024,
                            streamer=streamer)

