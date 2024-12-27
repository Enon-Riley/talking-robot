#https://gist.github.com/niw/b8e8509147e27b943b4d9af01bea91cf

import os
os.environ['HF_HOME'] = '/data/cache/huggingface'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "cyberagent/open-calm-7b"
peft_model_path = "output-lora"

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TextStreamer

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config, #load_in_8bit=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

#for item in model.state_dict().items():
#    print(item[0])

#prompt_input = '以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{instruction}\n\n### 入力:{input}\n\n### 応答:\n{output}'

prompt_no_input = '以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{instruction}\n\n### 応答:\n{output}'

instruction = "松江高専直野寮について教えてください。"

query = prompt_no_input.format(instruction=instruction, output='')

#inputs = tokenizer(query, return_tensors='pt').to(model.device)
#inputs = tokenizer(question, return_tensors="pt").to(model.device)

#学習前の応答
#input_ids = tokenizer.encode(
#    query,  # 質問
#    add_special_tokens=False,
#    return_tensors="pt"
#)
#
#tokens = model.generate(
#    input_ids.to(device=model.device),
#    max_new_tokens=1024, # 回答の長さ
#    temperature=0.99,
#    top_p=0.95,
#    do_sample=True,
#    top_k=40,
#    repetition_penalty=5.0,
#    pad_token_id=tokenizer.pad_token_id,
#)
#output = tokenizer.decode(tokens[0], skip_special_tokens=True)
#print(output)


#with torch.no_grad():
#    tokens = model.generate(
#        **inputs,
#        do_sample=True,
#        #max_new_tokens=128,
#        #temperature=0.7,
#        #top_p=0.75,
#        max_new_tokens=256, #128,
#        temperature=0.99,
#        top_p=0.95,
#        top_k=40,
#        repetition_penalty=5.0,
#        pad_token_id=tokenizer.pad_token_id,
#    )
#
#print(tokenizer.decode(tokens[0], skip_special_tokens=True))

#モデルの調整

for param in model.parameters():
    param.requires_grad = False # freeze the model - train adapters later
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable() # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)
model.embed_out = CastOutputToFloat(model.embed_out)

from peft import LoraConfig, get_peft_model, PeftType, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    # OpenCALM-7B は GPT-NeoX なので、ターゲットモジュールは `query_key_value`
    target_modules=["query_key_value"],
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

from datasets import load_dataset

def convert(item):
    instruction = item['instruction']
    output = item['output']
    return tokenizer(prompt_no_input.format(instruction=instruction,output=output))

#data = load_dataset("kunishou/databricks-dolly-15k-ja")
data = load_dataset('json', data_files='naono.json')
data = data \
    .map(convert, remove_columns=["input", "instruction", "output", "category", "index"]) \
    .filter(lambda item: len(item["input_ids"]) <= 2048)

#data = data.map(lambda samples: tokenizer(samples["output"]), batched=True)
#print(data["train"][0])
#exit(0)

trainer = Trainer(
    model=model, 
    train_dataset=data['train'],
    args=TrainingArguments(
        per_device_train_batch_size=8, 
        gradient_accumulation_steps=4,
        warmup_steps=50,
        max_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir='outputs'
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# 保存

model.save_pretrained(peft_model_path)



   #matsue_generate.pyへ

#  # 読み込み
#  
#  from peft import PeftModel, PeftConfig
#  #from transformers import AutoModelForCausalLM, AutoTokenizer
#  
#  config = PeftConfig.from_pretrained(peft_model_path)
#  model = AutoModelForCausalLM.from_pretrained(
#      config.base_model_name_or_path,
#      quantization_config=quantization_config, #load_in_8bit=True,
#      device_map='auto'
#  )
#  
#  tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
#  
#  # ここでオリジナルのモデルに学習結果のLoRAを追加
#  model = PeftModel.from_pretrained(model, peft_model_path)
#  
#  # 学習後の応答
#  
#  #instruction = question
#  #inputs = tokenizer(prompt_no_input.format(instruction=question, output=""), return_tensors='pt').to(model.device)
#  #
#  #with torch.no_grad():
#  #    tokens = model.generate(
#  #        **inputs,
#  #        do_sample=True,
#  #        top_k=40,
#  #        #max_new_tokens=128,
#  #        #temperature=0.7,
#  #        #top_p=0.75,
#  #        max_new_tokens=256, #128,
#  #        temperature=0.99,
#  #        top_p=0.95,
#  #        repetition_penalty=5.0,
#  #        pad_token_id=tokenizer.pad_token_id,
#  #    )
#  #
#  #print(tokenizer.decode(tokens[0], skip_special_tokens=True))
#  
#     #matsue_generate.pyへ
#  input_ids = tokenizer.encode(
#      query,  # 質問
#      add_special_tokens=False,
#      return_tensors="pt"
#  )
#  
#  tokens = model.generate(
#      input_ids.to(device=model.device),
#      max_new_tokens=1024, # 回答の長さ
#      temperature=0.99,
#      top_p=0.95,
#      do_sample=True,
#      top_k=40,
#      repetition_penalty=5.0,
#      pad_token_id=tokenizer.pad_token_id,
#  )
#  output = tokenizer.decode(tokens[0], skip_special_tokens=True)
#  print(output)
