from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel, PeftConfig

peft_model_path = "output-lora"
quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16
        )

config = PeftConfig.from_pretrained(peft_model_path)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config=quantization_config, #load_in_8bit=True,
    device_map='auto'
    )

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

#ここでオリジナルのモデルに学習結果のLORAを 追加
model = PeftModel.from_pretrained(model,peft_model_path)

query = input("松江専郎：なんでもきいてください。\nあなた　：")
instruction = "以下に，あるタスクを説明する指示があり，それに付随する入力が更なる文脈を提供しています。リクエストを適切に完了するための回答を記述してください。\n\n指示:\n以下のトピックに関して可能な限り詳細な情報を提供してください。\n\n入力:\n{query}\n\n応答:\n"
prompt = instruction.format(query=query)


input_ids = tokenizer.encode(
    prompt,
    add_special_tokens=False,
    return_tensors="pt"
    )

tokens = model.generate(
    input_ids.to(device=model.device),
    max_new_tokens=1024,
    temperature=0.99,
    top_p=0.95,
    do_sample=True,
    top_k=40,
    repetition_penalty=5.0,
    pad_token_id=tokenizer.pad_token_id,
    )

output = tokenizer.decode(tokens[0],skip_special_tokens=True)
print(output)
