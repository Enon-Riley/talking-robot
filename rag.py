# モデルのキャッシュパスの変更
import os
os.environ['HF_HOME'] = '/data/cache/huggingface'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# モデルのキャッシュパスの確認
from transformers import file_utils
print(file_utils.default_cache_path)

modelname = 'tokyotech-llm/Swallow-13b-instruct-hf'
#modelname = 'tokyotech-llm/Swallow-70b-instruct-hf'
#modelname = 'elyza/ELYZA-japanese-Llama-2-13b-instruct'

tokenizer = AutoTokenizer.from_pretrained(modelname)

model = AutoModelForCausalLM.from_pretrained(
    modelname,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

prompt = '''
以下に、あるタスクを説明する指示があり、それに付随する入力が更なる文脈を提供しています。リクエストを適切に完了するための回答を記述してください。

### 指示:
以下のトピックに関して可能な限り詳細な情報を提供してください。

### 入力:
松江高専とは何ですか？

### 応答:
'''

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

with torch.no_grad():
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    output_ids = model.generate(
        token_ids.to(model.device),
        temperature=0.99,
        top_p=0.95,
        do_sample=True,
        max_new_tokens=256,
    )
    for i in output_ids:
      print('--------')
      output = tokenizer.decode(i,skip_special_tokens=True)
      print(output)

