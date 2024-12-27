# モデルのキャッシュパスの変更
import os
os.environ['HF_HOME'] = '/data/cache/huggingface'

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1'
#model_name = 'rinna/llama-3-youko-8b'
#model_name = 'tokyotech-llm/Swallow-13b-instruct-hf'
#model_name = 'elyza/Llama-3-ELYZA-JP-8B'
#model_name = 'Qwen/Qwen2-7B-Instruct'
#model_name = 'cyberagent/open-calm-7b'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype='auto', #torch.bfloat16,
    #low_cpu_mem_usage=True,
    device_map='auto',
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

prompt_input = '以下に、あるタスクを説明する指示があり、それに付随する入力が更なる文脈を提供しています。リクエストを適切に完了するための回答を記述してください。\n\n### 入力:\n{input}\n\n### 指示:\n{instruction}\n\n### 応答:'

prompt_no_input = '以下に、あるタスクを説明する指示があります。リクエストを適切に完了するための回答を記述してください。\n\n### 指示:\n{instruction}\n\n### 応答:'

context = '''
松江高専では，自宅からの通学が困難な学生のために，定員448名の「直野寮（なおのりょう）」と呼ぶ学生寮が本校敷地内に併設されています． この直野寮は，学生の居室・事務室を収容する7つの建物からなり，それらの建物とは独立した寮生専用の食堂が併設されています．平成13年度には新たに女子寮が設置されました． 充実した設備による安全性が確保された状況と共に，女子の寮生に対してより細かいケアが可能となっているのが特徴です． 17:00～翌日9:00までは寮母が勤務する体制となっています．また平成24年度には，開寮以来使用されてきた1号館の建替工事が行われました． 新1号館は女子寮として利用し，旧女子寮である7号館は男子寮（1年生男子）として利用しています．寮内の居室には机・ベッド・戸棚が備え付けられており，補食室，談話室，集会室などの施設が整備されています． さらに，平成13年度末に本校で行われた「高速キャンパス情報ネットワーク」の整備，平成21年度末の1～3号館への無線LANアクセスポイント設置により，直野寮も本格的にコンピュータネットワークへの接続を果たしました． 現在，全居室よりインターネットが利用可能となっています． このような環境の中で，寮生は1年から5年生まで，5歳の年齢差を感じさせないほど，和気あいあいと生活しています． 寮生で組織される寮生会の活動も活発で，寮祭・球技大会・カラオケ大会など沢山の行事を自主的かつ積極的に行っています．
'''

instruction = '松江高専直野寮について教えてください'

#query = prompt_no_input.format(instruction = instruction)
query = prompt_input.format(input = context, instruction = instruction)
#print(query)

input_ids = tokenizer.encode(
    query,
    add_special_tokens=False,
    return_tensors="pt"
)
#print(input_ids.shape)

tokens = model.generate(
    input_ids.to(device=model.device),
    max_new_tokens=256, #128,
    temperature=0.99,
    top_p=0.95,
    do_sample=True,
    #streamer=streamer,
)
#print(tokens)

output = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(output)
#print(model.device)
