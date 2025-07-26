import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ===== 模型設定 =====
model_id = "01-ai/Yi-1.5-6B-Chat"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# ===== Tokenizer 設定 =====
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# ===== System Prompt 固定 =====
system_prompt = """
請幫我判斷以下對話是詐騙，還是正常的投資？
並基於你的回答描述你在判斷時，是依據什麼「關鍵因素」來分辨這則對話。
請將「關鍵對話」呈現出來，並且解釋該對話與該關鍵因素的連結。
關鍵因素要淺顯易懂，讓無背景知識的使用者也可以理解：
"""

# ===== 生成函數（採用原始邏輯） =====
device = "cuda"
def generate_response_with_template(query, system_prompt=None):
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    else:
        messages = [
            {"role": "user", "content": query}
        ]

    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')

    output_ids = base_model.generate(
        input_ids.to(device),
        max_new_tokens=850,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
    )
    token = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(token, skip_special_tokens=True)
    return response, token

# ===== 互動介面 =====
if __name__ == "__main__":
    while True:
        query = input("請輸入你想判斷的對話內容（或輸入 'exit' 離開）：\n")
        if query.lower().strip() == 'exit':
            break
        response, _ = generate_response_with_template(query, system_prompt)
        print("\n生成回應如下：\n")
        print(response)
        print("\n===============================\n")

# python3 chat.py