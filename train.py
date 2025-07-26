import os
import argparse
import zipfile
import torch
from datasets import load_dataset
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from trl import SFTTrainer

# ===== 解析參數 =====
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="CSV 檔案名稱（位於 data/ 資料夾中）")
parser.add_argument("--batch_size", type=int, default=1, help="每裝置訓練批次大小")
parser.add_argument("--grad_steps", type=int, default=2, help="梯度累積步數")
parser.add_argument("--lr", type=float, default=1e-5, help="學習率")
parser.add_argument("--max_steps", type=int, default=1200, help="最大訓練步數")
args = parser.parse_args()

# ===== 載入資料 =====
data_path = os.path.join("data", args.data)
print(f"📥 載入資料：{data_path}")
dataset = load_dataset("csv", data_files=data_path)
training_data = dataset["train"].train_test_split(test_size=0.2)

# ===== 格式化 prompt =====
def formatting_func(example):
    if example.get("context", "") != "":
        input_prompt = (
            "以下是描述任務的指示，以及與任務相關的情境。"
            "請根據指示生成符合需求的回應。\n\n"
            "### 指示:\n"
            f"{example['instruction']}\n\n"
            "### 情境:\n"
            f"{example['context']}\n\n"
            "### 回應:\n"
            f"{example['response']}"
        )
    else:
        input_prompt = (
            "以下是描述任務的指示。"
            "請根據指示生成符合需求的回應。\n\n"
            "### 指示:\n"
            f"{example['instruction']}\n\n"
            "### 回應:\n"
            f"{example['response']}"
        )
    return {"text": input_prompt}

formatted_data = training_data.map(formatting_func)

# ===== 載入模型與 tokenizer =====
model_id = "01-ai/Yi-1.5-6B-Chat"

qlora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# ===== 建立 SFT 訓練器 =====
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_steps,
    learning_rate=args.lr,
    max_steps=args.max_steps,
    output_dir="output",
    optim="paged_adamw_8bit",
    fp16=True,
    report_to="none",
    logging_steps=100,
    eval_steps=100,
    evaluation_strategy="steps"
)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=formatted_data["train"],
    eval_dataset=formatted_data["test"],
    args=training_args,
    tokenizer=tokenizer,
    peft_config=qlora_config
)

# ===== 執行訓練 =====
print("🚀 開始訓練...")
trainer.train()
print("✅ 訓練完成")

# ===== 儲存模型並壓縮成 zip =====
os.makedirs("model_tmp", exist_ok=True)
trainer.save_model("model_tmp")
tokenizer.save_pretrained("model_tmp")

os.makedirs("model", exist_ok=True)
output_zip = os.path.join("model", "yi_v1.zip")
with zipfile.ZipFile(output_zip, 'w') as zipf:
    for root, _, files in os.walk("model_tmp"):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, "model_tmp")
            zipf.write(full_path, arcname)

print(f"🎉 模型已壓縮儲存至：{output_zip}")

# python3 train.py --data=train_data.csv --batch_size=1 --grad_steps=2 --lr=1e-5 --max_steps=1200

