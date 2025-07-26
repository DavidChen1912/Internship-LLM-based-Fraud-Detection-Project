import argparse
import os
import zipfile
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# ----------- 參數解析 -----------
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Model name inside model folder (e.g., yi_v1)')
parser.add_argument('--data', type=str, required=True, help='PDF filename inside data folder (e.g., rag_data.pdf)')
args = parser.parse_args()

# ----------- 解壓模型 -----------
model_zip_path = f"./model/{args.model}.zip"
model_extract_path = f"./model/{args.model}"
if not os.path.exists(model_extract_path):
    with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_extract_path)

# ----------- 載入模型與 tokenizer -----------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_extract_path)
base_model = AutoModelForCausalLM.from_pretrained(
    model_extract_path,
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, model_extract_path)

# ----------- 嵌入模型（使用多語言語義檢索模型） -----------
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# ----------- 讀取並處理 PDF -----------
DOC_PATH = f"./data/{args.data}"
loader = PyPDFLoader(DOC_PATH)
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
documents = text_splitter.split_documents(pages)

# ----------- 建立向量資料庫 -----------
CHROMA_PATH = "./chroma_db"
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_PATH)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ----------- System Prompt -----------
system_prompt = """
請判斷以下對話是詐騙還是合理投資，並基於「關鍵因素」來解釋您的判斷。列出關鍵對話，並解釋該對話與關鍵因素的連結。請用簡單的語言讓無背景知識的使用者也能理解，並在700字內完結對話。

參考範例：
<此對話是詐騙行為，原因在於對話中多次出現詐騙的「關鍵因素」，讓我們可以判斷這是一種典型的投資詐騙模式。以下列出幾個關鍵對話並附上判斷依據：
1.誘導加入「免費」群組或平台
關鍵對話：「我們老師的投資策略屢屢命中，幫不少人賺了不少錢呢。」 以及 「免費資金只會提供一小段時間，如果您想繼續投資，可能需要自行儲值一些。」
詐騙因素：詐騙者通常會用免費群組、免費資金、或一開始的「免費試用」來吸引目標對象入局，讓人感到投入無風險，增加信任度。然而，一旦投入資金，後續的「額外費用」要求便開始出現，讓人不斷掏錢。

小結：
依照以上的「誘導免費群組」因素，可以清楚判斷這是一起詐騙事件。在真實的投資環境中，沒有任何正規平台會以此方式吸引投資或索取不合理的資金，因此判斷此對話不合理且具高風險。>
"""

# ----------- 使用 apply_chat_template 的生成函數 -----------
def generate_response_with_template(prompt_template):
    messages = [
        {"role": "user", "content": prompt_template}  # 將整個 prompt_template 當成 user prompt
    ]
    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')

    output_ids = model.generate(
        input_ids.to(model.device),
        max_new_tokens=850,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.7,
    )
    token = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(token, skip_special_tokens=True)
    return response, token

# ----------- 互動模式 -----------
while True:
    query = input("請輸入你想判斷的對話內容（或輸入 'exit' 離開）：\n")
    if query.lower().strip() == 'exit':
        break

    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt_template = f"""
{system_prompt}

以下是檢索到的相關背景資訊，僅供參考，非需要判定的問題內容：
{context if context else "未能檢索到相關資訊。"}

請幫我判斷以下對話是詐騙，還是正常的投資？並基於你的回答描述你在判斷時，是依據什麼「關鍵因素」來分辨這則對話。請將「關鍵對話」呈現出來，並且解釋這則對話與該關鍵因素的連結。關鍵因素要淺顯易懂，讓無背景知識的使用者也可以了解：
{query}

請僅針對上述「需要判定的對話」進行分析。
"""

    response, _ = generate_response_with_template(prompt_template)
    print("\n生成回應如下：\n")
    print(response)
    print("\n===============================\n")

# python3 rag.py --model=yi_v1 --data=rag_data.pdf

