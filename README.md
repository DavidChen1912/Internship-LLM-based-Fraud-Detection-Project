# Internship – Taishin International Bank  
**LLM-based Fraud Detection Project**

## ⚠️ Disclaimer  
This repository documents my internship project at **Taishin International Bank**.  
Due to the sensitivity of internal data, all training examples in this repository are either **synthetic (GPT-generated)** or based on **open-source datasets**. **No proprietary or confidential company data is included.**

---

## 📖 Introduction  
In Taiwan, financial fraud has become increasingly rampant in recent years, with scams relying heavily on **localized language patterns and cultural nuances**. Off-the-shelf open-source models often fail to capture these regional fraud characteristics.  

To address this, our team developed a **localized fraud-detection LLM** aimed at helping users safely analyze suspicious investment scenarios in a private environment. **The full project pipeline and documentation** can be found under the [`/docs`](./docs) directory.  

---

## 🚀 Usage  

Run the following commands in your terminal:  

```
# Train the model 
python3 train.py --data=train_data.csv --batch_size=1 --grad_steps=2 --lr=1e-5 --max_steps=1200

# Apply Retrieval-Augmented Generation (RAG)
python3 rag.py --model=yi_v1 --data=rag_data.pdf

# Launch the chatbot for inference
python3 chat.py
```

#### Notes:
- data is required and should point to your dataset file.
- Hyperparameters (e.g., batch_size, grad_steps, lr, max_steps) should be adjusted based on your dataset and analysis needs.
- By default, trained models from train.py are stored in the [`/model`](./model) directory.

#### Scripts:
- **`train.py`**: Fine-tunes an LLM using QLoRA.
- **`rag.py`**: Performs Retrieval-Augmented Generation (RAG) with the specified model.
- **`chat.py`**: Starts an interactive chatbot for user queries.




