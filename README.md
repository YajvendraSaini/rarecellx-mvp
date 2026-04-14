<h1 align="center">
  <br>
  <img src="frontend/public/window.svg" alt="RareCellX Logo" width="120">
  <br>
  RareCellX AI
  <br>
</h1>

<h4 align="center">Advanced AI-powered Rare Disease Cell State Classifier.</h4>

<p align="center">
  <a href="#about-the-project">About</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#complete-setup-guide">Setup Guide</a> •
  <a href="#local-preview">Local Preview</a> •
  <a href="#biological-context-outputs">Biology Context</a>
</p>

---

## 🧬 About the Project
**RareCellX** is an end-to-end AI platform designed to analyze single-cell RNA sequencing (scRNA-seq) data. Using a fine-tuned version of the **Geneformer model (316M parameters, BERT-based)**, the platform takes in `.h5ad` datasets to instantly classify cell health states, currently specializing in **Type 1 Diabetes (T1D)** classification. 

Rather than just providing a black-box label, RareCellX uses Transformer attention mechanisms to identify and surface **disease-driving genes** for each individual cell in the user interface.

**Performance Metrics on T1D Classification:**
* **F1 Score:** `0.9933`
* **Accuracy:** `0.9956`
* **AUROC:** `0.9998`

## ⚙️ How it Works
1. **Frontend (Next.js):** Provides a visual, user-centric dashboard where researchers can upload AnnData files.
2. **Backend (FastAPI):** A high-performance async API that receives datasets, tokenizes scRNA expressions, and routes them through the AI model. 
3. **Inference (PyTorch & HuggingFace):** The core transformer dynamically extracts attention scores per gene to map the genetic signature of unhealthy cells.

---

## 🚀 Complete Setup Guide

Follow these instructions strictly to get the local development environment up and running.

### Prerequisites
* **Python** `>= 3.10`
* **Node.js** `>= 18` (for the frontend)
* **Git LFS** (if downloading large models)

### 1. Clone & Prepare the Repository
```bash
git clone https://github.com/your-username/RareCellX.git
cd RareCellX
```

### 2. Backend Setup (FastAPI + AI Inference)

**Create a Virtual Environment**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

**Install Python Dependencies**
```bash
pip install -r requirements.txt
```

**Download Required AI Vocabulary/Mapping Files**
The custom transformer requires specific vocabulary dictionaries to map genes correctly. We provided a script to download these from HuggingFace to your `data/` folder automatically:
```bash
python download_vocabs.py
```
*(This ensures `token_dictionary_gc104M.pkl` and `gene_name_id_dict_gc104M.pkl` are located safely inside `data/`)*

### ⚠️ IMPORTANT: Adding Your Model & Datasets ⚠️
Because AI model files and genetic datasets are incredibly massive, they are **ignored by Git** (via `.gitignore`) to prevent repository bloating. 

**You MUST manually place your required resources into the project:**
1. **Model Weights:** Place your fine-tuned `model.safetensors` file into the `model/` directory. *(Ensure `model/config.json` is also present there).*
2. **Datasets (Optional for testing):** Place any `.h5ad` files you wish to test inside the root directory or `data/` directory.

### 3. Frontend Setup (Next.js UI)

Open a **separate terminal window** for your frontend.

```bash
cd frontend
npm install
```

---

## 🖥 Local Preview

Once setup is complete, you need to spin up both the FastAPI backend and the Next.js frontend to view the complete cohesive app.

### Terminal 1: Start the Backend Inference API
```bash
# Ensure your python virtual environment is activated!
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
*API will run at `http://localhost:8000`*
*Interactive API Swagger Docs at `http://localhost:8000/docs`*

### Terminal 2: Start the Frontend UI
```bash
cd frontend
npm run dev
```
*Website will be live at `http://localhost:3000`*

You can now navigate to `http://localhost:3000` in your web browser. When you upload a `.h5ad` file from the dashboard, it will pass through your local inference API and visualize the results.

---

## 🧪 Testing the API via Terminal (No UI)

Want to ping the inference service directly? You can use `curl` from a terminal:

**1. Health Check:**
```bash
curl http://localhost:8000/health
```

**2. Make predicting an AnnData file:**
```bash
curl -X POST http://localhost:8000/predict -F "file=@./mock_patient_data.h5ad"
```

---

## 🔬 Biological Context (Understanding the Outputs)
When testing RareCellX, the system extracts the genes that the transformer model "paid the most attention to" when deciding if a cell is diseased.

- **T1D Label:** Type 1 Diabetes Cell (Implies autoimmune destruction in beta cells).
- **Healthy Label:** Normal pancreatic cell expression.
- **Top Genes:** Genes the model heavily weighted for the classification.
  - **INS (Insulin):** Appearing in top genes implies dysregulated insulin expression in T1D beta cells.
  - **PTPRN:** A major T1D autoantigen attacked by the immune system.
  - **KCNJ11/CHGA:** Control insulin secretion and react to immune stress in islet cells.

---

## 🛡 License
This project is currently proprietary.
