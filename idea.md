# RareCellX — AI Inference API Build Guide
> Complete specification for building the RareCellX FastAPI inference service.  
> Built for AI IDEs (Cursor, Windsurf, etc.) — follow exactly to avoid errors.

---

## Project Overview

RareCellX is an AI-powered rare disease cell state classifier.  
It takes a `.h5ad` (AnnData) file containing single-cell RNA sequencing data,  
runs it through a fine-tuned Geneformer model (BERT-based, 316M params),  
and returns per-cell T1D classification probabilities + top disease-driving genes.

**Model:** Fine-tuned `ctheodoris/Geneformer` (gc104M variant)  
**Task:** Binary classification — T1D cell vs Healthy cell  
**Input:** `.h5ad` file (AnnData format from scanpy)  
**Output:** JSON with cell predictions + attention-based top genes  
**Achieved metrics:** F1: 0.9933 | Accuracy: 0.9956 | AUROC: 0.9998

---

## Project Structure

```
rarecellx/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entrypoint
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── predict.py           # /predict endpoint
│   │   └── health.py            # /health endpoint
│   ├── services/
│   │   ├── __init__.py
│   │   ├── tokenizer.py         # scRNA → Geneformer tokens
│   │   ├── inference.py         # model forward pass + attention
│   │   └── gene_mapper.py       # token IDs → Ensembl IDs → gene names
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response schemas
│   └── core/
│       ├── __init__.py
│       └── config.py            # env vars and constants
├── model/
│   ├── model.safetensors        # fine-tuned weights (epoch 1, 1.18GB)
│   └── config.json              # model architecture config
├── data/
│   ├── token_dictionary_gc104M.pkl   # token ID → Ensembl ID mapping
│   └── gene_name_id_dict_gc104M.pkl  # Ensembl ID → gene name mapping
├── tests/
│   ├── test_predict.py
│   └── sample.h5ad              # small test file (subset of HPAP data)
├── requirements.txt
├── .env
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI 0.110+ |
| ML Framework | PyTorch 2.0+ |
| Transformers | HuggingFace Transformers 4.38.0 |
| scRNA Processing | scanpy 1.9+, anndata 0.10+ |
| Data Validation | Pydantic v2 |
| Server | Uvicorn |
| File Handling | python-multipart |
| Environment | python-dotenv |

---

## Requirements

Create `requirements.txt`:

```txt
fastapi==0.110.0
uvicorn[standard]==0.27.0
python-multipart==0.0.9
pydantic==2.6.0
python-dotenv==1.0.0
torch==2.2.0
transformers==4.38.0
scanpy==1.9.8
anndata==0.10.6
numpy==1.26.4
pandas==2.2.0
scipy==1.12.0
scikit-learn==1.4.0
huggingface-hub==0.20.3
```

---

## Environment Variables

Create `.env` in root:

```env
MODEL_DIR=./model
DATA_DIR=./data
DEVICE=cuda          # change to cpu if no GPU
MAX_CELLS=500        # max cells to process per request
MAX_SEQ_LEN=512      # token sequence length
TOP_GENES=15         # top N genes to return per cell type
```

---

## Core Config

### `app/core/config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MODEL_DIR: str = os.getenv("MODEL_DIR", "./model")
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    DEVICE: str = os.getenv("DEVICE", "cpu")
    MAX_CELLS: int = int(os.getenv("MAX_CELLS", 500))
    MAX_SEQ_LEN: int = int(os.getenv("MAX_SEQ_LEN", 512))
    TOP_GENES: int = int(os.getenv("TOP_GENES", 15))
    
    TOKEN_DICT_PATH: str = os.path.join(os.getenv("DATA_DIR", "./data"), "token_dictionary_gc104M.pkl")
    GENE_NAME_DICT_PATH: str = os.path.join(os.getenv("DATA_DIR", "./data"), "gene_name_id_dict_gc104M.pkl")

settings = Settings()
```

---

## Pydantic Schemas

### `app/models/schemas.py`

```python
from pydantic import BaseModel
from typing import List, Optional, Dict

class GeneAttention(BaseModel):
    rank: int
    gene_name: str
    ensembl_id: str
    attention_score: float

class CellPrediction(BaseModel):
    cell_index: int
    cell_type: Optional[str]
    disease_state: Optional[str]
    t1d_probability: float
    healthy_probability: float
    predicted_label: str          # "T1D" or "Healthy"
    confidence: float

class CellTypeSummary(BaseModel):
    cell_type: str
    total_cells: int
    t1d_cells: int
    healthy_cells: int
    avg_t1d_probability: float
    top_genes: List[GeneAttention]

class PredictResponse(BaseModel):
    status: str
    total_cells_processed: int
    t1d_cells_detected: int
    healthy_cells_detected: int
    t1d_percentage: float
    cell_predictions: List[CellPrediction]
    cell_type_summaries: List[CellTypeSummary]
    model_version: str = "geneformer-t1d-epoch1"
    auroc_on_training: float = 0.9994

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    max_cells: int
```

---

## Services

### `app/services/gene_mapper.py`

```python
import pickle
from app.core.config import settings

class GeneMapper:
    def __init__(self):
        # token ID → Ensembl ID
        with open(settings.TOKEN_DICT_PATH, 'rb') as f:
            token_dict = pickle.load(f)
        self.id_to_ensembl = {v: k for k, v in token_dict.items()}
        
        # Ensembl ID → gene name
        with open(settings.GENE_NAME_DICT_PATH, 'rb') as f:
            gene_name_dict = pickle.load(f)
        self.ensembl_to_name = {v: k for k, v in gene_name_dict.items()}
    
    def token_to_gene_name(self, token_id: int) -> tuple[str, str]:
        """Returns (gene_name, ensembl_id) for a token ID."""
        ensembl_id = self.id_to_ensembl.get(token_id, f"UNKNOWN_{token_id}")
        gene_name = self.ensembl_to_name.get(ensembl_id, ensembl_id)
        return gene_name, ensembl_id

# singleton
gene_mapper = GeneMapper()
```

---

### `app/services/tokenizer.py`

```python
import numpy as np
import scanpy as sc
import anndata as ad
from typing import List, Dict
import pickle
from app.core.config import settings

class ScRNATokenizer:
    """
    Converts AnnData (.h5ad) to Geneformer token sequences.
    
    Geneformer tokenizes cells by ranking genes from highest to lowest expression,
    then maps each gene to its token ID from the vocabulary.
    Only genes present in the Geneformer vocabulary are kept.
    """
    
    def __init__(self):
        with open(settings.TOKEN_DICT_PATH, 'rb') as f:
            self.token_dict = pickle.load(f)
        # gene name → token ID (need gene names as var_names in AnnData)
        # token_dict keys are Ensembl IDs
        
        # also load gene name → ensembl mapping for flexible input
        with open(settings.GENE_NAME_DICT_PATH, 'rb') as f:
            gene_name_dict = pickle.load(f)
        # gene_name_dict: {gene_name: ensembl_id}
        self.name_to_ensembl = gene_name_dict
        self.ensembl_to_token = self.token_dict  # {ensembl_id: token_id}
    
    def tokenize_adata(self, adata: ad.AnnData) -> List[Dict]:
        """
        Takes AnnData, returns list of tokenized cells.
        
        Each cell: {
            'input_ids': List[int],   # gene token IDs ranked by expression
            'length': int,
            'label': int,             # 0=healthy, 1=T1D (if available)
            'cell_type': str,         # if available in obs
            'disease_state': str,     # if available in obs
        }
        """
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        tokenized = []
        
        # map var_names to ensembl IDs
        var_names = list(adata.var_names)
        
        # detect if var_names are gene symbols or ensembl IDs
        is_ensembl = var_names[0].startswith("ENSG")
        
        if not is_ensembl:
            # map gene symbols to ensembl IDs
            ensembl_ids = [self.name_to_ensembl.get(g, None) for g in var_names]
        else:
            ensembl_ids = var_names
        
        # map ensembl IDs to token IDs
        token_ids_per_gene = [
            self.ensembl_to_token.get(eid, None) if eid else None
            for eid in ensembl_ids
        ]
        
        # valid gene indices (those present in vocab)
        valid_indices = [i for i, t in enumerate(token_ids_per_gene) if t is not None]
        valid_token_ids = [token_ids_per_gene[i] for i in valid_indices]
        
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        for cell_idx in range(min(adata.n_obs, settings.MAX_CELLS)):
            expr = X[cell_idx, valid_indices]
            
            # rank genes by expression (descending)
            ranked_indices = np.argsort(expr)[::-1]
            ranked_tokens = [valid_token_ids[i] for i in ranked_indices if expr[i] > 0]
            ranked_tokens = ranked_tokens[:settings.MAX_SEQ_LEN]
            
            cell = {
                'input_ids': ranked_tokens,
                'length': len(ranked_tokens),
                'label': 0,
                'cell_type': 'unknown',
                'disease_state': 'unknown',
            }
            
            # add metadata if available
            if 'cell_type' in adata.obs.columns:
                cell['cell_type'] = str(adata.obs['cell_type'].iloc[cell_idx])
            if 'disease_state' in adata.obs.columns:
                ds = str(adata.obs['disease_state'].iloc[cell_idx])
                cell['disease_state'] = ds
                cell['label'] = 1 if 'T1D' in ds or 't1d' in ds.lower() else 0
            
            tokenized.append(cell)
        
        return tokenized

tokenizer_service = ScRNATokenizer()
```

---

### `app/services/inference.py`

```python
import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
from transformers import AutoModelForSequenceClassification
from app.core.config import settings
from app.services.gene_mapper import gene_mapper

class RareCellXInference:
    def __init__(self):
        self.device = torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")
        print(f"Loading RareCellX model on {self.device}...")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            settings.MODEL_DIR,
            num_labels=2,
            output_attentions=True,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()
        print(f"✅ Model loaded! Device: {self.device}")
    
    def predict_cells(self, tokenized_cells: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Runs inference on tokenized cells.
        Returns (cell_predictions, cell_type_attention_scores)
        """
        cell_predictions = []
        cell_type_attention = defaultdict(lambda: defaultdict(list))
        
        for cell in tokenized_cells:
            ids = cell['input_ids'][:settings.MAX_SEQ_LEN]
            if len(ids) == 0:
                continue
            
            input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)
            attention_mask = (input_ids != 0).long()
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                attentions = outputs.attentions
            
            # classification probabilities
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_label = "T1D" if probs[1] > 0.5 else "Healthy"
            
            cell_predictions.append({
                'cell_index': len(cell_predictions),
                'cell_type': cell.get('cell_type', 'unknown'),
                'disease_state': cell.get('disease_state', 'unknown'),
                'healthy_probability': float(probs[0]),
                't1d_probability': float(probs[1]),
                'predicted_label': predicted_label,
                'confidence': float(max(probs)),
            })
            
            # attention extraction — last layer CLS token
            last_layer_attn = attentions[-1]           # (1, heads, seq, seq)
            cls_attn = last_layer_attn[0, :, 0, :]     # (heads, seq)
            cls_attn = cls_attn.mean(dim=0).cpu().numpy()  # (seq,)
            
            cell_type = cell.get('cell_type', 'unknown')
            for idx, gene_id in enumerate(ids):
                if gene_id != 0 and idx < len(cls_attn):
                    cell_type_attention[cell_type][gene_id].append(float(cls_attn[idx]))
            
            del input_ids, attention_mask, outputs, attentions
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        return cell_predictions, dict(cell_type_attention)
    
    def get_top_genes(self, cell_type_attention: Dict, top_n: int = 15) -> Dict:
        """Converts raw attention scores to ranked top genes per cell type."""
        result = {}
        for cell_type, gene_scores in cell_type_attention.items():
            avg_scores = {
                gene_id: np.mean(scores)
                for gene_id, scores in gene_scores.items()
            }
            top_genes = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            result[cell_type] = []
            for rank, (gene_id, score) in enumerate(top_genes, 1):
                gene_name, ensembl_id = gene_mapper.token_to_gene_name(gene_id)
                result[cell_type].append({
                    'rank': rank,
                    'gene_name': gene_name,
                    'ensembl_id': ensembl_id,
                    'attention_score': float(score),
                })
        return result

# singleton — loaded once at startup
inference_service = RareCellXInference()
```

---

## Routes

### `app/routes/health.py`

```python
from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.core.config import settings
import torch

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=True,
        device=settings.DEVICE if torch.cuda.is_available() else "cpu",
        max_cells=settings.MAX_CELLS
    )
```

---

### `app/routes/predict.py`

```python
import io
import tempfile
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import PredictResponse, CellPrediction, CellTypeSummary, GeneAttention
from app.services.tokenizer import tokenizer_service
from app.services.inference import inference_service
from app.core.config import settings
import anndata as ad
from collections import defaultdict
import numpy as np

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload a .h5ad file → get T1D cell classification + top disease genes.
    
    - Accepts: .h5ad (AnnData format from scanpy)
    - Max cells processed: controlled by MAX_CELLS env var (default 500)
    - Returns: per-cell predictions + cell-type-level gene attention summary
    """
    
    if not file.filename.endswith('.h5ad'):
        raise HTTPException(status_code=400, detail="Only .h5ad files are accepted")
    
    # save upload to temp file
    with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # load AnnData
        adata = ad.read_h5ad(tmp_path)
        
        if adata.n_obs == 0:
            raise HTTPException(status_code=400, detail="Empty AnnData file")
        
        # tokenize
        tokenized_cells = tokenizer_service.tokenize_adata(adata)
        
        if len(tokenized_cells) == 0:
            raise HTTPException(status_code=400, detail="No valid cells after tokenization. Check gene names match Geneformer vocabulary.")
        
        # inference + attention
        cell_predictions_raw, cell_type_attention = inference_service.predict_cells(tokenized_cells)
        top_genes_per_celltype = inference_service.get_top_genes(
            cell_type_attention, 
            top_n=settings.TOP_GENES
        )
        
        # build response
        cell_predictions = [CellPrediction(**p) for p in cell_predictions_raw]
        
        t1d_cells = sum(1 for p in cell_predictions if p.predicted_label == "T1D")
        healthy_cells = len(cell_predictions) - t1d_cells
        
        # cell type summaries
        cell_type_groups = defaultdict(list)
        for p in cell_predictions:
            cell_type_groups[p.cell_type].append(p)
        
        summaries = []
        for ct, preds in cell_type_groups.items():
            t1d_count = sum(1 for p in preds if p.predicted_label == "T1D")
            avg_t1d_prob = float(np.mean([p.t1d_probability for p in preds]))
            top_genes = [
                GeneAttention(**g) 
                for g in top_genes_per_celltype.get(ct, [])
            ]
            summaries.append(CellTypeSummary(
                cell_type=ct,
                total_cells=len(preds),
                t1d_cells=t1d_count,
                healthy_cells=len(preds) - t1d_count,
                avg_t1d_probability=avg_t1d_prob,
                top_genes=top_genes
            ))
        
        return PredictResponse(
            status="success",
            total_cells_processed=len(cell_predictions),
            t1d_cells_detected=t1d_cells,
            healthy_cells_detected=healthy_cells,
            t1d_percentage=round(t1d_cells / len(cell_predictions) * 100, 2),
            cell_predictions=cell_predictions,
            cell_type_summaries=summaries,
        )
    
    finally:
        os.unlink(tmp_path)
```

---

## Main App

### `app/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict, health

app = FastAPI(
    title="RareCellX",
    description="AI-powered rare disease cell state classifier using single-cell RNA sequencing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Inference"])

@app.on_event("startup")
async def startup():
    print("🧬 RareCellX API started")
    print("📖 Docs available at /docs")
```

---

## Running the API

```bash
# install dependencies
pip install -r requirements.txt

# run dev server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# run production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

API will be live at `http://localhost:8000`  
Swagger docs at `http://localhost:8000/docs`

---

## Testing the API

```bash
# health check
curl http://localhost:8000/health

# predict (upload h5ad file)
curl -X POST http://localhost:8000/predict \
  -F "file=@./tests/sample.h5ad"
```

### Python test client:

```python
import requests

url = "http://localhost:8000/predict"
with open("hpap_t1d_processed.h5ad", "rb") as f:
    response = requests.post(url, files={"file": f})

result = response.json()
print(f"Total cells: {result['total_cells_processed']}")
print(f"T1D detected: {result['t1d_cells_detected']} ({result['t1d_percentage']}%)")

for summary in result['cell_type_summaries']:
    print(f"\n{summary['cell_type']}: {summary['t1d_cells']}/{summary['total_cells']} T1D")
    for gene in summary['top_genes'][:5]:
        print(f"  {gene['rank']}. {gene['gene_name']} ({gene['attention_score']:.4f})")
```

---

## Expected API Response (sample)

```json
{
  "status": "success",
  "total_cells_processed": 500,
  "t1d_cells_detected": 162,
  "healthy_cells_detected": 338,
  "t1d_percentage": 32.4,
  "model_version": "geneformer-t1d-epoch1",
  "auroc_on_training": 0.9994,
  "cell_type_summaries": [
    {
      "cell_type": "type B pancreatic cell",
      "total_cells": 120,
      "t1d_cells": 98,
      "healthy_cells": 22,
      "avg_t1d_probability": 0.847,
      "top_genes": [
        {"rank": 1, "gene_name": "INS", "ensembl_id": "ENSG00000254647", "attention_score": 0.0977},
        {"rank": 2, "gene_name": "CHGA", "ensembl_id": "ENSG00000100604", "attention_score": 0.0468},
        {"rank": 3, "gene_name": "KCNJ11", "ensembl_id": "ENSG00000187486", "attention_score": 0.0432}
      ]
    }
  ]
}
```

---

## Key Implementation Notes

1. **Model loading** happens once at startup via singleton pattern — do not reload per request
2. **GPU memory** — fp16 inference cuts VRAM from ~8GB to ~4GB. If running on CPU, float32 is used automatically
3. **MAX_CELLS=500** — Geneformer inference is slow on CPU (~2s/cell). Limit cells for demo
4. **Tokenization** — var_names in AnnData must be either gene symbols (e.g. `INS`) or Ensembl IDs (e.g. `ENSG00000254647`). Both are handled
5. **Attention extraction** uses last transformer layer, CLS token attention, averaged across all 18 heads
6. **Token dictionary files** must be downloaded from HuggingFace and placed in `data/`:
   - `geneformer/token_dictionary_gc104M.pkl`
   - `geneformer/gene_name_id_dict_gc104M.pkl`

### Download vocab files:
```python
from huggingface_hub import hf_hub_download

hf_hub_download("ctheodoris/Geneformer", "geneformer/token_dictionary_gc104M.pkl", local_dir="./data")
hf_hub_download("ctheodoris/Geneformer", "geneformer/gene_name_id_dict_gc104M.pkl", local_dir="./data")
```

---

## Biology Context (for understanding outputs)

- **T1D label = 1** — Type 1 Diabetes cell (autoimmune destruction of beta cells)
- **Healthy label = 0** — Normal pancreatic cell
- **Top genes** = genes model attended to most when classifying — not causal genes but **disease-state markers**
- **INS** (insulin) appearing in top genes = dysregulated insulin expression in T1D beta cells
- **PTPRN** = major T1D autoantigen attacked by immune system
- **KCNJ11** = potassium channel controlling insulin secretion, T1D GWAS hit
- **CHGA/CHGB** = chromogranins released under immune stress in islet cells

---

## Future Extensions (Phase 2)

- [ ] MedGemma integration — generate clinical report from predictions
- [ ] Gaucher Disease classifier (same pipeline, different fine-tuning)
- [ ] Lupus + Multiple Sclerosis classifiers
- [ ] Frontend dashboard (React + recharts)
- [ ] Docker deployment
- [ ] Epoch 3 model weights (F1: 0.9933, AUROC: 0.9998)