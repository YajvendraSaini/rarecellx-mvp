import torch
import numpy as np
import os
from collections import defaultdict
from typing import List, Dict, Tuple
from transformers import AutoModelForSequenceClassification
from app.core.config import settings
from app.services.gene_mapper import gene_mapper

class RareCellXInference:
    def __init__(self):
        self.device = torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")
        print(f"Loading RareCellX model on {self.device}...")
        
        model_path = os.path.join(settings.MODEL_DIR, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please place the model there.")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            settings.MODEL_DIR,
            num_labels=2,
            output_attentions=True,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()
        print(f"[SUCCESS] Model loaded! Device: {self.device}")
    
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
