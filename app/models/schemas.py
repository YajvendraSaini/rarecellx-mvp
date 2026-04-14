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
