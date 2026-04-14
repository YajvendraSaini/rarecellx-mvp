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
