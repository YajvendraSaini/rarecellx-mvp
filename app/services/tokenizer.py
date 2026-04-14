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
