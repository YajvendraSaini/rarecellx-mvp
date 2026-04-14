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
