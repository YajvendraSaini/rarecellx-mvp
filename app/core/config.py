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
    
    TOKEN_DICT_PATH: str = os.path.join(os.getenv("DATA_DIR", "./data"), "geneformer", "token_dictionary_gc104M.pkl")
    GENE_NAME_DICT_PATH: str = os.path.join(os.getenv("DATA_DIR", "./data"), "geneformer", "gene_name_id_dict_gc104M.pkl")

settings = Settings()
