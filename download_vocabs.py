from huggingface_hub import hf_hub_download
import os

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    print("Downloading Geneformer vocabulary dicts to ./data ...")
    hf_hub_download(repo_id="ctheodoris/Geneformer", filename="geneformer/token_dictionary_gc104M.pkl", local_dir="./data", local_dir_use_symlinks=False)
    hf_hub_download(repo_id="ctheodoris/Geneformer", filename="geneformer/gene_name_id_dict_gc104M.pkl", local_dir="./data", local_dir_use_symlinks=False)
    print("Download complete.")
