import anndata as ad
import numpy as np
import pandas as pd

def generate_mock_data():
    print("Generating synthetic single-cell RNA-seq data...")
    
    # Common genes that definitely exist in Geneformer vocab
    # Including some top disease markers for T1D (INS, PTPRN, CHGA)
    gene_names = [
        "INS", "CHGA", "CHGB", "KCNJ11", "PTPRN", 
        "GAPDH", "ACTB", "HLA-A", "HLA-B", "HLA-C",
        "GCG", "SST", "PPY", "GHRL", "IAPP",
        "CD4", "CD8A", "IL2RA", "FOXP3", "CD19"
    ]
    
    n_cells = 100
    n_genes = len(gene_names)
    
    # 1. Generate random count data (simulating scRNA-seq expression)
    # Using Poisson distribution for realistic-looking sparse count data
    X = np.random.poisson(lam=1.5, size=(n_cells, n_genes)).astype(np.float32)
    
    # Emphasize INS and PTPRN in the first 50 cells (T1D Beta cells)
    X[:50, 0] += np.random.poisson(lam=10, size=50) # INS
    X[:50, 4] += np.random.poisson(lam=5, size=50)  # PTPRN
    
    # 2. Setup Variables (Genes)
    var = pd.DataFrame(index=gene_names)
    
    # 3. Setup Observations (Cells)
    cell_types = ["type B pancreatic cell"] * 50 + ["type A pancreatic cell"] * 30 + ["T cell"] * 20
    disease_states = ["T1D"] * 30 + ["normal"] * 20 + ["normal"] * 30 + ["T1D"] * 10 + ["normal"] * 10
    
    obs = pd.DataFrame({
        "cell_type": cell_types,
        "disease_state": disease_states
    }, index=[f"cell_{i}" for i in range(n_cells)])
    
    # 4. Create AnnData Object
    adata = ad.AnnData(X=X, obs=obs, var=var)
    
    # 5. Save to h5ad
    output_path = "mock_patient_data.h5ad"
    adata.write(output_path)
    print(f"\n✅ Success! Created {output_path}")
    print(f"Contains {n_cells} cells and {n_genes} genes.")

if __name__ == "__main__":
    generate_mock_data()
