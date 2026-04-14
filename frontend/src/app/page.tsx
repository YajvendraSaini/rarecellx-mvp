"use client";

import { useState, useRef } from "react";
import { UploadCloud, File, AlertCircle, Loader2, Activity, ShieldCheck, ShieldAlert, Cpu } from "lucide-react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith('.h5ad')) {
        setFile(droppedFile);
        setError(null);
      } else {
        setError("Only .h5ad files are accepted.");
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

    try {
      const res = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Failed to analyze file");
      }

      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main style={{ padding: "4rem 2rem", maxWidth: "1200px", margin: "0 auto" }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: "4rem" }} className="animate-fade-in">
        <div className="glass-pill" style={{ display: "inline-flex", alignItems: "center", gap: "8px", marginBottom: "1.5rem" }}>
          <Activity size={16} className="text-warning" />
          <span>Geneformer AI Powered</span>
        </div>
        <h1>Rare<span className="text-gradient">Cell</span>X</h1>
        <p className="subtitle" style={{ margin: "0 auto" }}>
          Upload your single-cell RNA sequencing data (.h5ad) to rapidly classify T1D disease states utilizing fine-tuned transformer attention.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "2rem" }}>
        
        {/* Upload Section */}
        <section className="glass-panel animate-fade-in" style={{ padding: "2rem", animationDelay: "0.1s" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
            <h2 style={{ fontSize: "1.25rem", display: "flex", alignItems: "center", gap: "10px" }}>
              <Cpu size={20} className="text-gradient" />
              Inference Engine
            </h2>
            {file && (
              <span className="glass-pill" style={{ borderColor: 'var(--success)' }}>
                Ready to Analyze
              </span>
            )}
          </div>

          <input 
            type="file" 
            accept=".h5ad" 
            ref={fileInputRef} 
            onChange={handleFileSelect} 
            style={{ display: "none" }} 
          />

          {!file ? (
            <div 
              className={`upload-zone ${isDragging ? "drag-active" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <UploadCloud size={48} style={{ color: "var(--accent-hover)", marginBottom: "1rem" }} />
              <h3 style={{ fontSize: "1.1rem", marginBottom: "0.5rem" }}>Click or drag to upload</h3>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>.h5ad AnnData files only. Max 500 cells processed per batch.</p>
            </div>
          ) : (
            <div style={{ background: "rgba(255,255,255,0.03)", padding: "1.5rem", borderRadius: "12px", border: "1px solid var(--glass-border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                <File size={32} style={{ color: "var(--accent-primary)" }} />
                <div>
                  <h4 style={{ margin: 0, fontSize: "1rem" }}>{file.name}</h4>
                  <p style={{ margin: 0, color: "var(--text-secondary)", fontSize: "0.85rem" }}>
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
              <div style={{ display: "flex", gap: "1rem" }}>
                <button className="btn-secondary" onClick={() => { setFile(null); setResult(null); }}>Cancel</button>
                <button 
                  className="btn-primary" 
                  onClick={handleAnalyze} 
                  disabled={loading}
                >
                  {loading ? <Loader2 size={18} className="animate-spin" /> : <Activity size={18} />}
                  {loading ? "Analyzing..." : "Analyze Cells"}
                </button>
              </div>
            </div>
          )}

          {error && (
            <div style={{ marginTop: "1rem", padding: "1rem", background: "var(--danger-glow)", color: "#fca5a5", borderRadius: "8px", display: "flex", alignItems: "center", gap: "8px" }}>
              <AlertCircle size={18} />
              {error}
            </div>
          )}
        </section>

        {/* Results Section */}
        {result && (
          <section className="glass-panel animate-fade-in" style={{ padding: "2rem", animationDelay: "0.2s" }}>
            <h2 style={{ fontSize: "1.5rem", marginBottom: "2rem", borderBottom: "1px solid var(--glass-border)", paddingBottom: "1rem" }}>Analysis Report</h2>
            
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem", marginBottom: "3rem" }}>
              <div style={{ background: "rgba(255,255,255,0.03)", padding: "1.5rem", borderRadius: "12px", border: "1px solid var(--glass-border)" }}>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", marginBottom: "0.5rem" }}>Total Cells</p>
                <p style={{ fontSize: "2rem", fontWeight: "600" }}>{result.total_cells_processed}</p>
              </div>
              <div style={{ background: "var(--danger-glow)", padding: "1.5rem", borderRadius: "12px", border: "1px solid rgba(239, 68, 68, 0.3)" }}>
                <p style={{ color: "#fca5a5", fontSize: "0.9rem", marginBottom: "0.5rem", display: "flex", alignItems: "center", gap: "6px" }}>
                  <ShieldAlert size={16} /> T1D Cells Found
                </p>
                <p style={{ fontSize: "2rem", fontWeight: "600", color: "#f87171" }}>
                  {result.t1d_cells_detected} <span style={{ fontSize: "1rem", opacity: 0.7 }}>({result.t1d_percentage}%)</span>
                </p>
              </div>
              <div style={{ background: "var(--success-glow)", padding: "1.5rem", borderRadius: "12px", border: "1px solid rgba(16, 185, 129, 0.3)" }}>
                <p style={{ color: "#6ee7b7", fontSize: "0.9rem", marginBottom: "0.5rem", display: "flex", alignItems: "center", gap: "6px" }}>
                  <ShieldCheck size={16} /> Healthy Cells
                </p>
                <p style={{ fontSize: "2rem", fontWeight: "600", color: "#34d399" }}>{result.healthy_cells_detected}</p>
              </div>
            </div>

            <h3 style={{ fontSize: "1.2rem", marginBottom: "1rem", color: "var(--text-secondary)" }}>Cell Type Anomalies (Top Autoantigens)</h3>
            
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              {result.cell_type_summaries?.map((summary: any, idx: number) => (
                <div key={idx} style={{ background: "rgba(0,0,0,0.2)", borderRadius: "12px", padding: "1.5rem", border: "1px solid var(--glass-border)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
                    <h4 style={{ fontSize: "1.1rem", margin: 0 }}>{summary.cell_type === 'unknown' ? 'Unknown Cell Type' : summary.cell_type}</h4>
                    <span className="glass-pill" style={{ background: summary.t1d_cells > 0 ? "var(--danger-glow)" : "var(--success-glow)", color: summary.t1d_cells > 0 ? "#fca5a5" : "#6ee7b7", borderColor: 'transparent' }}>
                      {summary.t1d_cells} / {summary.total_cells} T1D
                    </span>
                  </div>
                  
                  {summary.top_genes && summary.top_genes.length > 0 ? (
                    <div>
                      <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>Top Disease Driving Genes</p>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                        {summary.top_genes.slice(0, 5).map((gene: any, i: number) => (
                          <div key={i} style={{ background: "rgba(255,255,255,0.05)", padding: "0.5rem 1rem", borderRadius: "6px", fontSize: "0.9rem", display: "flex", alignItems: "center", gap: "8px", border: "1px solid var(--glass-border)" }}>
                            <span style={{ color: "var(--accent-hover)", fontWeight: "bold" }}>{gene.gene_name}</span>
                            <span style={{ color: "var(--text-secondary)", fontSize: "0.8rem" }}>{gene.attention_score.toFixed(3)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>No significant disease markers detected.</p>
                  )}
                </div>
              ))}
            </div>

          </section>
        )}
      </div>
    </main>
  );
}
