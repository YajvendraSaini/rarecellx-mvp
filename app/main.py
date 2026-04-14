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
    print("[INFO] RareCellX API started")
    print("[INFO] Docs available at /docs")
