from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.core.config import settings
import torch

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=True,
        device=settings.DEVICE if torch.cuda.is_available() else "cpu",
        max_cells=settings.MAX_CELLS
    )
