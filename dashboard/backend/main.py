"""FastAPI app for the insurance claims dashboard."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import Policy, PredictionResponse, SchemaResponse
from models_loader import Bundle


bundle: Bundle | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle
    bundle = Bundle()
    yield


app = FastAPI(title="Insurance Claims Dashboard API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/schema", response_model=SchemaResponse)
def get_schema():
    if bundle is None:
        raise HTTPException(503, "Models not loaded")
    p = bundle.preprocessing
    return SchemaResponse(
        numeric_ranges=p['numeric_ranges'],
        vocab=p['vocab'],
        display_vocab=p['display_vocab'],
        bands={
            'bonus_malus': bundle.portfolio_stats['pp_by_bonus_malus_band'],
            'age': bundle.portfolio_stats['pp_by_age_band'],
        },
    )


@app.post("/api/predict", response_model=PredictionResponse)
def predict(policy: Policy):
    if bundle is None:
        raise HTTPException(503, "Models not loaded")
    predictions = bundle.predict_all(policy.model_dump())
    return PredictionResponse(profile=policy, predictions=predictions)


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": bundle is not None}
