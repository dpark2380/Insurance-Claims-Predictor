"""Pydantic models for the dashboard API."""
from typing import Literal
from pydantic import BaseModel, Field


class Policy(BaseModel):
    VehPower: int = Field(..., ge=4, le=15)
    VehAge: int = Field(..., ge=0, le=100)
    DrivAge: int = Field(..., ge=18, le=100)
    BonusMalus: int = Field(..., ge=50, le=230)
    Density: int = Field(..., ge=1, le=27000)
    Exposure: float = Field(default=1.0, gt=0.0, le=2.5)
    Area: str
    VehBrand: str
    VehGas: str
    Region: str


class FeatureAttribution(BaseModel):
    feature: str
    contribution: float
    value: str | int | float


class ModelPrediction(BaseModel):
    model: Literal['GLM', 'XGBoost', 'Neural Net']
    frequency_rate: float
    severity_eur: float
    pure_premium_eur: float
    portfolio_mean_pure_premium: float
    delta_vs_portfolio: float
    attributions: list[FeatureAttribution]


class PredictionResponse(BaseModel):
    profile: Policy
    predictions: list[ModelPrediction]


class BandContext(BaseModel):
    bonus_malus_band: str | None
    age_band: str | None
    bm_band_means: dict     # {model: {band: mean_pp}}
    age_band_means: dict    # {model: {band: mean_pp}}


class SchemaResponse(BaseModel):
    numeric_ranges: dict
    vocab: dict
    display_vocab: dict
    bands: dict
