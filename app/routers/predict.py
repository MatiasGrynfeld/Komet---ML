from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from ..models.test import predict_seismic_impact

router = APIRouter()

class PredictionData(BaseModel):
    mass: float
    velocity: float
    latitude: float
    longitude: float

class PredictionResponse(BaseModel):
    success: bool
    magnitude: float
    energy: float
    cluster: int
    alert_level: str
    alert_level_numeric: int
    intensity_mmi: float
    community_intensity_cdi: float
    significance: float
    depth: float
    tsunami_warning: bool
    location: dict
    asteroid_properties: dict
    message: Optional[str] = None

@router.get("/")
async def health():
    return {"status": "ok"}

@router.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionData):
    try:
        if data.mass <= 0:
            raise HTTPException(status_code=400, detail="Mass must be positive")
        if data.velocity <= 0:
            raise HTTPException(status_code=400, detail="Velocity must be positive")
        if not -90 <= data.latitude <= 90:
            raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90 degrees")
        if not -180 <= data.longitude <= 180:
            raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180 degrees")
        
        result, energy, cluster = predict_seismic_impact(
            mass_kg=data.mass,
            velocity_ms=data.velocity,
            latitude=data.latitude,
            longitude=data.longitude
        )
        
        if result is None:
            raise HTTPException(status_code=500, detail="Prediction model failed to load or execute")
        
        alert_names = {0: "No Alert", 1: "Green", 2: "Yellow", 3: "Orange", 4: "Red"}
        alert_level_numeric = int(result['alert'])
        alert_level = alert_names.get(alert_level_numeric, "Unknown")
        
        response = PredictionResponse(
            success=True,
            magnitude=float(result['mag']),
            energy=float(energy),
            cluster=int(cluster),
            alert_level=alert_level,
            alert_level_numeric=alert_level_numeric,
            intensity_mmi=float(result['mmi']),
            community_intensity_cdi=float(result['cdi']),
            significance=float(result['sig']),
            depth=float(result['depth']),
            tsunami_warning=bool(result['tsunami']),
            location={
                "latitude": data.latitude,
                "longitude": data.longitude
            },
            asteroid_properties={
                "mass_kg": data.mass,
                "velocity_ms": data.velocity,
                "kinetic_energy_joules": energy
            },
            message=f"Predicted seismic impact: Magnitude {result['mag']:.2f}, {alert_level} alert level"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")