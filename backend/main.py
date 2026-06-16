from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import INDEX_HTML, STATIC_DIR
from .inference import PredictorService
from .schemas import ClassesResponse, HealthResponse, PredictRequest, PredictResponse

app = FastAPI(title="QuickDraw Real-Time Guessing", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

predictor = PredictorService()
predictor.load_model()


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(INDEX_HTML)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=predictor.model_loaded)


@app.get("/api/classes", response_model=ClassesResponse)
def classes() -> ClassesResponse:
    return ClassesResponse(classes=predictor.get_classes())


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if not predictor.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train model first: python ml/train.py",
        )

    try:
        result = predictor.predict(payload.image_b64, payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc

    return PredictResponse(**result)
