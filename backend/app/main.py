from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.core.config import get_settings
from app.core.logging import configure_logging

settings = get_settings()
configure_logging(settings.debug)

app = FastAPI(title=settings.app_name)

        
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(chat_router, prefix=settings.api_prefix)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/debug/cors")
def debug_cors():
    return {
        "frontend_origins": settings.frontend_origins,
        "frontend_origins_list": settings.frontend_origins_list,
    }