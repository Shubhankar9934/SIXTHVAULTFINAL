from fastapi import APIRouter
from typing import Dict, Any
from app.utils.s3_storage import check_s3_health
from app.config import settings
import asyncio
import time

router = APIRouter(tags=["health"])

@router.get("/health")
@router.head("/health")
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check including S3 storage"""
    start_time = time.time()
    
    health_status = {
        "status": "healthy",
        "timestamp": start_time,
        "services": {},
        "overall_status": "healthy"
    }
    
    # Check S3 storage health
    try:
        s3_health = await check_s3_health()
        health_status["services"]["s3_storage"] = s3_health
        
        if s3_health["status"] != "healthy":
            health_status["overall_status"] = "degraded"
            
    except Exception as e:
        health_status["services"]["s3_storage"] = {
            "status": "unhealthy",
            "error": str(e),
            "message": "S3 health check failed"
        }
        health_status["overall_status"] = "unhealthy"
    
    # Add basic system health
    health_status["services"]["api"] = {
        "status": "healthy",
        "message": "API is operational"
    }
    
    # Calculate response time
    health_status["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
    
    return health_status

@router.get("/health/s3")
@router.head("/health/s3")
async def s3_health_check() -> Dict[str, Any]:
    """Dedicated S3 health check endpoint"""
    return await check_s3_health()

@router.get("/config/features")
@router.head("/config/features")
async def get_feature_config() -> Dict[str, Any]:
    """Get feature configuration including Ollama enabled status"""
    return {
        "ollama_enabled": settings.ollama_enabled,
        "bedrock_enabled": settings.bedrock_enabled,
        "features": {
            "local_ai": settings.ollama_enabled,
            "cloud_ai": settings.bedrock_enabled,
            "s3_storage": bool(settings.aws_s3_access_key_id),
            "authentication": True
        },
        "providers": {
            "ollama": {
                "enabled": settings.ollama_enabled,
                "host": settings.ollama_host if settings.ollama_enabled else None
            },
            "bedrock": {
                "enabled": settings.bedrock_enabled,
                "region": settings.aws_bedrock_region if settings.bedrock_enabled else None
            }
        }
    }
