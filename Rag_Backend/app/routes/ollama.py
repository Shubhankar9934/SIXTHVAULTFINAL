"""
Ollama API Routes

Provides REST API endpoints for managing Ollama models and monitoring status.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional
from pydantic import BaseModel
import asyncio

from app.services.ollama_manager import (
    ollama_manager, 
    get_ollama_status, 
    ensure_model_available,
    get_best_available_model,
    setup_ollama_for_first_use
)
from app.services.llm_factory import get_llm, OllamaChat
from app.config_llm import LLM_CONFIGS
from app.config import settings

router = APIRouter(prefix="/ollama", tags=["ollama"])

# Decorator to check if Ollama is enabled
def check_ollama_enabled():
    """Decorator to check if Ollama is enabled before processing requests"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if not settings.ollama_enabled:
                raise HTTPException(
                    status_code=503, 
                    detail="Ollama functionality is disabled. Enable it in configuration to use Ollama features."
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Pydantic models for request/response
class ModelPullRequest(BaseModel):
    model_name: str
    
class ModelRemoveRequest(BaseModel):
    model_name: str

class ChatRequest(BaseModel):
    model: str
    prompt: str

class ChatResponse(BaseModel):
    response: str
    model_used: str
    response_time: float
    tokens_estimated: int

class ModelInfo(BaseModel):
    name: str
    size: int
    size_gb: float
    modified_at: str
    digest: str
    performance_score: float
    recommended_for: str

class OllamaStatusResponse(BaseModel):
    is_running: bool
    version: str
    models_count: int
    total_size: int
    total_size_gb: float
    available_models: List[ModelInfo]
    recommendations: List[str]

@router.get("/status", response_model=OllamaStatusResponse)
@check_ollama_enabled()
async def get_status():
    """Get comprehensive Ollama status"""
    try:
        status = await get_ollama_status()
        
        # Convert models to response format
        models_info = []
        for model in status.available_models:
            # Get recommended use case from config
            model_config = LLM_CONFIGS.get("ollama", {}).get(model.name, {})
            recommended_for = getattr(model_config, 'recommended_for', 'General use') if model_config else 'General use'
            
            models_info.append(ModelInfo(
                name=model.name,
                size=model.size,
                size_gb=round(model.size / (1024**3), 2),
                modified_at=model.modified_at,
                digest=model.digest,
                performance_score=model.performance_score,
                recommended_for=recommended_for
            ))
        
        # Get general recommendations
        recommendations = await ollama_manager.get_recommended_models("general")
        
        return OllamaStatusResponse(
            is_running=status.is_running,
            version=status.version,
            models_count=status.models_count,
            total_size=status.total_size,
            total_size_gb=round(status.total_size / (1024**3), 2),
            available_models=models_info,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Ollama status: {str(e)}")

@router.get("/health")
async def health_check():
    """Quick health check for Ollama server"""
    try:
        is_healthy = await ollama_manager.health_check()
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "is_running": is_healthy
        }
    except Exception as e:
        return {
            "status": "error",
            "is_running": False,
            "error": str(e)
        }

@router.get("/models")
@check_ollama_enabled()
async def list_models():
    """List all available models with detailed information"""
    try:
        models = await ollama_manager.list_models_detailed()
        
        models_info = []
        for model in models:
            model_config = LLM_CONFIGS.get("ollama", {}).get(model.name, {})
            recommended_for = getattr(model_config, 'recommended_for', 'General use') if model_config else 'General use'
            
            models_info.append({
                "name": model.name,
                "size": model.size,
                "size_gb": round(model.size / (1024**3), 2),
                "modified_at": model.modified_at,
                "digest": model.digest,
                "performance_score": model.performance_score,
                "recommended_for": recommended_for,
                "config": model_config.__dict__ if model_config else None
            })
        
        return {"models": models_info}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    try:
        model_info = await ollama_manager.get_model_info(model_name)
        
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        model_config = LLM_CONFIGS.get("ollama", {}).get(model_name, {})
        
        return {
            "name": model_info.name,
            "size": model_info.size,
            "size_gb": round(model_info.size / (1024**3), 2),
            "modified_at": model_info.modified_at,
            "digest": model_info.digest,
            "details": model_info.details,
            "performance_score": model_info.performance_score,
            "config": model_config.__dict__ if model_config else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.post("/models/pull")
async def pull_model(request: ModelPullRequest, background_tasks: BackgroundTasks):
    """Pull/download a model"""
    try:
        # Check if model is already available
        if await ollama_manager.is_model_available(request.model_name):
            return {
                "status": "already_available",
                "message": f"Model {request.model_name} is already available"
            }
        
        # Start pull in background
        async def pull_task():
            try:
                success = await ollama_manager.pull_model(request.model_name)
                if success:
                    print(f"Successfully pulled model {request.model_name}")
                else:
                    print(f"Failed to pull model {request.model_name}")
            except Exception as e:
                print(f"Error pulling model {request.model_name}: {e}")
        
        background_tasks.add_task(pull_task)
        
        return {
            "status": "started",
            "message": f"Started pulling model {request.model_name}. Check status for progress."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start model pull: {str(e)}")

@router.delete("/models/{model_name}")
async def remove_model(model_name: str):
    """Remove a model"""
    try:
        # Check if model exists
        if not await ollama_manager.is_model_available(model_name):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        success = await ollama_manager.remove_model(model_name)
        
        if success:
            return {
                "status": "success",
                "message": f"Model {model_name} removed successfully"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to remove model {model_name}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove model: {str(e)}")

@router.post("/chat", response_model=ChatResponse)
async def chat_with_model(request: ChatRequest):
    """Chat with a specific Ollama model"""
    try:
        import time
        from app.services.llm_factory import estimate_tokens
        
        start_time = time.time()
        
        # Ensure model is available
        if not await ensure_model_available(request.model):
            raise HTTPException(
                status_code=404, 
                detail=f"Model {request.model} is not available and could not be pulled"
            )
        
        # Create Ollama chat instance
        ollama_chat = OllamaChat(model=request.model)
        
        # Get response
        response = await ollama_chat.chat(request.prompt)
        
        response_time = time.time() - start_time
        tokens_estimated = estimate_tokens(request.prompt)
        
        # Track performance
        ollama_manager.track_model_performance(request.model, response_time, True)
        
        return ChatResponse(
            response=response,
            model_used=request.model,
            response_time=response_time,
            tokens_estimated=tokens_estimated
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Track failed performance
        ollama_manager.track_model_performance(request.model, 0, False)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@router.get("/recommendations/{use_case}")
async def get_recommendations(use_case: str):
    """Get model recommendations for a specific use case"""
    try:
        recommendations = await ollama_manager.get_recommended_models(use_case)
        
        # Check which recommended models are available
        available_recommendations = []
        for model in recommendations:
            is_available = await ollama_manager.is_model_available(model)
            model_config = LLM_CONFIGS.get("ollama", {}).get(model, {})
            
            available_recommendations.append({
                "model": model,
                "is_available": is_available,
                "recommended_for": getattr(model_config, 'recommended_for', 'General use') if model_config else 'General use',
                "config": model_config.__dict__ if model_config else None
            })
        
        return {
            "use_case": use_case,
            "recommendations": available_recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@router.post("/setup")
async def setup_for_first_use():
    """Set up Ollama with recommended models for first-time use"""
    try:
        success = await setup_ollama_for_first_use()
        
        if success:
            status = await get_ollama_status()
            return {
                "status": "success",
                "message": "Ollama setup completed successfully",
                "models_count": status.models_count
            }
        else:
            return {
                "status": "failed",
                "message": "Failed to set up Ollama. Please check if Ollama server is running."
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Setup failed: {str(e)}")

@router.get("/performance")
async def get_performance_report():
    """Get performance report for all models"""
    try:
        report = await ollama_manager.get_performance_report()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance report: {str(e)}")

@router.get("/optimize")
async def get_optimization_recommendations():
    """Get optimization recommendations based on usage and performance"""
    try:
        optimization = await ollama_manager.optimize_models()
        return optimization
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get optimization recommendations: {str(e)}")

@router.post("/auto-setup/{use_case}")
async def auto_setup_for_use_case(use_case: str):
    """Automatically set up the best model for a specific use case"""
    try:
        model = await ollama_manager.auto_setup_recommended_model(use_case)
        
        if model:
            return {
                "status": "success",
                "message": f"Successfully set up {model} for {use_case}",
                "model": model
            }
        else:
            return {
                "status": "failed",
                "message": f"Failed to set up any model for {use_case}"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-setup failed: {str(e)}")

# Provider availability endpoint for frontend
@router.get("/provider-status")
async def get_provider_status():
    """Get Ollama provider status for frontend integration"""
    try:
        is_running = await ollama_manager.health_check()
        status = await get_ollama_status() if is_running else None
        
        return {
            "provider": "ollama",
            "available": is_running,
            "status": "running" if is_running else "not_running",
            "models_count": status.models_count if status else 0,
            "version": status.version if status else None,
            "features": {
                "local_processing": True,
                "no_api_costs": True,
                "privacy_focused": True,
                "offline_capable": True
            }
        }
        
    except Exception as e:
        return {
            "provider": "ollama",
            "available": False,
            "status": "error",
            "error": str(e),
            "features": {
                "local_processing": True,
                "no_api_costs": True,
                "privacy_focused": True,
                "offline_capable": True
            }
        }

@router.get("/all-providers")
async def get_all_providers():
    """Get all AI providers and their available models for frontend dropdown"""
    try:
        from app.services.llm_factory import health_check
        from app.config import settings
        
        # Get health status of all providers
        health_status = await health_check()
        
        providers = []
        
        # Ollama Provider - Only include if enabled
        if settings.ollama_enabled:
            ollama_models = []
            try:
                # Get actual Ollama models
                status = await get_ollama_status()
                if status and status.available_models:
                    for model in status.available_models:
                        ollama_models.append({
                            "name": model.name,
                            "displayName": model.name.replace(":", " ").title(),
                            "description": f"Local model - {round(model.size / (1024**3), 1)}GB",
                            "isAvailable": True,
                            "isLocal": True,
                            "size": model.size
                        })
            except Exception as e:
                print(f"Failed to get Ollama models: {e}")
                # Add default models
                default_ollama_models = [
                    {"name": "llama3.2:latest", "displayName": "Llama 3.2 Latest", "description": "Latest Llama 3.2 model"},
                    {"name": "llama3.2:3b", "displayName": "Llama 3.2 3B", "description": "Fast 3B parameter model"},
                    {"name": "llama3.2:1b", "displayName": "Llama 3.2 1B", "description": "Ultra-fast 1B parameter model"},
                    {"name": "phi3:mini", "displayName": "Phi-3 Mini", "description": "Microsoft's efficient small model"},
                    {"name": "qwen2.5:3b", "displayName": "Qwen 2.5 3B", "description": "Alibaba's multilingual model"}
                ]
                for model in default_ollama_models:
                    ollama_models.append({
                        **model,
                        "isAvailable": True,
                        "isLocal": True,
                        "size": 0
                    })
            
            if not ollama_models:
                # Ensure at least one default model
                ollama_models.append({
                    "name": "llama3.2:latest",
                    "displayName": "Llama 3.2 Latest",
                    "description": "Default high-performance local model",
                    "isAvailable": True,
                    "isLocal": True,
                    "size": 0
                })
            
            providers.append({
                "name": "ollama",
                "displayName": "Ollama (Local AI)",
                "description": "Free local AI models - No API costs, complete privacy",
                "isAvailable": health_status.get("providers", {}).get("ollama", {}).get("status") == "healthy",
                "isDefault": not (settings.bedrock_enabled and settings.aws_bedrock_access_key_id),  # Default only if Bedrock not enabled
                "features": ["Local Processing", "No API Costs", "Privacy Focused", "Offline Capable"],
                "models": ollama_models
            })
        
        # OpenAI Provider
        openai_models = [
            {
                "name": "gpt-4o-mini",
                "displayName": "GPT-4o Mini",
                "description": "Fast and cost-effective GPT-4 model",
                "isAvailable": bool(settings.openai_api_key),
                "isLocal": False
            },
            {
                "name": "gpt-3.5-turbo",
                "displayName": "GPT-3.5 Turbo",
                "description": "Fast and reliable ChatGPT model",
                "isAvailable": bool(settings.openai_api_key),
                "isLocal": False
            },
            {
                "name": "gpt-4",
                "displayName": "GPT-4",
                "description": "Most capable OpenAI model",
                "isAvailable": bool(settings.openai_api_key),
                "isLocal": False
            }
        ]
        
        providers.append({
            "name": "openai",
            "displayName": "OpenAI",
            "description": "Industry-leading AI models from OpenAI",
            "isAvailable": bool(settings.openai_api_key),
            "isDefault": False,
            "features": ["High Quality", "Fast Response", "Advanced Reasoning"],
            "models": openai_models
        })
        
        # Groq Provider
        groq_models = [
            {
                "name": "llama3-8b-8192",
                "displayName": "Llama 3 8B",
                "description": "Fast Llama 3 model with 8K context",
                "isAvailable": bool(settings.groq_api_key),
                "isLocal": False
            },
            {
                "name": "mixtral-8x7b-32768",
                "displayName": "Mixtral 8x7B",
                "description": "High-performance mixture of experts model",
                "isAvailable": bool(settings.groq_api_key),
                "isLocal": False
            },
            {
                "name": "gemma-7b-it",
                "displayName": "Gemma 7B",
                "description": "Google's efficient instruction-tuned model",
                "isAvailable": bool(settings.groq_api_key),
                "isLocal": False
            }
        ]
        
        providers.append({
            "name": "groq",
            "displayName": "Groq",
            "description": "Ultra-fast AI inference with specialized hardware",
            "isAvailable": bool(settings.groq_api_key),
            "isDefault": False,
            "features": ["Ultra Fast", "Low Latency", "High Throughput"],
            "models": groq_models
        })
        
        # Gemini Provider
        gemini_models = [
            {
                "name": "gemini-1.5-flash",
                "displayName": "Gemini 1.5 Flash",
                "description": "Fast and efficient Gemini model",
                "isAvailable": bool(settings.gemini_api_key),
                "isLocal": False
            },
            {
                "name": "gemini-1.5-pro",
                "displayName": "Gemini 1.5 Pro",
                "description": "Most capable Gemini model",
                "isAvailable": bool(settings.gemini_api_key),
                "isLocal": False
            }
        ]
        
        providers.append({
            "name": "gemini",
            "displayName": "Google Gemini",
            "description": "Google's advanced multimodal AI models",
            "isAvailable": bool(settings.gemini_api_key),
            "isDefault": False,
            "features": ["Multimodal", "Large Context", "Advanced Reasoning"],
            "models": gemini_models
        })
        
        # AWS Bedrock Provider (Priority 0 - Primary)
        bedrock_models = [
            {
                "name": "anthropic.claude-3-haiku-20240307-v1:0",
                "displayName": "Claude 3 Haiku",
                "description": "Ultra-fast, cost-effective Claude 3 model - 200K context",
                "isAvailable": bool(settings.bedrock_enabled and settings.aws_bedrock_access_key_id),
                "isLocal": False
            }
        ]
        
        # Insert Bedrock as the FIRST provider (highest priority)
        providers.insert(0, {
            "name": "bedrock",
            "displayName": "AWS Bedrock (Claude 3 Haiku)",
            "description": "Enterprise-grade Claude 3 Haiku - 3-5x faster than local models",
            "isAvailable": bool(settings.bedrock_enabled and settings.aws_bedrock_access_key_id),
            "isDefault": settings.bedrock_enabled,  # Default if enabled
            "features": ["Ultra Fast", "200K Context", "Enterprise Grade", "99.9% Uptime"],
            "models": bedrock_models
        })
        
        # DeepSeek Provider
        deepseek_models = [
            {
                "name": "deepseek-chat",
                "displayName": "DeepSeek Chat",
                "description": "High-performance chat model from DeepSeek",
                "isAvailable": bool(settings.deepseek_api_key),
                "isLocal": False
            },
            {
                "name": "deepseek-coder",
                "displayName": "DeepSeek Coder",
                "description": "Specialized coding model from DeepSeek",
                "isAvailable": bool(settings.deepseek_api_key),
                "isLocal": False
            }
        ]
        
        providers.append({
            "name": "deepseek",
            "displayName": "DeepSeek",
            "description": "Cost-effective high-performance AI models",
            "isAvailable": bool(settings.deepseek_api_key),
            "isDefault": False,
            "features": ["Cost Effective", "High Performance", "Coding Specialized"],
            "models": deepseek_models
        })
        
        # Determine default provider based on availability and configuration
        default_provider = "ollama"  # Fallback default
        
        # Prioritize Bedrock if enabled and available
        if settings.bedrock_enabled and settings.aws_bedrock_access_key_id:
            default_provider = "bedrock"
        
        return {
            "providers": providers,
            "defaultProvider": default_provider,
            "healthStatus": health_status
        }
        
    except Exception as e:
        print(f"Error getting all providers: {e}")
        # Return minimal fallback
        return {
            "providers": [
                {
                    "name": "ollama",
                    "displayName": "Ollama (Local AI)",
                    "description": "Free local AI models",
                    "isAvailable": True,
                    "isDefault": True,
                    "features": ["Local Processing", "No API Costs"],
                    "models": [
                        {
                            "name": "llama3.2:latest",
                            "displayName": "Llama 3.2 Latest",
                            "description": "Default local model",
                            "isAvailable": True,
                            "isLocal": True
                        }
                    ]
                }
            ],
            "defaultProvider": "ollama",
            "error": str(e)
        }
