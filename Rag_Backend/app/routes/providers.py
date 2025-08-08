# app/routes/providers.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from app.deps import get_current_user
from app.database import User
from app.services.llm_factory import config, health_monitor, has_required_credentials
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/providers", tags=["providers"])

# Response Models
class ProviderModel(BaseModel):
    name: str
    display_name: str
    models: List[str]
    enabled: bool
    priority: int
    status: str
    health_score: Optional[float] = None
    can_execute: bool
    description: str

class ProviderResponse(BaseModel):
    providers: List[ProviderModel]
    total_providers: int
    available_providers: int
    default_provider: str

class ProviderHealthResponse(BaseModel):
    provider: str
    status: str
    health_score: float
    response_time_avg: Optional[float] = None
    last_success: Optional[str] = None
    last_failure: Optional[str] = None
    can_execute: bool

# Helper functions
def get_provider_display_name(provider_name: str) -> str:
    """Get user-friendly display name for provider"""
    display_names = {
        "bedrock": "AWS Bedrock (Claude 3 Haiku)",
        "groq": "Groq (Llama & Mixtral)",
        "openai": "OpenAI (GPT Models)",
        "deepseek": "DeepSeek (DeepSeek Chat)",
        "gemini": "Google Gemini (Gemini 1.5)"
    }
    return display_names.get(provider_name, provider_name.title())

def get_provider_description(provider_name: str) -> str:
    """Get description for provider"""
    descriptions = {
        "bedrock": "AWS Bedrock with Claude 3 Haiku - Fast, accurate, and cost-effective AI assistant",
        "groq": "Ultra-fast inference with Llama and Mixtral models - Optimized for speed",
        "openai": "OpenAI's GPT models - Industry-leading language understanding",
        "deepseek": "DeepSeek's advanced chat model - Specialized for reasoning tasks",
        "gemini": "Google's Gemini models - Multimodal AI with strong reasoning capabilities"
    }
    return descriptions.get(provider_name, f"{provider_name.title()} AI provider")

@router.get("/all", response_model=ProviderResponse)
async def get_all_providers(
    current_user: User = Depends(get_current_user)
):
    """Get all available LLM providers with their status and models"""
    try:
        providers = []
        available_count = 0
        
        # Get available providers from health monitor
        available_provider_names = health_monitor.get_available_providers()
        
        for provider_name, provider_config in config.providers.items():
            # Get circuit breaker status
            circuit_breaker = health_monitor.get_circuit_breaker(provider_name)
            circuit_status = circuit_breaker.get_status()
            
            # Check if provider has credentials
            has_credentials = has_required_credentials(provider_name)
            
            # Determine overall status
            if not provider_config.enabled:
                status = "disabled"
            elif not has_credentials:
                status = "no_credentials"
            elif circuit_status["can_execute"]:
                status = "available"
                available_count += 1
            else:
                status = circuit_status["state"]
            
            # Get health score
            health_score = health_monitor.health_scores.get(provider_name, 0.5)
            
            provider_model = ProviderModel(
                name=provider_name,
                display_name=get_provider_display_name(provider_name),
                models=provider_config.models,
                enabled=provider_config.enabled and has_credentials,
                priority=provider_config.priority,
                status=status,
                health_score=health_score,
                can_execute=circuit_status["can_execute"] and has_credentials,
                description=get_provider_description(provider_name)
            )
            
            providers.append(provider_model)
        
        # Sort by priority
        providers.sort(key=lambda x: x.priority)
        
        # Determine default provider (first available one)
        default_provider = "bedrock"  # Default fallback
        for provider in providers:
            if provider.can_execute:
                default_provider = provider.name
                break
        
        return ProviderResponse(
            providers=providers,
            total_providers=len(providers),
            available_providers=available_count,
            default_provider=default_provider
        )
        
    except Exception as e:
        logger.error(f"Error fetching providers for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch providers")

@router.get("/health", response_model=List[ProviderHealthResponse])
async def get_providers_health(
    current_user: User = Depends(get_current_user)
):
    """Get detailed health information for all providers"""
    try:
        health_responses = []
        
        for provider_name, provider_config in config.providers.items():
            if not provider_config.enabled:
                continue
                
            circuit_breaker = health_monitor.get_circuit_breaker(provider_name)
            circuit_status = circuit_breaker.get_status()
            
            # Get response time average
            response_times = health_monitor.response_times.get(provider_name, [])
            avg_response_time = sum(response_times) / len(response_times) if response_times else None
            
            health_response = ProviderHealthResponse(
                provider=provider_name,
                status=circuit_status["state"],
                health_score=health_monitor.health_scores.get(provider_name, 0.5),
                response_time_avg=avg_response_time,
                last_success=circuit_status["last_success"],
                last_failure=circuit_status["last_failure"],
                can_execute=circuit_status["can_execute"]
            )
            
            health_responses.append(health_response)
        
        return health_responses
        
    except Exception as e:
        logger.error(f"Error fetching provider health for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch provider health")

@router.get("/available", response_model=List[str])
async def get_available_providers(
    current_user: User = Depends(get_current_user)
):
    """Get list of currently available provider names"""
    try:
        available_providers = health_monitor.get_available_providers()
        
        # Filter to only include providers with credentials
        available_with_credentials = [
            provider for provider in available_providers 
            if has_required_credentials(provider)
        ]
        
        return available_with_credentials
        
    except Exception as e:
        logger.error(f"Error fetching available providers for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch available providers")

@router.get("/models", response_model=Dict[str, List[str]])
async def get_provider_models(
    current_user: User = Depends(get_current_user)
):
    """Get all models grouped by provider"""
    try:
        provider_models = {}
        
        for provider_name, provider_config in config.providers.items():
            if provider_config.enabled and has_required_credentials(provider_name):
                provider_models[provider_name] = provider_config.models
        
        return provider_models
        
    except Exception as e:
        logger.error(f"Error fetching provider models for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch provider models")

@router.get("/{provider_name}", response_model=ProviderModel)
async def get_provider_details(
    provider_name: str,
    current_user: User = Depends(get_current_user)
):
    """Get detailed information about a specific provider"""
    try:
        provider_config = config.providers.get(provider_name)
        if not provider_config:
            raise HTTPException(status_code=404, detail="Provider not found")
        
        # Get circuit breaker status
        circuit_breaker = health_monitor.get_circuit_breaker(provider_name)
        circuit_status = circuit_breaker.get_status()
        
        # Check credentials
        has_credentials = has_required_credentials(provider_name)
        
        # Determine status
        if not provider_config.enabled:
            status = "disabled"
        elif not has_credentials:
            status = "no_credentials"
        elif circuit_status["can_execute"]:
            status = "available"
        else:
            status = circuit_status["state"]
        
        provider_model = ProviderModel(
            name=provider_name,
            display_name=get_provider_display_name(provider_name),
            models=provider_config.models,
            enabled=provider_config.enabled and has_credentials,
            priority=provider_config.priority,
            status=status,
            health_score=health_monitor.health_scores.get(provider_name, 0.5),
            can_execute=circuit_status["can_execute"] and has_credentials,
            description=get_provider_description(provider_name)
        )
        
        return provider_model
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching provider {provider_name} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch provider details")

@router.post("/{provider_name}/test")
async def test_provider(
    provider_name: str,
    current_user: User = Depends(get_current_user)
):
    """Test a specific provider with a simple request"""
    try:
        provider_config = config.providers.get(provider_name)
        if not provider_config:
            raise HTTPException(status_code=404, detail="Provider not found")
        
        if not provider_config.enabled:
            raise HTTPException(status_code=400, detail="Provider is disabled")
        
        if not has_required_credentials(provider_name):
            raise HTTPException(status_code=400, detail="Provider credentials not configured")
        
        # Import and test the provider
        from app.services.llm_factory import get_llm
        import time
        
        start_time = time.time()
        llm = get_llm(provider=provider_name)
        
        # Simple test prompt
        response = await llm.chat(
            prompt="Hello, please respond with 'Test successful'",
            max_tokens=50
        )
        
        duration = time.time() - start_time
        
        return {
            "success": True,
            "provider": provider_name,
            "response": response,
            "response_time": round(duration, 2),
            "message": f"Provider {provider_name} test completed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing provider {provider_name} for user {current_user.id}: {e}")
        return {
            "success": False,
            "provider": provider_name,
            "error": str(e),
            "message": f"Provider {provider_name} test failed"
        }
