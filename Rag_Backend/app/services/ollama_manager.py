"""
Ollama Management Service

This service provides comprehensive management for Ollama models including:
- Health monitoring and status checks
- Model lifecycle management (download, remove, update)
- Performance monitoring and optimization
- Automatic fallback and recovery strategies
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from app.config import settings
from app.config_llm import LLM_CONFIGS

@dataclass
class ModelInfo:
    """Information about an Ollama model"""
    name: str
    size: int
    modified_at: str
    digest: str
    details: Dict
    is_running: bool = False
    last_used: Optional[float] = None
    performance_score: float = 0.0

@dataclass
class OllamaStatus:
    """Overall status of Ollama server"""
    is_running: bool
    version: str
    models_count: int
    total_size: int
    available_models: List[ModelInfo]
    system_info: Dict
    last_check: float

class OllamaManager:
    """Comprehensive Ollama management service"""
    
    def __init__(self):
        self.base_url = settings.ollama_host.rstrip('/')
        self.timeout = settings.ollama_timeout
        self._session = None
        self._status_cache = None
        self._cache_ttl = 30  # Cache status for 30 seconds
        self._model_performance = {}  # Track model performance metrics
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _close_session(self):
        """Close aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> bool:
        """Quick health check for Ollama server"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags", timeout=5) as response:
                return response.status == 200
        except Exception:
            return False

    async def get_server_info(self) -> Dict:
        """Get detailed server information"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/version") as response:
                if response.status == 200:
                    return await response.json()
                return {}
        except Exception:
            return {}

    async def get_status(self, force_refresh: bool = False) -> OllamaStatus:
        """Get comprehensive Ollama status with caching"""
        current_time = time.time()
        
        # Return cached status if still valid
        if (not force_refresh and 
            self._status_cache and 
            current_time - self._status_cache.last_check < self._cache_ttl):
            return self._status_cache
        
        # Check if server is running
        is_running = await self.health_check()
        
        if not is_running:
            self._status_cache = OllamaStatus(
                is_running=False,
                version="",
                models_count=0,
                total_size=0,
                available_models=[],
                system_info={},
                last_check=current_time
            )
            return self._status_cache
        
        # Get server info and models
        server_info = await self.get_server_info()
        models = await self.list_models_detailed()
        
        total_size = sum(model.size for model in models)
        
        self._status_cache = OllamaStatus(
            is_running=True,
            version=server_info.get("version", "unknown"),
            models_count=len(models),
            total_size=total_size,
            available_models=models,
            system_info=server_info,
            last_check=current_time
        )
        
        return self._status_cache

    async def list_models_detailed(self) -> List[ModelInfo]:
        """Get detailed information about all available models"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                models = []
                
                for model_data in data.get("models", []):
                    model_info = ModelInfo(
                        name=model_data.get("name", ""),
                        size=model_data.get("size", 0),
                        modified_at=model_data.get("modified_at", ""),
                        digest=model_data.get("digest", ""),
                        details=model_data.get("details", {}),
                        performance_score=self._model_performance.get(
                            model_data.get("name", ""), 0.0
                        )
                    )
                    models.append(model_info)
                
                return models
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get detailed information about a specific model"""
        try:
            session = await self._get_session()
            payload = {"name": model_name}
            
            async with session.post(f"{self.base_url}/api/show", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return ModelInfo(
                        name=model_name,
                        size=data.get("size", 0),
                        modified_at=data.get("modified_at", ""),
                        digest=data.get("digest", ""),
                        details=data.get("details", {}),
                        performance_score=self._model_performance.get(model_name, 0.0)
                    )
                return None
        except Exception as e:
            print(f"Error getting model info for {model_name}: {e}")
            return None

    async def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available"""
        models = await self.list_models_detailed()
        return any(model.name == model_name for model in models)

    async def pull_model(self, model_name: str, progress_callback=None) -> bool:
        """Pull/download a model with progress tracking"""
        try:
            session = await self._get_session()
            payload = {"name": model_name}
            
            print(f"Ollama: Starting pull for model {model_name}...")
            
            async with session.post(
                f"{self.base_url}/api/pull", 
                json=payload
            ) as response:
                if response.status == 200:
                    total_size = 0
                    downloaded = 0
                    
                    async for line in response.content:
                        if line:
                            try:
                                progress = json.loads(line.decode())
                                status = progress.get("status", "")
                                
                                if "total" in progress and "completed" in progress:
                                    total_size = progress["total"]
                                    downloaded = progress["completed"]
                                    percentage = (downloaded / total_size * 100) if total_size > 0 else 0
                                    
                                    if progress_callback:
                                        await progress_callback(status, percentage, downloaded, total_size)
                                    else:
                                        print(f"Ollama pull: {status} - {percentage:.1f}%")
                                else:
                                    if progress_callback:
                                        await progress_callback(status, 0, 0, 0)
                                    else:
                                        print(f"Ollama pull: {status}")
                                        
                            except json.JSONDecodeError:
                                continue
                    
                    # Invalidate cache after successful pull
                    self._status_cache = None
                    print(f"Ollama: Successfully pulled model {model_name}")
                    return True
                else:
                    print(f"Failed to pull model {model_name}: {response.status}")
                    return False
                    
        except Exception as e:
            print(f"Error pulling model {model_name}: {e}")
            return False

    async def remove_model(self, model_name: str) -> bool:
        """Remove a model from Ollama"""
        try:
            session = await self._get_session()
            payload = {"name": model_name}
            
            async with session.delete(f"{self.base_url}/api/delete", json=payload) as response:
                if response.status == 200:
                    # Invalidate cache after successful removal
                    self._status_cache = None
                    # Remove from performance tracking
                    self._model_performance.pop(model_name, None)
                    print(f"Ollama: Successfully removed model {model_name}")
                    return True
                else:
                    print(f"Failed to remove model {model_name}: {response.status}")
                    return False
                    
        except Exception as e:
            print(f"Error removing model {model_name}: {e}")
            return False

    async def get_recommended_models(self, use_case: str = "general") -> List[str]:
        """Get recommended models based on use case and system capabilities"""
        recommendations = {
            "general": ["deepseek-r1:8b", "llama3.1:8b", "mistral:7b"],
            "coding": ["deepseek-r1:8b", "codellama:13b", "qwen2:7b"],
            "large_context": ["qwen2:7b", "deepseek-r1:8b", "codellama:13b"],
            "fast_response": ["phi3:mini", "mistral:7b", "deepseek-r1:8b"],
            "high_quality": ["deepseek-r1:8b", "llama3.1:70b", "llama3.1:8b"],
            "reasoning": ["deepseek-r1:8b", "llama3.1:8b", "qwen2:7b"]
        }
        
        return recommendations.get(use_case, recommendations["general"])

    async def auto_setup_recommended_model(self, use_case: str = "general") -> Optional[str]:
        """Automatically set up the best available model for a use case"""
        recommended = await self.get_recommended_models(use_case)
        
        # Check if any recommended models are already available
        for model in recommended:
            if await self.is_model_available(model):
                print(f"Ollama: Using existing model {model} for {use_case}")
                return model
        
        # Try to pull the first recommended model
        for model in recommended:
            print(f"Ollama: Attempting to pull recommended model {model} for {use_case}")
            if await self.pull_model(model):
                return model
        
        print(f"Ollama: Failed to set up any recommended model for {use_case}")
        return None

    def track_model_performance(self, model_name: str, response_time: float, success: bool):
        """Track model performance metrics"""
        if model_name not in self._model_performance:
            self._model_performance[model_name] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_response_time": 0.0,
                "average_response_time": 0.0,
                "success_rate": 0.0,
                "score": 0.0
            }
        
        metrics = self._model_performance[model_name]
        metrics["total_requests"] += 1
        
        if success:
            metrics["successful_requests"] += 1
            metrics["total_response_time"] += response_time
        
        # Update calculated metrics
        if metrics["total_requests"] > 0:
            metrics["success_rate"] = metrics["successful_requests"] / metrics["total_requests"]
            
        if metrics["successful_requests"] > 0:
            metrics["average_response_time"] = metrics["total_response_time"] / metrics["successful_requests"]
        
        # Calculate performance score (0-100)
        # Factors: success rate (60%), response time (40%)
        success_score = metrics["success_rate"] * 60
        
        # Response time score (faster = better, cap at 30 seconds)
        time_score = max(0, 40 - (metrics["average_response_time"] / 30 * 40))
        
        metrics["score"] = success_score + time_score

    async def get_performance_report(self) -> Dict:
        """Get performance report for all models"""
        return {
            "models": self._model_performance,
            "summary": {
                "total_models_tracked": len(self._model_performance),
                "best_performing": max(
                    self._model_performance.items(),
                    key=lambda x: x[1]["score"],
                    default=(None, {"score": 0})
                )[0] if self._model_performance else None
            }
        }

    async def optimize_models(self) -> Dict:
        """Optimize model setup based on performance and usage"""
        status = await self.get_status()
        performance_report = await self.get_performance_report()
        
        recommendations = []
        
        # Check for unused models (low performance score)
        for model in status.available_models:
            perf_data = self._model_performance.get(model.name, {})
            if perf_data.get("score", 0) < 20 and perf_data.get("total_requests", 0) > 10:
                recommendations.append({
                    "action": "consider_removing",
                    "model": model.name,
                    "reason": "Low performance score",
                    "score": perf_data.get("score", 0)
                })
        
        # Check for missing recommended models
        general_recommended = await self.get_recommended_models("general")
        available_names = [model.name for model in status.available_models]
        
        for rec_model in general_recommended[:2]:  # Check top 2 recommendations
            if rec_model not in available_names:
                recommendations.append({
                    "action": "consider_pulling",
                    "model": rec_model,
                    "reason": "Recommended model not available"
                })
        
        return {
            "current_status": {
                "models_count": status.models_count,
                "total_size_gb": round(status.total_size / (1024**3), 2),
                "is_running": status.is_running
            },
            "performance_summary": performance_report["summary"],
            "recommendations": recommendations
        }

    async def cleanup(self):
        """Cleanup resources"""
        await self._close_session()

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, '_session') and self._session and not self._session.closed:
            try:
                asyncio.create_task(self._close_session())
            except:
                pass

# Global instance for easy access
ollama_manager = OllamaManager()

# Utility functions for easy access
async def get_ollama_status() -> OllamaStatus:
    """Get current Ollama status"""
    return await ollama_manager.get_status()

async def ensure_model_available(model_name: str) -> bool:
    """Ensure a model is available, pull if necessary"""
    if await ollama_manager.is_model_available(model_name):
        return True
    
    print(f"Model {model_name} not found, attempting to pull...")
    return await ollama_manager.pull_model(model_name)

async def get_best_available_model(use_case: str = "general") -> Optional[str]:
    """Get the best available model for a use case"""
    recommended = await ollama_manager.get_recommended_models(use_case)
    
    for model in recommended:
        if await ollama_manager.is_model_available(model):
            return model
    
    return None

async def setup_ollama_for_first_use() -> bool:
    """Set up Ollama with recommended models for first-time use"""
    if not await ollama_manager.health_check():
        print("Ollama server is not running. Please start Ollama first.")
        return False
    
    status = await ollama_manager.get_status()
    
    if status.models_count == 0:
        print("No models found. Setting up recommended models...")
        
        # Try to set up a basic model for general use
        model = await ollama_manager.auto_setup_recommended_model("general")
        if model:
            print(f"Successfully set up {model} for general use")
            return True
        else:
            print("Failed to set up any models")
            return False
    
    print(f"Ollama is ready with {status.models_count} models")
    return True
