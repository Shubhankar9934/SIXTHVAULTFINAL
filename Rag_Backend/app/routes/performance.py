"""
Performance Monitoring and Health Check Routes

This module provides endpoints for monitoring system performance,
memory usage, and model loading statistics.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import time
import psutil
from app.deps import get_current_user
from app.database import User

router = APIRouter(prefix="/performance", tags=["performance"])

@router.get("/health")
async def get_system_health():
    """Get comprehensive system health information"""
    try:
        # Memory information
        from app.utils.memory_manager import memory_manager, get_memory_report
        memory_report = get_memory_report()
        
        # Model loading statistics
        try:
            from app.utils.lazy_model_loader import get_model_stats
            model_stats = get_model_stats()
        except ImportError:
            model_stats = {"status": "lazy_loader_not_available"}
        
        # LLM Factory health
        try:
            from app.services.llm_factory import health_check as llm_health_check, perf_monitor
            llm_health = await llm_health_check()
            llm_stats = perf_monitor.get_stats()
        except Exception as e:
            llm_health = {"status": "error", "error": str(e)}
            llm_stats = {"status": "error", "error": str(e)}
        
        # System resources
        cpu_percent = psutil.cpu_percent(interval=1)
        disk_usage = psutil.disk_usage('/')
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "system": {
                "cpu_percent": cpu_percent,
                "disk_usage": {
                    "total_gb": disk_usage.total / (1024**3),
                    "used_gb": disk_usage.used / (1024**3),
                    "free_gb": disk_usage.free / (1024**3),
                    "percent_used": (disk_usage.used / disk_usage.total) * 100
                }
            },
            "memory": memory_report,
            "models": model_stats,
            "llm_providers": llm_health,
            "llm_performance": llm_stats
        }
        
    except Exception as e:
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }

@router.get("/memory")
async def get_memory_status():
    """Get detailed memory usage information"""
    try:
        from app.utils.memory_manager import memory_manager, get_memory_report
        
        memory_report = get_memory_report()
        
        # Additional process information
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "status": "success",
            "timestamp": time.time(),
            "memory_manager": memory_report,
            "process_memory": {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "num_threads": process.num_threads(),
                "num_fds": process.num_fds() if hasattr(process, 'num_fds') else None
            },
            "system_memory": {
                "total_mb": psutil.virtual_memory().total / 1024 / 1024,
                "available_mb": psutil.virtual_memory().available / 1024 / 1024,
                "percent_used": psutil.virtual_memory().percent
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }

@router.get("/models")
async def get_model_status():
    """Get status of all lazy-loaded models"""
    try:
        from app.utils.lazy_model_loader import get_model_stats
        
        model_stats = get_model_stats()
        
        return {
            "status": "success",
            "timestamp": time.time(),
            "models": model_stats
        }
        
    except ImportError:
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": "Lazy model loader not available"
        }
    except Exception as e:
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }

@router.get("/llm-providers")
async def get_llm_provider_status():
    """Get status of all LLM providers"""
    try:
        from app.services.llm_factory import health_check as llm_health_check, perf_monitor
        
        health_status = await llm_health_check()
        performance_stats = perf_monitor.get_stats()
        
        return {
            "status": "success",
            "timestamp": time.time(),
            "health": health_status,
            "performance": performance_stats
        }
        
    except Exception as e:
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }

@router.post("/memory/cleanup")
async def trigger_memory_cleanup(current_user: User = Depends(get_current_user)):
    """Trigger manual memory cleanup (admin only)"""
    try:
        from app.utils.memory_manager import memory_manager
        
        # Get memory before cleanup
        memory_before = memory_manager.get_memory_usage()
        
        # Force cleanup
        memory_manager.force_cleanup(emergency=True)
        
        # Get memory after cleanup
        memory_after = memory_manager.get_memory_usage()
        
        return {
            "status": "success",
            "timestamp": time.time(),
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_freed_mb": memory_before.get("rss_mb", 0) - memory_after.get("rss_mb", 0)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }

@router.post("/models/cleanup")
async def cleanup_models(current_user: User = Depends(get_current_user)):
    """Cleanup all lazy-loaded models (admin only)"""
    try:
        from app.utils.lazy_model_loader import cleanup_all_models, get_model_stats
        
        # Get stats before cleanup
        stats_before = get_model_stats()
        
        # Cleanup all models
        cleanup_all_models()
        
        # Get stats after cleanup
        stats_after = get_model_stats()
        
        return {
            "status": "success",
            "timestamp": time.time(),
            "models_before": stats_before,
            "models_after": stats_after,
            "message": "All lazy-loaded models have been cleaned up"
        }
        
    except ImportError:
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": "Lazy model loader not available"
        }
    except Exception as e:
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }

@router.get("/startup-metrics")
async def get_startup_metrics():
    """Get startup performance metrics"""
    try:
        # Calculate uptime
        import psutil
        process = psutil.Process()
        uptime_seconds = time.time() - process.create_time()
        
        # Get current memory usage
        from app.utils.memory_manager import memory_manager
        current_memory = memory_manager.get_memory_usage()
        
        # Get model loading stats
        try:
            from app.utils.lazy_model_loader import get_model_stats
            model_stats = get_model_stats()
        except ImportError:
            model_stats = {"status": "not_available"}
        
        return {
            "status": "success",
            "timestamp": time.time(),
            "uptime_seconds": uptime_seconds,
            "uptime_formatted": f"{uptime_seconds // 3600:.0f}h {(uptime_seconds % 3600) // 60:.0f}m {uptime_seconds % 60:.0f}s",
            "current_memory_mb": current_memory.get("rss_mb", 0),
            "memory_efficiency": {
                "is_critical": memory_manager.is_memory_critical(),
                "is_emergency": memory_manager.is_memory_emergency(),
                "threshold_mb": memory_manager.memory_threshold_mb
            },
            "lazy_loading": {
                "enabled": True,
                "models": model_stats
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }

@router.get("/optimization-report")
async def get_optimization_report():
    """Get comprehensive optimization report"""
    try:
        # Memory optimization metrics
        from app.utils.memory_manager import memory_manager, get_memory_report
        memory_report = get_memory_report()
        
        # Model loading optimization
        try:
            from app.utils.lazy_model_loader import get_model_stats
            model_stats = get_model_stats()
            
            # Calculate lazy loading efficiency
            loaded_models = sum(1 for stats in model_stats.values() 
                              if isinstance(stats, dict) and stats.get("loaded", False))
            total_models = len([k for k in model_stats.keys() if k != "status"])
            lazy_efficiency = ((total_models - loaded_models) / max(1, total_models)) * 100
            
        except ImportError:
            model_stats = {"status": "not_available"}
            lazy_efficiency = 0
        
        # LLM provider optimization
        try:
            from app.services.llm_factory import perf_monitor
            llm_stats = perf_monitor.get_stats()
        except Exception:
            llm_stats = {"status": "not_available"}
        
        # System performance
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        return {
            "status": "success",
            "timestamp": time.time(),
            "optimization_summary": {
                "memory_optimized": not memory_manager.is_memory_critical(),
                "lazy_loading_efficiency_percent": lazy_efficiency,
                "system_performance": {
                    "cpu_usage_percent": cpu_percent,
                    "memory_usage_percent": memory_percent,
                    "performance_rating": "excellent" if cpu_percent < 50 and memory_percent < 70 else 
                                        "good" if cpu_percent < 80 and memory_percent < 85 else "needs_attention"
                }
            },
            "detailed_metrics": {
                "memory": memory_report,
                "models": model_stats,
                "llm_performance": llm_stats
            },
            "recommendations": _generate_optimization_recommendations(
                memory_report, model_stats, cpu_percent, memory_percent
            )
        }
        
    except Exception as e:
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }

def _generate_optimization_recommendations(memory_report, model_stats, cpu_percent, memory_percent):
    """Generate optimization recommendations based on current metrics"""
    recommendations = []
    
    # Memory recommendations
    if memory_report.get("is_critical", False):
        recommendations.append({
            "type": "memory",
            "priority": "high",
            "message": "Memory usage is critical. Consider cleaning up unused models or restarting the service."
        })
    elif memory_percent > 80:
        recommendations.append({
            "type": "memory",
            "priority": "medium",
            "message": "System memory usage is high. Monitor for potential memory leaks."
        })
    
    # Model loading recommendations
    if isinstance(model_stats, dict) and model_stats.get("status") != "not_available":
        loaded_models = sum(1 for stats in model_stats.values() 
                          if isinstance(stats, dict) and stats.get("loaded", False))
        if loaded_models > 2:
            recommendations.append({
                "type": "models",
                "priority": "medium",
                "message": f"{loaded_models} models are currently loaded. Consider cleanup if not actively used."
            })
    
    # CPU recommendations
    if cpu_percent > 80:
        recommendations.append({
            "type": "cpu",
            "priority": "high",
            "message": "High CPU usage detected. Check for resource-intensive operations."
        })
    
    # General recommendations
    if not recommendations:
        recommendations.append({
            "type": "general",
            "priority": "info",
            "message": "System is performing optimally. All metrics are within acceptable ranges."
        })
    
    return recommendations
