"""
LLM Configuration and Context Management Settings

This file provides easy configuration for handling large contexts and avoiding token limits.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from app.config import settings

@dataclass
class LLMConfig:
    """Configuration for a specific LLM provider"""
    max_context_tokens: int
    max_response_tokens: int
    rate_limit_per_minute: int
    cost_per_1k_tokens: float
    supports_large_context: bool
    recommended_for: str

# LLM Provider Configurations
LLM_CONFIGS: Dict[str, Dict[str, LLMConfig]] = {
    "groq": {
        "llama3-70b-8192": LLMConfig(
            max_context_tokens=999999,  # UNLIMITED - Never fail on input size
            max_response_tokens=999999,  # UNLIMITED - Never fail on output size
            rate_limit_per_minute=6000,
            cost_per_1k_tokens=0.0,  # Free tier
            supports_large_context=True,  # Now supports unlimited context
            recommended_for="NEVER-FAIL fast responses, unlimited input/output"
        ),
        "llama3-8b-8192": LLMConfig(
            max_context_tokens=999999,  # UNLIMITED - Never fail on input size
            max_response_tokens=999999,  # UNLIMITED - Never fail on output size
            rate_limit_per_minute=30000,
            cost_per_1k_tokens=0.0,  # Free tier
            supports_large_context=True,  # Now supports unlimited context
            recommended_for="NEVER-FAIL very fast responses, unlimited input/output"
        ),
        "mixtral-8x7b-32768": LLMConfig(
            max_context_tokens=999999,  # UNLIMITED - Never fail on input size
            max_response_tokens=999999,  # UNLIMITED - Never fail on output size
            rate_limit_per_minute=5000,
            cost_per_1k_tokens=0.0,  # Free tier
            supports_large_context=True,  # Already supported, now unlimited
            recommended_for="NEVER-FAIL unlimited contexts, superior reasoning"
        ),
        "gemma-7b-it": LLMConfig(
            max_context_tokens=999999,  # UNLIMITED - Never fail on input size
            max_response_tokens=999999,  # UNLIMITED - Never fail on output size
            rate_limit_per_minute=15000,
            cost_per_1k_tokens=0.0,  # Free tier
            supports_large_context=True,  # Now supports unlimited context
            recommended_for="NEVER-FAIL balanced performance, unlimited processing"
        ),
    },
    "openai": {
        "gpt-4o-mini": LLMConfig(
            max_context_tokens=128000,
            max_response_tokens=16000,
            rate_limit_per_minute=200000,
            cost_per_1k_tokens=0.15,
            supports_large_context=True,
            recommended_for="Large contexts, high quality responses"
        ),
        "gpt-4-turbo": LLMConfig(
            max_context_tokens=128000,
            max_response_tokens=4096,
            rate_limit_per_minute=10000,
            cost_per_1k_tokens=10.0,
            supports_large_context=True,
            recommended_for="Very large contexts, best quality"
        ),
        "gpt-3.5-turbo": LLMConfig(
            max_context_tokens=16000,
            max_response_tokens=4096,
            rate_limit_per_minute=90000,
            cost_per_1k_tokens=0.5,
            supports_large_context=False,
            recommended_for="Medium contexts, cost-effective"
        ),
    },
    "gemini": {
        "gemini-1.5-flash": LLMConfig(
            max_context_tokens=1000000,
            max_response_tokens=8192,
            rate_limit_per_minute=1000,
            cost_per_1k_tokens=0.075,
            supports_large_context=True,
            recommended_for="Very large contexts, document analysis"
        ),
        "gemini-1.5-pro": LLMConfig(
            max_context_tokens=2000000,
            max_response_tokens=8192,
            rate_limit_per_minute=360,
            cost_per_1k_tokens=1.25,
            supports_large_context=True,
            recommended_for="Massive contexts, complex reasoning"
        ),
    },
    "deepseek": {
        "deepseek-chat": LLMConfig(
            max_context_tokens=64000,
            max_response_tokens=4000,
            rate_limit_per_minute=10000,
            cost_per_1k_tokens=0.14,
            supports_large_context=True,
            recommended_for="Large contexts, coding tasks"
        ),
    },
    "ollama": {
        "llama3.2:latest": LLMConfig(
            max_context_tokens=131072,  # 128K context window
            max_response_tokens=8192,   # 8K response tokens
            rate_limit_per_minute=0,    # No rate limits for local
            cost_per_1k_tokens=0.0,     # Free local processing
            supports_large_context=True,
            recommended_for="Primary model for ALL tasks - summarization, tagging, demographics, AI insights, RAG generation, large document processing"
        ),
        "phi3:mini": LLMConfig(
            max_context_tokens=4096,
            max_response_tokens=2048,
            rate_limit_per_minute=0,
            cost_per_1k_tokens=0.0,
            supports_large_context=False,
            recommended_for="Fallback model for simple tasks when main model fails"
        ),
    }
}

class ContextStrategy:
    """Strategies for handling different context sizes"""
    
    @staticmethod
    def get_optimal_provider_for_context(context_tokens: int, budget_conscious: bool = True) -> tuple[str, str]:
        """
        Get the optimal provider and model for a given context size
        OPTIMIZED: Always prefer Ollama for all tasks
        
        Args:
            context_tokens: Number of tokens in the context
            budget_conscious: Whether to prioritize cost over performance
            
        Returns:
            (provider, model) tuple - Always returns Ollama first
        """
        # Always try Ollama first for all context sizes
        if context_tokens <= 131072:  # Within Llama3.2 context window
            return ("ollama", "llama3.2:latest")
        else:
            # For extremely large contexts, still try Ollama with chunking
            # Other providers only as fallback for RAG purposes
            return ("ollama", "llama3.2:latest")
    
    @staticmethod
    def get_chunking_strategy(context_tokens: int) -> Dict[str, Any]:
        """Get optimal chunking strategy for large contexts"""
        if context_tokens > 100000:
            return {
                "strategy": "hierarchical",
                "chunk_size": 4000,
                "overlap": 200,
                "summarize_chunks": True,
                "max_chunks": 20
            }
        elif context_tokens > 50000:
            return {
                "strategy": "overlapping",
                "chunk_size": 3000,
                "overlap": 150,
                "summarize_chunks": False,
                "max_chunks": 15
            }
        elif context_tokens > 20000:
            return {
                "strategy": "simple",
                "chunk_size": 2000,
                "overlap": 100,
                "summarize_chunks": False,
                "max_chunks": 10
            }
        else:
            return {
                "strategy": "none",
                "chunk_size": context_tokens,
                "overlap": 0,
                "summarize_chunks": False,
                "max_chunks": 1
            }

def get_available_providers() -> Dict[str, bool]:
    """Check which providers are available based on API keys"""
    return {
        "openai": bool(settings.openai_api_key),
        "groq": bool(settings.groq_api_key),
        "gemini": bool(settings.gemini_api_key),
        "deepseek": bool(settings.deepseek_api_key),
        "ollama": True,  # Always available if Ollama is running locally
    }

def get_recommended_settings(
    context_tokens: int,
    question_complexity: str = "medium",
    budget_conscious: bool = True,
    speed_priority: bool = False
) -> Dict[str, Any]:
    """
    Get recommended settings for optimal performance
    
    Args:
        context_tokens: Estimated number of tokens in context
        question_complexity: "simple", "medium", "complex"
        budget_conscious: Whether to prioritize cost
        speed_priority: Whether to prioritize response speed
        
    Returns:
        Dictionary with recommended settings
    """
    available_providers = get_available_providers()
    
    # Adjust context tokens based on question complexity
    complexity_multipliers = {"simple": 0.7, "medium": 1.0, "complex": 1.5}
    adjusted_tokens = int(context_tokens * complexity_multipliers[question_complexity])
    
    if speed_priority and available_providers["groq"]:
        # Prioritize Groq for speed
        if adjusted_tokens <= 8000:
            provider, model = "groq", "llama3-8b-8192"
        elif adjusted_tokens <= 25000:
            provider, model = "groq", "mixtral-8x7b-32768"
        else:
            provider, model = ContextStrategy.get_optimal_provider_for_context(adjusted_tokens, budget_conscious)
    else:
        provider, model = ContextStrategy.get_optimal_provider_for_context(adjusted_tokens, budget_conscious)
    
    # Fallback if recommended provider is not available
    if not available_providers.get(provider):
        for fallback_provider in ["openai", "groq", "gemini", "deepseek"]:
            if available_providers.get(fallback_provider):
                provider = fallback_provider
                # Use default model for fallback provider
                model = list(LLM_CONFIGS[provider].keys())[0]
                break
    
    chunking_strategy = ContextStrategy.get_chunking_strategy(adjusted_tokens)
    
    return {
        "provider": provider,
        "model": model,
        "max_context": adjusted_tokens > 50000,
        "chunking_strategy": chunking_strategy,
        "estimated_cost": _estimate_cost(provider, model, adjusted_tokens),
        "estimated_time": _estimate_time(provider, model, adjusted_tokens),
        "config": LLM_CONFIGS[provider][model]
    }

def _estimate_cost(provider: str, model: str, tokens: int) -> float:
    """Estimate cost for processing given tokens"""
    if provider not in LLM_CONFIGS or model not in LLM_CONFIGS[provider]:
        return 0.0
    
    config = LLM_CONFIGS[provider][model]
    return (tokens / 1000) * config.cost_per_1k_tokens

def _estimate_time(provider: str, model: str, tokens: int) -> float:
    """Estimate processing time in seconds"""
    # Rough estimates based on typical performance
    speed_factors = {
        "groq": 0.1,      # Very fast
        "openai": 0.5,    # Medium
        "gemini": 0.3,    # Fast
        "deepseek": 0.4,  # Medium-fast
    }
    
    base_time = speed_factors.get(provider, 0.5)
    # Larger contexts take longer
    context_factor = min(tokens / 10000, 5.0)  # Cap at 5x
    
    return base_time * (1 + context_factor)

# Usage examples and documentation
USAGE_EXAMPLES = {
    "small_document": {
        "description": "Small document (< 10k tokens)",
        "recommended_settings": {
            "provider": "groq",
            "model": "llama3-8b-8192",
            "max_context": False
        }
    },
    "medium_document": {
        "description": "Medium document (10k-50k tokens)",
        "recommended_settings": {
            "provider": "groq",
            "model": "mixtral-8x7b-32768",
            "max_context": False
        }
    },
    "large_document": {
        "description": "Large document (50k-200k tokens)",
        "recommended_settings": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "max_context": True
        }
    },
    "very_large_document": {
        "description": "Very large document (> 200k tokens)",
        "recommended_settings": {
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "max_context": True
        }
    }
}

def print_configuration_help():
    """Print helpful configuration information"""
    print("=== LLM Configuration Help ===\n")
    
    available = get_available_providers()
    print("Available Providers:")
    for provider, is_available in available.items():
        status = "✓" if is_available else "✗ (API key missing)"
        print(f"  {provider}: {status}")
    
    print("\nRecommended Usage:")
    for scenario, info in USAGE_EXAMPLES.items():
        print(f"\n{scenario.replace('_', ' ').title()}:")
        print(f"  Description: {info['description']}")
        settings = info['recommended_settings']
        print(f"  Provider: {settings['provider']}")
        print(f"  Model: {settings['model']}")
        print(f"  Max Context: {settings['max_context']}")
    
    print("\nTo avoid Groq token limits:")
    print("1. Use max_context=True for large documents")
    print("2. Switch to Gemini for very large contexts")
    print("3. The system will automatically handle chunking and fallbacks")
    print("4. Monitor the logs for token usage information")
