"""
RAG Model Configuration Management
=================================

Manages embedding and reranker model configurations for optimal RAG performance.
Supports switching between different model strategies and performance profiles.

OPTIMAL CONFIGURATION (Based on benchmarks):
- Embedding: JinaAI-v2-base-en (Hit Rate: 0.938, MRR: 0.868)
- Reranker: BAAI/bge-reranker-large (Best Overall Performance)
"""

from __future__ import annotations
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for a model"""
    name: str
    model_path: str
    vector_size: int
    expected_hit_rate: float
    expected_mrr: float
    load_time_estimate: float  # seconds
    memory_usage_mb: int
    description: str
    benchmark_source: str

@dataclass
class RAGStrategy:
    """Complete RAG strategy configuration"""
    name: str
    description: str
    embedding_config: ModelConfig
    reranker_config: ModelConfig
    priority: int  # Lower = higher priority
    use_cases: List[str]
    performance_profile: str  # "speed", "quality", "balanced"

class PerformanceProfile(Enum):
    SPEED = "speed"
    QUALITY = "quality" 
    BALANCED = "balanced"

class ModelConfigManager:
    """Manages RAG model configurations and strategies"""
    
    def __init__(self, config_dir: str = "model_configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize predefined configurations
        self.embedding_configs = self._init_embedding_configs()
        self.reranker_configs = self._init_reranker_configs()
        self.strategies = self._init_strategies()
        
    def _init_embedding_configs(self) -> Dict[str, ModelConfig]:
        """Initialize embedding model configurations"""
        return {
            "jina-v2-base-en": ModelConfig(
                name="JinaAI v2 Base EN",
                model_path="jinaai/jina-embeddings-v2-base-en",
                vector_size=768,
                expected_hit_rate=0.938,
                expected_mrr=0.868,
                load_time_estimate=15.0,
                memory_usage_mb=1200,
                description="Best overall performance embedding model based on benchmarks",
                benchmark_source="Best Overall Performance Study"
            ),
            "bge-large-en": ModelConfig(
                name="BGE Large EN v1.5",
                model_path="BAAI/bge-large-en-v1.5",
                vector_size=1024,
                expected_hit_rate=0.885,
                expected_mrr=0.820,
                load_time_estimate=12.0,
                memory_usage_mb=1400,
                description="High-quality embedding model with larger vectors",
                benchmark_source="Baseline Comparison"
            ),
            "bge-base-en": ModelConfig(
                name="BGE Base EN v1.5",
                model_path="BAAI/bge-base-en-v1.5",
                vector_size=768,
                expected_hit_rate=0.850,
                expected_mrr=0.780,
                load_time_estimate=8.0,
                memory_usage_mb=800,
                description="Fast, efficient embedding model",
                benchmark_source="Speed Comparison"
            ),
            "all-minilm-l6": ModelConfig(
                name="All MiniLM L6 v2",
                model_path="sentence-transformers/all-MiniLM-L6-v2",
                vector_size=384,
                expected_hit_rate=0.750,
                expected_mrr=0.680,
                load_time_estimate=5.0,
                memory_usage_mb=400,
                description="Lightweight, fast embedding model",
                benchmark_source="Lightweight Comparison"
            )
        }
    
    def _init_reranker_configs(self) -> Dict[str, ModelConfig]:
        """Initialize reranker model configurations"""
        return {
            "bge-reranker-large": ModelConfig(
                name="BGE Reranker Large",
                model_path="BAAI/bge-reranker-large",
                vector_size=0,  # Not applicable for rerankers
                expected_hit_rate=0.938,
                expected_mrr=0.868,
                load_time_estimate=20.0,
                memory_usage_mb=2500,
                description="Best performing reranker model (optimal benchmark results)",
                benchmark_source="Best Overall Performance Study"
            ),
            "bge-reranker-base": ModelConfig(
                name="BGE Reranker Base",
                model_path="BAAI/bge-reranker-base",
                vector_size=0,
                expected_hit_rate=0.910,
                expected_mrr=0.845,
                load_time_estimate=15.0,
                memory_usage_mb=1800,
                description="Balanced performance and efficiency reranker",
                benchmark_source="Efficiency Comparison"
            ),
            "ms-marco-minilm": ModelConfig(
                name="MS MARCO MiniLM L6 v2",
                model_path="cross-encoder/ms-marco-MiniLM-L-6-v2",
                vector_size=0,
                expected_hit_rate=0.870,
                expected_mrr=0.780,
                load_time_estimate=8.0,
                memory_usage_mb=900,
                description="Fast, lightweight reranker",
                benchmark_source="Speed Comparison"
            ),
            "ms-marco-roberta": ModelConfig(
                name="MS MARCO RoBERTa Base",
                model_path="cross-encoder/ms-marco-MiniLM-L-12-v2",
                vector_size=0,
                expected_hit_rate=0.890,
                expected_mrr=0.810,
                load_time_estimate=12.0,
                memory_usage_mb=1300,
                description="High-quality reranker with good speed/quality balance",
                benchmark_source="Quality Comparison"
            )
        }
    
    def _init_strategies(self) -> Dict[str, RAGStrategy]:
        """Initialize predefined RAG strategies"""
        return {
            "optimal": RAGStrategy(
                name="Optimal Performance",
                description="Best overall performance based on benchmark results",
                embedding_config=self.embedding_configs["jina-v2-base-en"],
                reranker_config=self.reranker_configs["bge-reranker-large"],
                priority=1,
                use_cases=["production", "high-quality", "benchmark"],
                performance_profile="quality"
            ),
            "balanced": RAGStrategy(
                name="Balanced Performance",
                description="Good balance of speed and quality",
                embedding_config=self.embedding_configs["bge-base-en"],
                reranker_config=self.reranker_configs["bge-reranker-base"],
                priority=2,
                use_cases=["general", "balanced", "medium-scale"],
                performance_profile="balanced"
            ),
            "speed": RAGStrategy(
                name="Speed Optimized",
                description="Maximum speed with acceptable quality",
                embedding_config=self.embedding_configs["all-minilm-l6"],
                reranker_config=self.reranker_configs["ms-marco-minilm"],
                priority=3,
                use_cases=["development", "fast-prototyping", "high-throughput"],
                performance_profile="speed"
            ),
            "quality": RAGStrategy(
                name="Quality Focused",
                description="Maximum quality regardless of speed",
                embedding_config=self.embedding_configs["bge-large-en"],
                reranker_config=self.reranker_configs["bge-reranker-large"],
                priority=4,
                use_cases=["research", "critical-applications", "accuracy-first"],
                performance_profile="quality"
            )
        }
    
    def get_strategy(self, strategy_name: str) -> Optional[RAGStrategy]:
        """Get a strategy by name"""
        return self.strategies.get(strategy_name)
    
    def get_optimal_strategy(self) -> RAGStrategy:
        """Get the optimal strategy (benchmark winner)"""
        return self.strategies["optimal"]
    
    def get_strategy_by_profile(self, profile: PerformanceProfile) -> RAGStrategy:
        """Get strategy by performance profile"""
        for strategy in self.strategies.values():
            if strategy.performance_profile == profile.value:
                return strategy
        
        # Default to balanced if profile not found
        return self.strategies["balanced"]
    
    def recommend_strategy(
        self,
        use_case: str = None,
        max_memory_mb: int = None,
        max_load_time_s: float = None,
        min_hit_rate: float = None,
        min_mrr: float = None
    ) -> List[RAGStrategy]:
        """Recommend strategies based on constraints"""
        candidates = []
        
        for strategy in self.strategies.values():
            # Check use case
            if use_case and use_case not in strategy.use_cases:
                continue
            
            # Check memory constraints
            total_memory = (
                strategy.embedding_config.memory_usage_mb + 
                strategy.reranker_config.memory_usage_mb
            )
            if max_memory_mb and total_memory > max_memory_mb:
                continue
            
            # Check load time constraints
            total_load_time = (
                strategy.embedding_config.load_time_estimate + 
                strategy.reranker_config.load_time_estimate
            )
            if max_load_time_s and total_load_time > max_load_time_s:
                continue
            
            # Check performance constraints
            if min_hit_rate and strategy.embedding_config.expected_hit_rate < min_hit_rate:
                continue
            
            if min_mrr and strategy.embedding_config.expected_mrr < min_mrr:
                continue
            
            candidates.append(strategy)
        
        # Sort by priority (lower number = higher priority)
        candidates.sort(key=lambda s: s.priority)
        return candidates
    
    def compare_strategies(self, strategy_names: List[str]) -> Dict[str, Any]:
        """Compare multiple strategies"""
        comparison = {
            "strategies": {},
            "summary": {
                "fastest_load": None,
                "best_hit_rate": None,
                "best_mrr": None,
                "lowest_memory": None,
                "recommended": None
            }
        }
        
        for name in strategy_names:
            strategy = self.strategies.get(name)
            if not strategy:
                continue
            
            total_memory = (
                strategy.embedding_config.memory_usage_mb + 
                strategy.reranker_config.memory_usage_mb
            )
            total_load_time = (
                strategy.embedding_config.load_time_estimate + 
                strategy.reranker_config.load_time_estimate
            )
            
            comparison["strategies"][name] = {
                "description": strategy.description,
                "performance_profile": strategy.performance_profile,
                "total_memory_mb": total_memory,
                "total_load_time_s": total_load_time,
                "expected_hit_rate": strategy.embedding_config.expected_hit_rate,
                "expected_mrr": strategy.embedding_config.expected_mrr,
                "use_cases": strategy.use_cases,
                "embedding_model": strategy.embedding_config.name,
                "reranker_model": strategy.reranker_config.name
            }
        
        # Find best in each category
        if comparison["strategies"]:
            strategies_data = comparison["strategies"]
            
            # Find fastest load time
            fastest = min(strategies_data.items(), 
                         key=lambda x: x[1]["total_load_time_s"])
            comparison["summary"]["fastest_load"] = fastest[0]
            
            # Find best hit rate
            best_hit = max(strategies_data.items(), 
                          key=lambda x: x[1]["expected_hit_rate"])
            comparison["summary"]["best_hit_rate"] = best_hit[0]
            
            # Find best MRR
            best_mrr = max(strategies_data.items(), 
                          key=lambda x: x[1]["expected_mrr"])
            comparison["summary"]["best_mrr"] = best_mrr[0]
            
            # Find lowest memory
            lowest_mem = min(strategies_data.items(), 
                           key=lambda x: x[1]["total_memory_mb"])
            comparison["summary"]["lowest_memory"] = lowest_mem[0]
            
            # Recommended (optimal strategy if present)
            if "optimal" in strategies_data:
                comparison["summary"]["recommended"] = "optimal"
            else:
                comparison["summary"]["recommended"] = best_hit[0]
        
        return comparison
    
    def export_current_config(self) -> Dict[str, Any]:
        """Export current configuration"""
        from app.utils.qdrant_store import get_model_info
        from app.services.rag import reranker
        
        try:
            embedding_info = get_model_info()
            reranker_info = reranker.get_reranker_info()
            
            return {
                "timestamp": time.time(),
                "current_models": {
                    "embedding": embedding_info,
                    "reranker": reranker_info
                },
                "available_strategies": {
                    name: {
                        "description": strategy.description,
                        "performance_profile": strategy.performance_profile,
                        "embedding_model": strategy.embedding_config.model_path,
                        "reranker_model": strategy.reranker_config.model_path,
                        "expected_performance": {
                            "hit_rate": strategy.embedding_config.expected_hit_rate,
                            "mrr": strategy.embedding_config.expected_mrr
                        }
                    }
                    for name, strategy in self.strategies.items()
                }
            }
        except Exception as e:
            return {
                "error": f"Could not export current config: {e}",
                "available_strategies": {
                    name: strategy.description 
                    for name, strategy in self.strategies.items()
                }
            }
    
    def save_config(self, filename: str = None):
        """Save configuration to file"""
        if not filename:
            filename = f"rag_config_{int(time.time())}.json"
        
        config_file = self.config_dir / filename
        config_data = self.export_current_config()
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"📁 Configuration saved to {config_file}")
        return config_file
    
    def print_strategy_report(self, strategy_name: str = None):
        """Print detailed strategy report"""
        if strategy_name:
            strategies = [strategy_name]
        else:
            strategies = list(self.strategies.keys())
        
        print("\n" + "="*80)
        print("🔧 RAG MODEL CONFIGURATION REPORT")
        print("="*80)
        
        for name in strategies:
            strategy = self.strategies.get(name)
            if not strategy:
                continue
            
            print(f"\n📋 STRATEGY: {strategy.name.upper()}")
            print(f"   Description: {strategy.description}")
            print(f"   Profile: {strategy.performance_profile}")
            print(f"   Priority: {strategy.priority}")
            print(f"   Use Cases: {', '.join(strategy.use_cases)}")
            
            print(f"\n   🔤 EMBEDDING MODEL:")
            emb = strategy.embedding_config
            print(f"      • Name: {emb.name}")
            print(f"      • Path: {emb.model_path}")
            print(f"      • Vector Size: {emb.vector_size}")
            print(f"      • Expected Hit Rate: {emb.expected_hit_rate:.3f}")
            print(f"      • Expected MRR: {emb.expected_mrr:.3f}")
            print(f"      • Load Time: ~{emb.load_time_estimate:.1f}s")
            print(f"      • Memory: ~{emb.memory_usage_mb}MB")
            
            print(f"\n   🔄 RERANKER MODEL:")
            rer = strategy.reranker_config
            print(f"      • Name: {rer.name}")
            print(f"      • Path: {rer.model_path}")
            print(f"      • Expected Hit Rate: {rer.expected_hit_rate:.3f}")
            print(f"      • Expected MRR: {rer.expected_mrr:.3f}")
            print(f"      • Load Time: ~{rer.load_time_estimate:.1f}s")
            print(f"      • Memory: ~{rer.memory_usage_mb}MB")
            
            total_memory = emb.memory_usage_mb + rer.memory_usage_mb
            total_load_time = emb.load_time_estimate + rer.load_time_estimate
            
            print(f"\n   📊 TOTAL REQUIREMENTS:")
            print(f"      • Total Memory: ~{total_memory}MB")
            print(f"      • Total Load Time: ~{total_load_time:.1f}s")
            print(f"      • Combined Performance Score: {(emb.expected_hit_rate + emb.expected_mrr) / 2 * 100:.1f}/100")
        
        # Show current configuration
        try:
            current_config = self.export_current_config()
            if "current_models" in current_config:
                print(f"\n🔧 CURRENT ACTIVE MODELS:")
                current = current_config["current_models"]
                print(f"   • Embedding: {current['embedding'].get('model_name', 'Unknown')}")
                print(f"   • Reranker: {current['reranker'].get('model_name', 'Unknown')}")
        except Exception:
            print(f"\n⚠️  Could not determine current active models")
        
        print("="*80)

# Global instance
model_config_manager = ModelConfigManager()

# Convenience functions
def get_optimal_strategy() -> RAGStrategy:
    """Get the optimal strategy based on benchmarks"""
    return model_config_manager.get_optimal_strategy()

def recommend_strategy(**kwargs) -> List[RAGStrategy]:
    """Recommend strategies based on constraints"""
    return model_config_manager.recommend_strategy(**kwargs)

def compare_strategies(strategy_names: List[str]) -> Dict[str, Any]:
    """Compare multiple strategies"""
    return model_config_manager.compare_strategies(strategy_names)

def print_strategy_report(strategy_name: str = None):
    """Print strategy report"""
    return model_config_manager.print_strategy_report(strategy_name)

if __name__ == "__main__":
    # Example usage
    print("🚀 RAG Model Configuration Manager")
    
    # Show optimal strategy
    optimal = get_optimal_strategy()
    print(f"\n✅ Optimal Strategy: {optimal.name}")
    print(f"   Embedding: {optimal.embedding_config.name}")
    print(f"   Reranker: {optimal.reranker_config.name}")
    print(f"   Expected Hit Rate: {optimal.embedding_config.expected_hit_rate:.3f}")
    print(f"   Expected MRR: {optimal.embedding_config.expected_mrr:.3f}")
    
    # Show all strategies
    print_strategy_report()
    
    # Compare top strategies
    comparison = compare_strategies(["optimal", "balanced", "speed"])
    print(f"\n📊 Strategy Comparison:")
    print(f"   • Fastest Load: {comparison['summary']['fastest_load']}")
    print(f"   • Best Hit Rate: {comparison['summary']['best_hit_rate']}")
    print(f"   • Best MRR: {comparison['summary']['best_mrr']}")
    print(f"   • Lowest Memory: {comparison['summary']['lowest_memory']}")
    print(f"   • Recommended: {comparison['summary']['recommended']}")
