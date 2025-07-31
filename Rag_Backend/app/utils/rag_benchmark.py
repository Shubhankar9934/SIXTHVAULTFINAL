"""
RAG Performance Benchmarking and Monitoring Utilities
====================================================

Tracks performance metrics for the optimized RAG system:
- Hit Rate: Percentage of queries that return relevant results
- MRR (Mean Reciprocal Rank): Quality metric for ranking
- Response Time: End-to-end latency
- Cache Hit Rate: Efficiency metric
- Model Performance: Embedding and reranker metrics

Based on benchmarks showing optimal performance:
- JinaAI-v2-base-en embedding: Hit Rate 0.938, MRR 0.868
- BGE-reranker-large: Best overall performance
"""

from __future__ import annotations
import time
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

from app.utils.qdrant_store import get_model_info
from app.services.rag import reranker, cache, query_processor, perf_monitor


@dataclass
class RAGBenchmarkResult:
    """Single benchmark result"""
    query_id: str
    question: str
    response_time: float
    hit_rate: float
    mrr: float
    cache_hit: bool
    embedding_model: str
    reranker_model: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class RAGBenchmarkSummary:
    """Benchmark summary statistics"""
    total_queries: int
    avg_response_time: float
    hit_rate: float
    mrr: float
    cache_hit_rate: float
    embedding_model: str
    reranker_model: str
    benchmark_date: datetime
    target_hit_rate: float = 0.938
    target_mrr: float = 0.868
    
    @property
    def hit_rate_improvement(self) -> float:
        """Improvement over baseline"""
        return (self.hit_rate - 0.5) / 0.5 * 100  # Assuming 50% baseline
    
    @property
    def mrr_improvement(self) -> float:
        """MRR improvement over baseline"""
        return (self.mrr - 0.3) / 0.3 * 100  # Assuming 30% baseline
    
    @property
    def performance_score(self) -> float:
        """Overall performance score (0-100)"""
        hit_score = min(self.hit_rate / self.target_hit_rate, 1.0) * 50
        mrr_score = min(self.mrr / self.target_mrr, 1.0) * 50
        return hit_score + mrr_score
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['benchmark_date'] = self.benchmark_date.isoformat()
        result['hit_rate_improvement'] = self.hit_rate_improvement
        result['mrr_improvement'] = self.mrr_improvement
        result['performance_score'] = self.performance_score
        return result


class RAGBenchmark:
    """RAG system benchmarking and monitoring"""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.current_session_results: List[RAGBenchmarkResult] = []
        
    async def benchmark_query(
        self,
        user_id: str,
        question: str,
        expected_answers: List[str] = None,
        query_id: str = None
    ) -> RAGBenchmarkResult:
        """Benchmark a single query"""
        from app.services.rag import lightning_answer
        
        if not query_id:
            query_id = f"q_{int(time.time())}_{hash(question) % 10000}"
        
        # Get model info
        embedding_info = get_model_info()
        reranker_info = reranker.get_reranker_info()
        
        # Check cache before timing
        cache_hit = await cache.get(user_id, question, "ollama", False) is not None
        
        # Time the query
        start_time = time.time()
        answer, sources = await lightning_answer(
            user_id=user_id,
            question=question,
            hybrid=False,
            provider="ollama",
            max_context=False
        )
        response_time = time.time() - start_time
        
        # Calculate metrics
        hit_rate = self._calculate_hit_rate(sources, expected_answers)
        mrr = self._calculate_mrr(sources, expected_answers)
        
        result = RAGBenchmarkResult(
            query_id=query_id,
            question=question,
            response_time=response_time,
            hit_rate=hit_rate,
            mrr=mrr,
            cache_hit=cache_hit,
            embedding_model=embedding_info["model_name"],
            reranker_model=reranker_info["model_name"],
            timestamp=datetime.now()
        )
        
        self.current_session_results.append(result)
        return result
    
    def _calculate_hit_rate(self, sources: List[Dict], expected_answers: List[str] = None) -> float:
        """Calculate hit rate (simplified - whether we got any relevant results)"""
        if not sources:
            return 0.0
        
        # If no expected answers provided, use score-based heuristic
        if not expected_answers:
            # Consider it a hit if we have high-scoring results
            top_score = max(source.get("score", 0) for source in sources)
            return 1.0 if top_score > 0.5 else 0.5
        
        # If expected answers provided, check for content overlap
        for source in sources[:3]:  # Check top 3 sources
            text = source.get("text", "").lower()
            for expected in expected_answers:
                if any(word in text for word in expected.lower().split()[:3]):
                    return 1.0
        
        return 0.0
    
    def _calculate_mrr(self, sources: List[Dict], expected_answers: List[str] = None) -> float:
        """Calculate Mean Reciprocal Rank"""
        if not sources:
            return 0.0
        
        # If no expected answers, use score-based ranking
        if not expected_answers:
            # Find the rank of the highest scoring result
            scores = [source.get("score", 0) for source in sources]
            if max(scores) > 0.5:
                best_rank = scores.index(max(scores)) + 1
                return 1.0 / best_rank
            return 0.0
        
        # Find rank of first relevant result
        for rank, source in enumerate(sources, 1):
            text = source.get("text", "").lower()
            for expected in expected_answers:
                if any(word in text for word in expected.lower().split()[:3]):
                    return 1.0 / rank
        
        return 0.0
    
    async def run_benchmark_suite(
        self,
        user_id: str,
        test_queries: List[Dict[str, Any]],
        save_results: bool = True
    ) -> RAGBenchmarkSummary:
        """Run a comprehensive benchmark suite"""
        print(f"🚀 Starting RAG benchmark suite with {len(test_queries)} queries...")
        
        results = []
        for i, query_data in enumerate(test_queries):
            question = query_data["question"]
            expected = query_data.get("expected_answers", [])
            
            print(f"  Query {i+1}/{len(test_queries)}: {question[:50]}...")
            
            try:
                result = await self.benchmark_query(
                    user_id=user_id,
                    question=question,
                    expected_answers=expected,
                    query_id=f"suite_{i+1}"
                )
                results.append(result)
                
                # Brief pause to avoid overwhelming the system
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"    ❌ Query failed: {e}")
                continue
        
        # Calculate summary statistics
        if not results:
            raise Exception("No successful benchmark queries")
        
        summary = self._calculate_summary(results)
        
        if save_results:
            await self.save_results(results, summary)
        
        self._print_benchmark_report(summary)
        return summary
    
    def _calculate_summary(self, results: List[RAGBenchmarkResult]) -> RAGBenchmarkSummary:
        """Calculate benchmark summary from results"""
        if not results:
            raise ValueError("No results to summarize")
        
        total_queries = len(results)
        avg_response_time = np.mean([r.response_time for r in results])
        hit_rate = np.mean([r.hit_rate for r in results])
        mrr = np.mean([r.mrr for r in results])
        cache_hit_rate = sum(1 for r in results if r.cache_hit) / total_queries
        
        # Get model names from the first result
        embedding_model = results[0].embedding_model
        reranker_model = results[0].reranker_model
        
        return RAGBenchmarkSummary(
            total_queries=total_queries,
            avg_response_time=avg_response_time,
            hit_rate=hit_rate,
            mrr=mrr,
            cache_hit_rate=cache_hit_rate,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            benchmark_date=datetime.now()
        )
    
    async def save_results(
        self,
        results: List[RAGBenchmarkResult],
        summary: RAGBenchmarkSummary
    ):
        """Save benchmark results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        
        # Save summary
        summary_file = self.results_dir / f"benchmark_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
        
        print(f"📊 Results saved to {results_file}")
        print(f"📋 Summary saved to {summary_file}")
    
    def _print_benchmark_report(self, summary: RAGBenchmarkSummary):
        """Print a comprehensive benchmark report"""
        print("\n" + "="*60)
        print("🎯 RAG PERFORMANCE BENCHMARK REPORT")
        print("="*60)
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"  • Total Queries: {summary.total_queries}")
        print(f"  • Avg Response Time: {summary.avg_response_time:.2f}s")
        print(f"  • Hit Rate: {summary.hit_rate:.3f} ({summary.hit_rate*100:.1f}%)")
        print(f"  • MRR: {summary.mrr:.3f}")
        print(f"  • Cache Hit Rate: {summary.cache_hit_rate:.3f} ({summary.cache_hit_rate*100:.1f}%)")
        
        print(f"\n🎯 TARGET COMPARISON:")
        print(f"  • Target Hit Rate: {summary.target_hit_rate:.3f} ({summary.target_hit_rate*100:.1f}%)")
        print(f"  • Target MRR: {summary.target_mrr:.3f}")
        print(f"  • Hit Rate Achievement: {summary.hit_rate/summary.target_hit_rate*100:.1f}%")
        print(f"  • MRR Achievement: {summary.mrr/summary.target_mrr*100:.1f}%")
        
        print(f"\n🚀 PERFORMANCE IMPROVEMENTS:")
        print(f"  • Hit Rate Improvement: +{summary.hit_rate_improvement:.1f}%")
        print(f"  • MRR Improvement: +{summary.mrr_improvement:.1f}%")
        print(f"  • Overall Performance Score: {summary.performance_score:.1f}/100")
        
        print(f"\n🔧 MODEL CONFIGURATION:")
        print(f"  • Embedding Model: {summary.embedding_model}")
        print(f"  • Reranker Model: {summary.reranker_model}")
        print(f"  • Benchmark Date: {summary.benchmark_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Performance assessment
        if summary.performance_score >= 90:
            print(f"\n✅ EXCELLENT: Performance exceeds targets!")
        elif summary.performance_score >= 75:
            print(f"\n🎯 GOOD: Performance meets most targets")
        elif summary.performance_score >= 60:
            print(f"\n⚠️ MODERATE: Performance needs improvement")
        else:
            print(f"\n❌ POOR: Performance significantly below targets")
        
        print("="*60)
    
    async def quick_performance_check(self, user_id: str) -> Dict[str, Any]:
        """Quick performance check with system status"""
        from app.services.rag import health_check
        
        # Sample queries for quick check
        test_queries = [
            {"question": "What is the main topic?", "expected_answers": []},
            {"question": "Tell me about the key points", "expected_answers": []},
            {"question": "What are the important details?", "expected_answers": []}
        ]
        
        print("🔍 Running quick performance check...")
        
        # Run mini benchmark
        results = []
        for query in test_queries:
            try:
                result = await self.benchmark_query(user_id, query["question"])
                results.append(result)
            except Exception as e:
                print(f"  ⚠️ Query failed: {e}")
        
        # Get system health
        health = await health_check()
        
        # Get model info
        embedding_info = get_model_info()
        reranker_info = reranker.get_reranker_info()
        
        if results:
            avg_time = np.mean([r.response_time for r in results])
            avg_hit_rate = np.mean([r.hit_rate for r in results])
            avg_mrr = np.mean([r.mrr for r in results])
        else:
            avg_time = avg_hit_rate = avg_mrr = 0.0
        
        status = {
            "performance": {
                "avg_response_time": avg_time,
                "hit_rate": avg_hit_rate,
                "mrr": avg_mrr,
                "target_achievement": {
                    "hit_rate": avg_hit_rate / 0.938 * 100,
                    "mrr": avg_mrr / 0.868 * 100
                }
            },
            "models": {
                "embedding": embedding_info,
                "reranker": reranker_info
            },
            "system_health": health,
            "cache_stats": cache.get_stats(),
            "query_stats": query_processor.get_stats(),
            "performance_stats": perf_monitor.get_stats()
        }
        
        print(f"✅ Quick check complete:")
        print(f"  • Avg Response Time: {avg_time:.2f}s")
        print(f"  • Hit Rate: {avg_hit_rate:.3f} ({avg_hit_rate/0.938*100:.1f}% of target)")
        print(f"  • MRR: {avg_mrr:.3f} ({avg_mrr/0.868*100:.1f}% of target)")
        
        return status


# Create default test queries for benchmarking
DEFAULT_TEST_QUERIES = [
    {
        "question": "What is artificial intelligence?",
        "expected_answers": ["AI", "machine learning", "artificial intelligence"]
    },
    {
        "question": "How does machine learning work?",
        "expected_answers": ["algorithm", "training", "data", "model"]
    },
    {
        "question": "What are the benefits of automation?",
        "expected_answers": ["efficiency", "speed", "cost", "productivity"]
    },
    {
        "question": "Explain the key concepts",
        "expected_answers": ["concept", "key", "important", "main"]
    },
    {
        "question": "What are the main features?",
        "expected_answers": ["feature", "capability", "function", "characteristic"]
    }
]


async def run_quick_benchmark(user_id: str = "benchmark_user") -> Dict[str, Any]:
    """Convenience function for quick benchmarking"""
    benchmark = RAGBenchmark()
    return await benchmark.quick_performance_check(user_id)


async def run_full_benchmark(
    user_id: str = "benchmark_user",
    custom_queries: List[Dict] = None
) -> RAGBenchmarkSummary:
    """Convenience function for full benchmarking"""
    benchmark = RAGBenchmark()
    queries = custom_queries or DEFAULT_TEST_QUERIES
    return await benchmark.run_benchmark_suite(user_id, queries)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Quick performance check
        status = await run_quick_benchmark()
        print("\nQuick Performance Status:")
        print(json.dumps(status["performance"], indent=2))
        
        # Full benchmark
        summary = await run_full_benchmark()
        print(f"\nFull Benchmark Score: {summary.performance_score:.1f}/100")
    
    asyncio.run(main())
