"""
AWS Bedrock Claude 3 Haiku Client - Ultra-Fast, Never-Fail, Unlimited Token Support
===================================================================================

FEATURES:
- Claude 3 Haiku only (ultra-fast, cost-effective)
- Unlimited token handling via intelligent chunking
- Never-fail architecture with exponential backoff
- Parallel processing support
- Cost tracking and optimization
- Circuit breaker pattern for reliability
- Smart text synthesis for large documents

PERFORMANCE TARGETS:
- Simple tasks: <1 second
- Complex tasks: 2-4 seconds  
- Large documents: 5-15 seconds
- Cost: $0.25/1M input tokens, $1.25/1M output tokens
- Reliability: 99.9% uptime
"""

import asyncio
import boto3
import json
import time
import hashlib
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from botocore.exceptions import ClientError, BotoCoreError
from botocore.config import Config
import nltk

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BedrockConfig:
    """Configuration for AWS Bedrock Claude 3 Haiku"""
    # AWS Configuration
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str = "us-east-1"
    
    # Model Configuration
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    max_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 0.9
    
    # Performance Configuration
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    timeout: int = 300  # 5 minutes max per request
    
    # Chunking Configuration
    max_input_tokens: int = 180000  # Leave buffer for Claude 3 Haiku's 200K limit
    chunk_size: int = 150000  # Optimal chunk size
    chunk_overlap: int = 5000  # Overlap for context preservation
    max_chunks_parallel: int = 8  # Parallel processing limit
    
    # Cost Tracking
    input_token_cost: float = 0.00000025  # $0.25 per 1M tokens
    output_token_cost: float = 0.00000125  # $1.25 per 1M tokens
    
    # Circuit Breaker
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    """Circuit breaker for Bedrock reliability"""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    
    def can_execute(self, config: BedrockConfig) -> bool:
        """Check if request can be executed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if (self.last_failure_time and 
                (datetime.now() - self.last_failure_time).seconds > config.recovery_timeout):
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker moving to HALF_OPEN state")
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful request"""
        self.failure_count = 0
        self.last_failure_time = None
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info("Circuit breaker CLOSED - service recovered")
    
    def record_failure(self, config: BedrockConfig):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")

@dataclass
class CostTracker:
    """Track Bedrock usage costs"""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    total_cost: float = 0.0
    session_start: datetime = field(default_factory=datetime.now)
    
    def add_usage(self, input_tokens: int, output_tokens: int, config: BedrockConfig):
        """Add usage to cost tracking"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1
        
        request_cost = (
            input_tokens * config.input_token_cost +
            output_tokens * config.output_token_cost
        )
        self.total_cost += request_cost
        
        logger.info(f"Request cost: ${request_cost:.6f} | Total: ${self.total_cost:.4f}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cost statistics"""
        session_duration = (datetime.now() - self.session_start).total_seconds() / 3600
        
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_requests": self.total_requests,
            "total_cost_usd": round(self.total_cost, 4),
            "session_duration_hours": round(session_duration, 2),
            "avg_cost_per_request": round(self.total_cost / max(1, self.total_requests), 6),
            "tokens_per_hour": round((self.total_input_tokens + self.total_output_tokens) / max(0.1, session_duration)),
            "estimated_monthly_cost": round(self.total_cost * (720 / max(0.1, session_duration)), 2)
        }

class BedrockHaikuClient:
    """Ultra-fast, never-fail AWS Bedrock Claude 3 Haiku client"""
    
    def __init__(self, config: BedrockConfig):
        self.config = config
        self.circuit_breaker = CircuitBreaker()
        self.cost_tracker = CostTracker()
        self._client = None
        self._cache = {}  # Simple response cache
        
        # Initialize AWS client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize AWS Bedrock client with optimized configuration"""
        try:
            # Optimized boto3 configuration for performance
            boto_config = Config(
                region_name=self.config.aws_region,
                retries={'max_attempts': 0},  # We handle retries manually
                max_pool_connections=50,
                read_timeout=self.config.timeout,
                connect_timeout=10
            )
            
            self._client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key,
                config=boto_config
            )
            
            logger.info("✅ Bedrock Claude 3 Haiku client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Bedrock client: {e}")
            raise Exception(f"Bedrock initialization failed: {e}")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Claude 3 Haiku: roughly 1 token per 4 characters
        return len(text) // 4
    
    def _generate_cache_key(self, prompt: str, system: str = None) -> str:
        """Generate cache key for request"""
        content = f"{system or ''}{prompt}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def _make_request_with_retry(
        self, 
        prompt: str, 
        system: str = None,
        max_tokens: int = None
    ) -> Tuple[str, int, int]:
        """Make request with exponential backoff retry"""
        
        if not self.circuit_breaker.can_execute(self.config):
            raise Exception("Circuit breaker is OPEN - service temporarily unavailable")
        
        # Check cache first
        cache_key = self._generate_cache_key(prompt, system)
        if cache_key in self._cache:
            logger.info("Cache hit - returning cached response")
            cached = self._cache[cache_key]
            return cached['response'], cached['input_tokens'], cached['output_tokens']
        
        # Prepare request
        messages = []
        if system:
            messages.append({"role": "user", "content": f"System: {system}\n\nUser: {prompt}"})
        else:
            messages.append({"role": "user", "content": prompt})
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "messages": messages
        }
        
        # Retry logic with exponential backoff
        last_exception = None
        delay = self.config.base_delay
        
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                
                # Make the request
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.invoke_model(
                        modelId=self.config.model_id,
                        body=json.dumps(body)
                    )
                )
                
                # Parse response
                response_body = json.loads(response['body'].read())
                content = response_body['content'][0]['text']
                
                # Calculate token usage
                input_tokens = self.estimate_tokens(prompt + (system or ''))
                output_tokens = self.estimate_tokens(content)
                
                # Track costs
                self.cost_tracker.add_usage(input_tokens, output_tokens, self.config)
                
                # Record success
                self.circuit_breaker.record_success()
                
                # Cache response
                self._cache[cache_key] = {
                    'response': content,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens
                }
                
                duration = time.time() - start_time
                logger.info(f"✅ Bedrock request successful in {duration:.2f}s (attempt {attempt + 1})")
                
                return content, input_tokens, output_tokens
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                last_exception = e
                
                if error_code == 'ThrottlingException':
                    logger.warning(f"Rate limited on attempt {attempt + 1}, retrying in {delay}s")
                elif error_code == 'ValidationException':
                    logger.error(f"Validation error: {e}")
                    self.circuit_breaker.record_failure(self.config)
                    raise Exception(f"Invalid request: {e}")
                else:
                    logger.warning(f"AWS error on attempt {attempt + 1}: {error_code}")
                
                if attempt == self.config.max_retries - 1:
                    self.circuit_breaker.record_failure(self.config)
                    break
                
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.config.max_delay)
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Request failed on attempt {attempt + 1}: {e}")
                
                if attempt == self.config.max_retries - 1:
                    self.circuit_breaker.record_failure(self.config)
                    break
                
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.config.max_delay)
        
        # All retries failed
        error_msg = f"All {self.config.max_retries} attempts failed. Last error: {last_exception}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def _intelligent_chunk_text(self, text: str) -> List[str]:
        """Intelligently chunk text preserving context"""
        
        estimated_tokens = self.estimate_tokens(text)
        
        if estimated_tokens <= self.config.max_input_tokens:
            return [text]
        
        logger.info(f"Text too large ({estimated_tokens} tokens), chunking...")
        
        # Split into sentences for better chunking
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback to simple splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.config.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.config.chunk_overlap)
                current_chunk = overlap_text + " " + sentence
                current_tokens = self.estimate_tokens(current_chunk)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        logger.info(f"Created {len(chunks)} chunks for processing")
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get overlap text from end of chunk"""
        words = text.split()
        overlap_words = overlap_tokens // 4  # Rough estimate
        
        if len(words) <= overlap_words:
            return text
        
        return " ".join(words[-overlap_words:])
    
    async def _process_chunks_parallel(
        self, 
        chunks: List[str], 
        system: str = None,
        task_type: str = "summarize"
    ) -> List[str]:
        """Process chunks in parallel with intelligent batching"""
        
        if len(chunks) == 1:
            response, _, _ = await self._make_request_with_retry(chunks[0], system)
            return [response]
        
        # Create chunk-specific prompts
        chunk_prompts = []
        for i, chunk in enumerate(chunks):
            if task_type == "summarize":
                chunk_prompt = f"Summarize this section (part {i+1} of {len(chunks)}):\n\n{chunk}"
            elif task_type == "extract_insights":
                chunk_prompt = f"Extract key insights from this section (part {i+1} of {len(chunks)}):\n\n{chunk}"
            elif task_type == "generate_tags":
                chunk_prompt = f"Generate relevant tags for this section (part {i+1} of {len(chunks)}):\n\n{chunk}"
            elif task_type == "analyze_demographics":
                chunk_prompt = f"Analyze demographics in this section (part {i+1} of {len(chunks)}):\n\n{chunk}"
            else:
                chunk_prompt = f"Process this section (part {i+1} of {len(chunks)}):\n\n{chunk}"
            
            chunk_prompts.append(chunk_prompt)
        
        # Process chunks in parallel batches
        batch_size = min(self.config.max_chunks_parallel, len(chunk_prompts))
        results = []
        
        for i in range(0, len(chunk_prompts), batch_size):
            batch = chunk_prompts[i:i + batch_size]
            
            # Create tasks for parallel processing
            tasks = []
            for prompt in batch:
                task = self._make_request_with_retry(prompt, system)
                tasks.append(task)
            
            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Chunk {i+j+1} failed: {result}")
                    results.append(f"[Error processing chunk {i+j+1}: {result}]")
                else:
                    response, _, _ = result
                    results.append(response)
        
        return results
    
    def _synthesize_chunk_results(self, chunk_results: List[str], task_type: str) -> str:
        """Synthesize results from multiple chunks"""
        
        if len(chunk_results) == 1:
            return chunk_results[0]
        
        # Combine results with appropriate synthesis prompt
        combined_text = "\n\n---\n\n".join([
            f"Section {i+1}: {result}" 
            for i, result in enumerate(chunk_results)
            if not result.startswith("[Error")
        ])
        
        if task_type == "summarize":
            synthesis_prompt = f"""Synthesize these section summaries into a comprehensive executive summary:

{combined_text}

Create a cohesive summary that captures all key insights and maintains the original structure and depth."""
        
        elif task_type == "extract_insights":
            synthesis_prompt = f"""Combine these insights from different sections into a unified analysis:

{combined_text}

Provide a comprehensive insight analysis that integrates all findings."""
        
        elif task_type == "generate_tags":
            synthesis_prompt = f"""Combine and deduplicate these tags from different sections:

{combined_text}

Provide a final list of the most relevant and comprehensive tags."""
        
        elif task_type == "analyze_demographics":
            synthesis_prompt = f"""Synthesize these demographic analyses into a comprehensive profile:

{combined_text}

Provide a unified demographic analysis that combines all findings."""
        
        else:
            synthesis_prompt = f"""Synthesize these results into a comprehensive response:

{combined_text}

Provide a unified and coherent final result."""
        
        return synthesis_prompt
    
    async def chat(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = None,
        task_type: str = "general"
    ) -> str:
        """Main chat interface with unlimited token support"""
        
        start_time = time.time()
        estimated_tokens = self.estimate_tokens(prompt + (system or ''))
        
        logger.info(f"Processing request: {estimated_tokens} tokens, task: {task_type}")
        
        try:
            # Handle large documents with chunking
            if estimated_tokens > self.config.max_input_tokens:
                logger.info("Using chunking strategy for large document")
                
                # Chunk the text
                chunks = self._intelligent_chunk_text(prompt)
                
                # Process chunks in parallel
                chunk_results = await self._process_chunks_parallel(chunks, system, task_type)
                
                # Synthesize results
                synthesis_prompt = self._synthesize_chunk_results(chunk_results, task_type)
                
                # Get final synthesized result
                final_result, _, _ = await self._make_request_with_retry(
                    synthesis_prompt, 
                    system, 
                    max_tokens
                )
                
                duration = time.time() - start_time
                logger.info(f"✅ Large document processed in {duration:.2f}s using {len(chunks)} chunks")
                
                return final_result
            
            else:
                # Direct processing for smaller documents
                result, _, _ = await self._make_request_with_retry(prompt, system, max_tokens)
                
                duration = time.time() - start_time
                logger.info(f"✅ Direct processing completed in {duration:.2f}s")
                
                return result
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Request failed after {duration:.2f}s: {e}")
            raise e
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get client health status"""
        return {
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "failure_count": self.circuit_breaker.failure_count,
            "last_failure": self.circuit_breaker.last_failure_time.isoformat() if self.circuit_breaker.last_failure_time else None,
            "cost_stats": self.cost_tracker.get_stats(),
            "cache_size": len(self._cache),
            "model_id": self.config.model_id,
            "region": self.config.aws_region
        }
    
    def clear_cache(self):
        """Clear response cache"""
        self._cache.clear()
        logger.info("Response cache cleared")
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Get detailed cost report"""
        return self.cost_tracker.get_stats()

# Global client instance (will be initialized when needed)
_bedrock_client: Optional[BedrockHaikuClient] = None

def get_bedrock_client(config: BedrockConfig = None) -> BedrockHaikuClient:
    """Get or create Bedrock client instance"""
    global _bedrock_client
    
    if _bedrock_client is None:
        if config is None:
            raise Exception("BedrockConfig required for first initialization")
        _bedrock_client = BedrockHaikuClient(config)
    
    return _bedrock_client

async def test_bedrock_connection(config: BedrockConfig) -> Dict[str, Any]:
    """Test Bedrock connection and performance"""
    try:
        client = BedrockHaikuClient(config)
        
        start_time = time.time()
        response = await client.chat("Hello! Please respond with 'Connection successful'")
        duration = time.time() - start_time
        
        return {
            "success": True,
            "response": response,
            "duration": duration,
            "health_status": client.get_health_status()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "duration": time.time() - start_time if 'start_time' in locals() else 0
        }

# Export main classes and functions
__all__ = [
    'BedrockConfig',
    'BedrockHaikuClient', 
    'get_bedrock_client',
    'test_bedrock_connection',
    'CircuitBreakerState',
    'CostTracker'
]
