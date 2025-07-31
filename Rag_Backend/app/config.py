from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Security
    jwt_secret: str

    # Database
    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: str
    postgres_db: str

    # Storage
    upload_dir: Path = Path("uploads")
    qdrant_local_path: Path = Path("qdrant_local")

    # AWS S3 Configuration
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "eu-north-1"
    s3_bucket_name: str = "sixthvaultbucket"
    s3_signed_url_expiry: int = 3600  # 1 hour
    s3_use_ssl: bool = True
    s3_endpoint_url: str | None = None  # For custom S3-compatible services

    # External Qdrant (optional)
    qdrant_remote_url: str | None = None
    qdrant_api_key: str | None = None

    # LLM API Keys
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    deepseek_api_key: str | None = None
    
    # AWS Bedrock Configuration - Claude 3 Haiku
    aws_bedrock_access_key_id: str | None = None
    aws_bedrock_secret_access_key: str | None = None
    aws_bedrock_region: str = "ap-south-1"
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    bedrock_max_tokens: int = 4096
    bedrock_temperature: float = 0.1
    bedrock_enabled: bool = True  # Default to True for document processing
    
    # Ollama Configuration - ULTRA-FAST PARALLEL PROCESSING SETTINGS
    ollama_enabled: bool = False  # Master switch to enable/disable Ollama functionality
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: int = 0  # NO TIMEOUT - Wait indefinitely for output
    ollama_max_retries: int = 2  # Reduced retries for speed
    ollama_default_model: str = "llama3.2:3b"  # FASTER 3B model
    ollama_fallback_model: str = "phi3:mini"  # Fast fallback
    ollama_enable_gpu: bool = True  # Enable GPU acceleration
    ollama_num_ctx: int = 8192  # OPTIMIZED to 8K for faster processing
    ollama_num_predict: int = 512  # OPTIMIZED to 512 for faster responses
    ollama_temperature: float = 0.1  # Lower temperature for consistent outputs
    ollama_top_p: float = 0.9
    ollama_repeat_penalty: float = 1.1
    ollama_num_thread: int = 8  # Optimized threading for parallel tasks
    ollama_num_gpu: int = 1  # Use single GPU for stability
    ollama_batch_size: int = 128  # OPTIMIZED batch size for parallel processing
    ollama_concurrent_requests: int = 32  # DOUBLED for true parallel processing
    
    # PERFORMANCE OPTIMIZATIONS
    # Document processing limits
    max_document_tokens: int = 50000  # Limit for ultra-fast processing
    max_chunks_per_document: int = 25  # Reduced from 80 for speed
    chunk_size_small: int = 600  # Optimized chunk sizes
    chunk_size_medium: int = 800
    chunk_size_large: int = 1000
    chunk_overlap: int = 100  # Reduced overlap for speed
    
    # AI processing timeouts - DISABLED FOR GUARANTEED OUTPUT
    ai_task_timeout: int = 0  # NO TIMEOUT - Wait for completion
    tagging_timeout: int = 0  # NO TIMEOUT for tagging
    demographics_timeout: int = 0   # NO TIMEOUT for demographics
    summary_timeout: int = 0  # NO TIMEOUT for summary
    insights_timeout: int = 0  # NO TIMEOUT for insights
    
    # RAG optimization settings - NO TIMEOUTS
    rag_retrieval_timeout: int = 0  # NO TIMEOUT for retrieval
    rag_generation_timeout: int = 0  # NO TIMEOUT for generation
    rag_max_chunks: int = 15  # Reduced chunks for speed
    rag_rerank_timeout: int = 0  # NO TIMEOUT for reranking
    
    # Caching and memory optimization
    enable_aggressive_caching: bool = True
    cache_ttl: int = 1800  # 30 minutes
    memory_limit_mb: int = 2048  # 2GB memory limit
    enable_streaming: bool = True  # Enable streaming responses
    
    # Email service
    resend_api_key: str | None = None
    email_domain: str = "sixth-vault.com"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields in .env file
    )

settings = Settings()

# create folders on first import
settings.upload_dir.mkdir(exist_ok=True, parents=True)
settings.qdrant_local_path.mkdir(exist_ok=True, parents=True)
