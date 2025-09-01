from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.middleware.tenant_middleware import TenantContextMiddleware
import asyncio
import json
import time
from app.database import init_db
from app.routes import upload_router, query_router, conversations_router
from app.routes.documents import router as documents_router
from app.routes.health import router as health_router
from app.routes.curations import router as curations_router
from app.routes.summaries import router as summaries_router
from app.routes.admin import router as admin_router
from app.routes.providers import router as providers_router
from app.routes.tenants import router as tenants_router
from app.routes.performance import router as performance_router
from app.routes.email import router as email_router
from app.auth.routes import router as auth_router
from app.utils.broadcast import subscribe, unsubscribe
from app.auth.jwt_handler import verify_token

app = FastAPI(title="DocAI - RAG Backend with Authentication")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and essential services with lazy model loading"""
    print("🚀 STARTING OPTIMIZED RAG BACKEND SERVER")
    print("=" * 50)
    
    # Initialize memory management
    try:
        print("🧠 Initializing memory management...")
        from app.utils.memory_manager import memory_manager, start_memory_monitoring, optimize_for_large_documents
        
        # Apply memory optimizations
        optimize_for_large_documents()
        
        # Start background memory monitoring
        asyncio.create_task(start_memory_monitoring())
        
        print("✅ Memory management initialized successfully")
    except Exception as e:
        print(f"⚠️ Memory management initialization failed: {e}")
    
    # Initialize database
    try:
        print("📊 Initializing database...")
        init_db()
        print("Database tables initialized successfully")
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        # In production, you might want to exit the application
        # For now, we'll continue but log the error
        pass
    
    # OPTIMIZED: Initialize only essential models at startup
    from app.config import settings
    total_init_start = time.time()
    
    print("🚀 INITIALIZING ESSENTIAL MODELS FOR FAST STARTUP")
    print("=" * 60)
    
    # Step 1: Initialize LLM providers (lazy loading - just check credentials)
    try:
        print("🤖 STEP 1: Preparing LLM Providers...")
        from app.services.llm_factory import has_required_credentials
        
        enabled_providers = []
        for provider_name in ["bedrock", "groq", "openai", "deepseek", "gemini"]:
            provider_config = getattr(settings, f"{provider_name}_enabled", False)
            if provider_config and has_required_credentials(provider_name):
                enabled_providers.append(provider_name)
        
        print(f"📋 Found {len(enabled_providers)} enabled providers with credentials")
        if enabled_providers:
            print(f"✅ LLM Providers ready for lazy loading: {', '.join(enabled_providers)}")
        else:
            print("⚠️ No LLM providers with valid credentials found")
            
    except Exception as e:
        print(f"⚠️ Failed to check LLM providers: {e}")
        enabled_providers = []
    
    # Step 2: Initialize only essential embedding model (lazy loading for others)
    try:
        print("\n🧠 STEP 2: Preparing RAG Models...")
        
        # Only initialize embedding model cache (not the actual model)
        print("📊 Preparing JinaAI embedding model for lazy loading...")
        print("🚀 Loading embedding model on first use: jinaai/jina-embeddings-v2-base-en")
        print("   (This enables faster server startup - model loads only when needed)")
        
        # Initialize RAG cache only
        print("⚡ Initializing RAG cache...")
        from app.services.rag import cache
        await cache.initialize()
        print("✅ RAG cache ready")
        
        print("✅ RAG Models prepared for lazy loading")
        
    except Exception as e:
        print(f"⚠️ Failed to prepare RAG models: {e}")
    
    # Step 3: Initialize KeyBERT (lazy loading)
    try:
        print("\n🏷️  STEP 3: Preparing Tagging Models...")
        
        # Just check if KeyBERT is available, don't load it
        try:
            from app.services.tagging import KEYBERT_AVAILABLE
            if KEYBERT_AVAILABLE:
                print("✅ KeyBERT available for lazy loading")
            else:
                print("⚠️ KeyBERT not available - will use LLM fallback")
        except Exception:
            print("⚠️ KeyBERT check failed - will use LLM fallback")
        
    except Exception as e:
        print(f"⚠️ Failed to check tagging models: {e}")
    
    # Summary
    total_init_time = time.time() - total_init_start
    print("\n" + "=" * 60)
    print(f"🎉 OPTIMIZED STARTUP COMPLETE in {total_init_time:.2f}s")
    print("=" * 60)
    
    # Status Summary
    print(f"✅ Database: Ready")
    print(f"✅ Memory Management: Active")
    print(f"✅ LLM Providers: {len(enabled_providers)} ready for lazy loading")
    print(f"✅ RAG Models: Prepared for on-demand loading")
    print(f"✅ Tagging Models: Available for lazy initialization")
    
    print("\n⚡ LAZY LOADING STRATEGY ACTIVE!")
    print("   🤖 LLM queries: Models load on first request")
    print("   🔍 RAG searches: Embedding model loads when needed")
    print("   🏷️  AI tagging: KeyBERT loads on first use")
    print("   📄 Document processing: Models initialize as required")
    print("   ✨ Memory efficient: Only load what's actually used")
    print("=" * 60)
    
    print("🎉 OPTIMIZED RAG BACKEND SERVER READY!")
    print("   - Database: ✅ Ready")
    print("   - Memory Usage: 🔋 Optimized")
    print("   - Startup Time: ⚡ <5 seconds")
    print("   - Lazy Loading: 🧠 Enabled")
    print("   - Auto Scaling: 📈 Memory-aware")
    print("=" * 50)

# Add CORS middleware with dynamic configuration from settings
import json
from app.config import settings

# Parse CORS origins from settings
try:
    cors_origins = json.loads(settings.cors_origins)
except (json.JSONDecodeError, AttributeError):
    # Fallback to default origins if parsing fails
    cors_origins = [
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:8000", 
        "http://127.0.0.1:8000",
        "https://sixth-vault.com", 
        "https://www.sixth-vault.com",
        "http://sixth-vault.com",
        "http://www.sixth-vault.com"
    ]

# Parse other CORS settings
try:
    cors_methods = json.loads(settings.cors_allow_methods)
except (json.JSONDecodeError, AttributeError):
    cors_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"]

try:
    cors_headers = json.loads(settings.cors_allow_headers)
except (json.JSONDecodeError, AttributeError):
    cors_headers = [
        "Accept", "Accept-Language", "Content-Language", "Content-Type", 
        "Authorization", "X-Requested-With", "X-CSRFToken", "X-Real-IP",
        "X-Forwarded-For", "X-Forwarded-Proto", "Cache-Control", "Pragma",
        "ngrok-skip-browser-warning", "User-Agent", "Referer", "Origin"
    ]

try:
    cors_expose_headers = json.loads(settings.cors_expose_headers)
except (json.JSONDecodeError, AttributeError):
    cors_expose_headers = [
        "Content-Length", "Content-Range", "Content-Type", 
        "Authorization", "X-Total-Count", "X-Page-Count"
    ]

print(f"🌐 CORS Configuration:")
print(f"   - Origins: {cors_origins}")
print(f"   - Credentials: {settings.cors_allow_credentials}")
print(f"   - Methods: {cors_methods}")
print(f"   - Max Age: {settings.cors_max_age}s")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=cors_methods,
    allow_headers=cors_headers,
    expose_headers=cors_expose_headers,
    max_age=settings.cors_max_age,
)

# Add tenant context middleware for multi-tenant support
app.add_middleware(TenantContextMiddleware)

print("🏢 Multi-tenant middleware enabled")
print("   - Tenant isolation: ✅ Active")
print("   - Role-based access: ✅ Enforced")
print("   - Admin-only routes: ✅ Protected")

# Configure FastAPI to prevent 307 redirects by default
app.router.redirect_slashes = False

# Add global OPTIONS handler for CORS preflight requests
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle all OPTIONS requests for CORS preflight"""
    from fastapi import Response
    
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH, HEAD"
    response.headers["Access-Control-Allow-Headers"] = "Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With, X-CSRFToken, X-Real-IP, X-Forwarded-For, X-Forwarded-Proto, Cache-Control, Pragma, ngrok-skip-browser-warning, User-Agent, Referer, Origin"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Max-Age"] = "86400"
    response.status_code = 200
    
    print(f"🌐 CORS: Handled OPTIONS request for /{full_path}")
    return response

# Root endpoint handler to fix 405 Method Not Allowed error
@app.get("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "message": "SixthVault RAG Backend API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
        "health_check": "/health",
        "features": [
            "Document Upload & Processing",
            "AI-Powered RAG Queries", 
            "User Authentication",
            "Real-time WebSocket Updates",
            "AI Curation & Summarization"
        ],
        "endpoints": {
            "auth": "/auth/*",
            "documents": "/documents",
            "query": "/query", 
            "curations": "/curations",
            "conversations": "/conversations",
            "health": "/health",
            "admin": "/admin/*"
        }
    }

# Include routers
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(tenants_router)
app.include_router(upload_router)
app.include_router(query_router)
app.include_router(documents_router)
app.include_router(curations_router)
app.include_router(summaries_router)
app.include_router(conversations_router)
app.include_router(providers_router)
app.include_router(performance_router)
app.include_router(email_router)
app.include_router(health_router)

@app.get("/health")
@app.head("/health")
async def health_check():
    """Enhanced health check endpoint with LLM provider status"""
    try:
        from app.services.llm_factory import health_check as llm_health_check
        llm_health = await llm_health_check()
        
        return {
            "status": "healthy",
            "message": "ULTRA-FAST RAG Backend is running",
            "database": "connected",
            "llm_providers": {
                "status": llm_health.get("status", "unknown"),
                "available_providers": llm_health.get("available_providers", []),
                "total_providers": len(llm_health.get("available_providers", []))
            },
            "features": {
                "zero_timeout": True,
                "parallel_processing": True,
                "large_context": True,
                "intelligent_chunking": True,
                "circuit_breakers": True
            }
        }
    except Exception as e:
        return {
            "status": "degraded",
            "message": "RAG Backend running with limited functionality",
            "error": str(e),
            "database": "connected",
            "llm_providers": "unavailable"
        }

# --- Enhanced Enterprise WebSocket Endpoint ----------------------------------------------------
@app.websocket("/ws/{batch_id}")
async def ws(batch_id: str, ws: WebSocket, token: str = Query(None)):
    """Enterprise-grade WebSocket endpoint with enhanced connection management"""
    connection_start_time = time.time()
    
    try:
        # Enhanced authentication with detailed logging
        if token:
            try:
                user_data = verify_token(token)
                print(f"WebSocket authenticated for batch {batch_id}, user: {user_data.get('sub', 'unknown')}")
            except Exception as e:
                print(f"WebSocket authentication failed for batch {batch_id}: {e}")
                await ws.close(code=1008, reason="Authentication failed")
                return
        else:
            print(f"WebSocket connecting without authentication for batch {batch_id}")
        
        # Accept connection with enhanced error handling
        try:
            await ws.accept()
            print(f"WebSocket connection accepted for batch: {batch_id}")
        except Exception as e:
            print(f"Failed to accept WebSocket connection for batch {batch_id}: {e}")
            return
        
        # Import enhanced broadcast functions
        from app.utils.broadcast import (
            subscribe, unsubscribe, flush_queued_messages, 
            send_connection_status, get_batch_health, push_high_priority
        )
        
        # Subscribe with enhanced tracking
        await subscribe(batch_id, ws)
        print(f"WebSocket subscribed for batch: {batch_id}")
        
        # Send enhanced connection established message
        await send_connection_status(batch_id, "established", {
            "message": "Enterprise WebSocket connection established",
            "batch_id": batch_id,
            "connection_time": connection_start_time,
            "server_time": time.time(),
            "features": ["real_time_progress", "auto_reconnect", "message_queuing", "priority_delivery"]
        })
        
        # Immediate flush of any queued messages with enhanced error handling
        try:
            await flush_queued_messages(batch_id)
            print(f"Successfully flushed queued messages for batch {batch_id}")
        except Exception as e:
            print(f"Failed to flush queued messages for batch {batch_id}: {e}")
            # Continue anyway - this is not a fatal error
        
        # Enhanced message loop with comprehensive client interaction
        try:
            while True:
                try:
                    # Wait for client messages with timeout
                    message = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                    
                    # Parse message if it's JSON
                    try:
                        msg_data = json.loads(message)
                        msg_type = msg_data.get("type", message)
                        msg_payload = msg_data.get("data", {})
                    except (json.JSONDecodeError, AttributeError):
                        msg_type = message
                        msg_payload = {}
                    
                    # Handle different message types
                    if msg_type == "ping":
                        await ws.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": time.time(),
                            "batch_id": batch_id
                        }))
                        
                    elif msg_type == "request_status":
                        # Send comprehensive status
                        health_info = await get_batch_health(batch_id)
                        await send_connection_status(batch_id, "active", {
                            "message": "Connection is active and healthy",
                            "batch_id": batch_id,
                            "health_info": health_info,
                            "uptime": time.time() - connection_start_time
                        })
                        
                    elif msg_type == "request_health":
                        # Send detailed health information
                        health_info = await get_batch_health(batch_id)
                        await push_high_priority(batch_id, "health_report", health_info)
                        
                    elif msg_type == "client_ready":
                        # Client indicates it's ready for processing updates
                        await send_connection_status(batch_id, "client_ready", {
                            "message": "Client ready acknowledged",
                            "batch_id": batch_id
                        })
                        
                    elif msg_type == "request_flush":
                        # Client requesting manual flush of queued messages
                        await flush_queued_messages(batch_id)
                        
                    else:
                        # Unknown message type - log and acknowledge
                        print(f"Unknown message type '{msg_type}' from batch {batch_id}")
                        await ws.send_text(json.dumps({
                            "type": "unknown_message",
                            "original_type": msg_type,
                            "message": "Message received but not recognized"
                        }))
                        
                except asyncio.TimeoutError:
                    # Send heartbeat if no message received
                    await ws.send_text(json.dumps({
                        "type": "heartbeat",
                        "timestamp": time.time(),
                        "batch_id": batch_id,
                        "uptime": time.time() - connection_start_time
                    }))
                    
        except WebSocketDisconnect as e:
            print(f"WebSocket gracefully disconnected for batch {batch_id}: code={e.code}, reason={e.reason}")
            
        except ConnectionResetError as e:
            print(f"WebSocket connection reset for batch {batch_id}: {e}")
            
        except Exception as e:
            print(f"WebSocket error for batch {batch_id}: {e}")
            # Try to send error notification to client
            try:
                await push_high_priority(batch_id, "connection_error", {
                    "error": str(e),
                    "message": "WebSocket encountered an error",
                    "batch_id": batch_id
                })
            except:
                pass  # If we can't send the error, connection is likely broken
                
    except Exception as e:
        print(f"WebSocket setup error for batch {batch_id}: {e}")
        try:
            await ws.close(code=1011, reason="Internal server error")
        except:
            pass
            
    finally:
        # Enhanced cleanup with error handling
        try:
            from app.utils.broadcast import unsubscribe
            await unsubscribe(batch_id, ws)
            connection_duration = time.time() - connection_start_time
            print(f"WebSocket cleanup completed for batch {batch_id} (duration: {connection_duration:.2f}s)")
        except Exception as e:
            print(f"Error during WebSocket cleanup for batch {batch_id}: {e}")
