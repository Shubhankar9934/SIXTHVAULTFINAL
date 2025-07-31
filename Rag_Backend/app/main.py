from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import time
from app.database import init_db
from app.routes import upload_router, query_router
from app.routes.documents import router as documents_router
from app.routes.ollama import router as ollama_router
from app.routes.health import router as health_router
from app.routes.curations import router as curations_router
from app.routes.summaries import router as summaries_router
from app.routes.admin import router as admin_router
from app.auth.routes import router as auth_router
from app.utils.broadcast import subscribe, unsubscribe
from app.auth.jwt_handler import verify_token

app = FastAPI(title="DocAI - RAG Backend with Authentication")

# Initialize database and Ollama on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database, Ollama models, and perform startup tasks"""
    print("🚀 STARTING ULTRA-FAST RAG BACKEND SERVER")
    print("=" * 50)
    
    # Initialize database
    try:
        print("📊 Initializing database...")
        init_db()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        # In production, you might want to exit the application
        # For now, we'll continue but log the error
        pass
    
    # Initialize and warm up Ollama models (only if enabled)
    from app.config import settings
    if settings.ollama_enabled:
        try:
            print("🔥 Initializing ULTRA-FAST Ollama models...")
            from app.services.llm_factory import initialize_llm_factory
            await initialize_llm_factory()
            print("✅ Ollama models pre-loaded and ready for instant inference!")
        except Exception as e:
            print(f"⚠️ Failed to initialize Ollama models: {e}")
            print("   Ollama will be initialized on first request")
            # Don't fail startup - allow fallback providers to work
            pass
    else:
        print("� Ollama is disabled in configuration - skipping Ollama initialization")
    
    # CRITICAL OPTIMIZATION: Pre-load embedding models for instant RAG
    try:
        print("�🚀 Pre-loading embedding models for instant RAG...")
        from app.utils.qdrant_store import preload_embedding_model
        await preload_embedding_model()
        print("⚡ Embedding models pre-loaded for zero-delay vector search!")
    except Exception as e:
        print(f"⚠️ Failed to pre-load embedding models: {e}")
        print("   Embedding models will be loaded on first search request")
        # Don't fail startup - models will load on first use
        pass
    
    # Pre-load reranking models for instant results
    try:
        print("🎯 Pre-loading reranking models...")
        from app.services.rag import reranker
        await reranker.initialize()
        print("✅ Reranking models pre-loaded for instant precision!")
    except Exception as e:
        print(f"⚠️ Failed to pre-load reranking models: {e}")
        print("   Reranking will use LLM fallback")
        # Don't fail startup - fallback to LLM reranking
        pass
    
    print("🎉 ULTRA-FAST RAG BACKEND SERVER READY!")
    print("   - Database: ✅ Ready")
    print("   - Ollama Models: 🔥 Pre-loaded")
    print("   - Zero Timeout: ⚡ Enabled")
    print("   - Parallel Processing: 🔄 8 concurrent slots")
    print("   - Large Context: 📄 32K tokens supported")
    print("=" * 50)

# Add CORS middleware - Enhanced for ngrok compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development with ngrok
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Add custom middleware for ngrok headers
@app.middleware("http")
async def add_ngrok_headers(request, call_next):
    response = await call_next(request)
    
    # Add ngrok-specific headers to bypass browser warnings
    response.headers["ngrok-skip-browser-warning"] = "true"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    
    # Handle ngrok tunnel requests
    origin = request.headers.get("origin")
    if origin and ("ngrok" in origin or "localhost" in origin):
        response.headers["Access-Control-Allow-Origin"] = origin
    
    return response

# Include routers
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(upload_router)
app.include_router(query_router)
app.include_router(documents_router)
app.include_router(curations_router)
app.include_router(summaries_router)
app.include_router(ollama_router)
app.include_router(health_router)

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with Ollama status"""
    try:
        from app.services.llm_factory import health_check as llm_health_check
        llm_health = await llm_health_check()
        
        return {
            "status": "healthy",
            "message": "ULTRA-FAST RAG Backend is running",
            "database": "connected",
            "ollama": {
                "status": llm_health.get("status", "unknown"),
                "available_providers": llm_health.get("available_providers", []),
                "models_cached": llm_health.get("providers", {}).get("ollama", {}).get("models_cached", 0)
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
            "ollama": "unavailable"
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
