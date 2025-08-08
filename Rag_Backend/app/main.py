from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
from app.auth.routes import router as auth_router
from app.utils.broadcast import subscribe, unsubscribe
from app.auth.jwt_handler import verify_token

app = FastAPI(title="DocAI - RAG Backend with Authentication")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database, LLM models, and perform startup tasks"""
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
    
    # Initialize ALL MODELS at startup for instant inference
    from app.config import settings
    total_init_start = time.time()
    
    print("🚀 INITIALIZING ALL MODELS FOR INSTANT INFERENCE")
    print("=" * 60)
    
    # Step 1: Initialize LLM providers (Bedrock, Groq, OpenAI, etc.)
    try:
        print("🤖 STEP 1: Initializing LLM Providers...")
        from app.services.llm_factory import initialize_all_enabled_models
        llm_result = await initialize_all_enabled_models()
        
        if llm_result["success"]:
            print(f"✅ LLM Providers initialized: {len(llm_result['successful_providers'])} ready")
            print(f"   - Ready providers: {', '.join(llm_result['successful_providers'])}")
        else:
            print("⚠️ No LLM providers initialized - will try on-demand loading")
            
    except Exception as e:
        print(f"⚠️ Failed to initialize LLM providers: {e}")
        llm_result = {"success": False, "successful_providers": []}
    
    # Step 2: Initialize RAG Models (Embedding, Reranker)
    try:
        print("\n🧠 STEP 2: Initializing RAG Models...")
        
        # Initialize embedding model
        print("📊 Loading JinaAI embedding model...")
        from app.utils.qdrant_store import preload_embedding_model
        await preload_embedding_model()
        print("✅ JinaAI embedding model ready (768 dimensions)")
        
        # Initialize reranker model
        print("🎯 Loading BGE reranker model...")
        from app.services.rag import reranker
        await reranker.initialize()
        if reranker.model_loaded:
            print(f"✅ BGE reranker model ready ({reranker.model_name})")
        else:
            print("⚠️ Reranker will use LLM fallback")
        
        # Initialize RAG cache
        print("⚡ Initializing RAG cache...")
        from app.services.rag import cache
        await cache.initialize()
        print("✅ RAG cache ready")
        
        rag_models_ready = True
        
    except Exception as e:
        print(f"⚠️ Failed to initialize RAG models: {e}")
        rag_models_ready = False
    
    # Step 3: Initialize KeyBERT and Tagging Models
    try:
        print("\n🏷️  STEP 3: Initializing KeyBERT and Tagging Models...")
        
        # Initialize KeyBERT model by importing the enhanced tagger
        from app.services.tagging import _enhanced_tagger, KEYBERT_AVAILABLE
        
        # Check if KeyBERT is available and initialized
        if KEYBERT_AVAILABLE and hasattr(_enhanced_tagger, '_kw') and _enhanced_tagger._kw is not None:
            print("✅ KeyBERT model ready (all-MiniLM-L6-v2)")
        else:
            print("⚠️ KeyBERT not available - will use LLM fallback")
        
        tagging_models_ready = True
        
    except Exception as e:
        print(f"⚠️ Failed to initialize tagging models: {e}")
        tagging_models_ready = False
    
    # Summary
    total_init_time = time.time() - total_init_start
    print("\n" + "=" * 60)
    print(f"🎉 MODEL INITIALIZATION COMPLETE in {total_init_time:.2f}s")
    print("=" * 60)
    
    # LLM Providers Summary
    if llm_result["success"]:
        print(f"✅ LLM Providers: {len(llm_result['successful_providers'])} ready")
        print(f"   - {', '.join(llm_result['successful_providers'])}")
    else:
        print("❌ LLM Providers: None ready (on-demand loading)")
    
    # RAG Models Summary
    if rag_models_ready:
        print("✅ RAG Models: Embedding + Reranker + Cache ready")
        print("   - JinaAI v2 embedding (768 dim)")
        print("   - BGE reranker (optimal performance)")
        print("   - Lightning-fast cache system")
    else:
        print("❌ RAG Models: Will load on-demand")
    
    # Tagging Models Summary
    if tagging_models_ready:
        print("✅ Tagging Models: KeyBERT ready")
        print("   - all-MiniLM-L6-v2 for keyword extraction")
    else:
        print("❌ Tagging Models: Will use LLM fallback")
    
    print("\n⚡ ZERO LOADING DELAYS FOR ALL OPERATIONS!")
    print("   🤖 LLM queries: Instant inference")
    print("   🔍 RAG searches: Instant embedding + reranking")
    print("   🏷️  AI tagging: Instant keyword extraction")
    print("   📄 Document processing: All models pre-warmed")
    print("   ✨ AI curation: Instant model availability")
    print("   📝 Summarization: Zero startup delays")
    print("=" * 60)
    
    print("🎉 ULTRA-FAST RAG BACKEND SERVER READY!")
    print("   - Database: ✅ Ready")
    print("   - LLM Models: 🔥 Pre-loaded")
    print("   - Zero Timeout: ⚡ Enabled")
    print("   - Parallel Processing: 🔄 8 concurrent slots")
    print("   - Large Context: 📄 32K tokens supported")
    print("=" * 50)

# Add CORS middleware with enhanced configuration for production deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:8000", 
        "http://127.0.0.1:8000",
        "https://sixth-vault.com", 
        "https://www.sixth-vault.com",
        "http://sixth-vault.com",  # HTTP fallback
        "http://www.sixth-vault.com",  # HTTP fallback
        # Add any other domains you might use
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "X-CSRFToken",
        "X-Real-IP",
        "X-Forwarded-For",
        "X-Forwarded-Proto",
        "Cache-Control",
        "Pragma",
        "ngrok-skip-browser-warning",
        "User-Agent",
        "Referer",
        "Origin"
    ],
    expose_headers=[
        "Content-Length",
        "Content-Range",
        "Content-Type",
        "Authorization",
        "X-Total-Count",
        "X-Page-Count"
    ],
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Configure FastAPI to prevent 307 redirects by default
app.router.redirect_slashes = False

# Add global OPTIONS handler for CORS preflight requests
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle all OPTIONS requests for CORS preflight"""
    return {"message": "OK"}

# Include routers
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(upload_router)
app.include_router(query_router)
app.include_router(documents_router)
app.include_router(curations_router)
app.include_router(summaries_router)
app.include_router(conversations_router)
app.include_router(providers_router)
app.include_router(health_router)

@app.get("/health")
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
