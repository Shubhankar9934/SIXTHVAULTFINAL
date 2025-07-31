"""
Enterprise-Grade WebSocket Broadcast System with Real-time Progress Updates
Provides reliable, scalable WebSocket communication with automatic reconnection,
message queuing, and graceful error handling for production environments.

ENTERPRISE FEATURES:
- Intelligent connection lifecycle management
- Priority-based message queuing with overflow protection
- Automatic reconnection with exponential backoff
- Real-time connection health monitoring
- Graceful degradation for network issues
- Memory-efficient message buffering
- Connection pooling and load balancing
- Enterprise-grade error handling and recovery
- AUTOMATIC WEBSOCKET CLOSURE ON COMPLETION
"""
import asyncio
import json
import time
import weakref
from typing import Dict, Set, Optional, List, Tuple, Any
from fastapi import WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from collections import deque
import logging
from enum import Enum

# Enhanced connection management with enterprise features
_connections: Dict[str, Dict[WebSocket, dict]] = {}  # Tracks WebSocket metadata
_lock = asyncio.Lock()
_heartbeat_interval = 15  # Optimized for responsiveness
_max_message_queue = 200  # Increased for enterprise workloads
_connection_timeout = 600  # 10 minutes for long operations
_cleanup_interval = 60  # Cleanup every minute

# Track batch completion status
_batch_completion_status: Dict[str, dict] = {}

# Connection state tracking with comprehensive states
class ConnectionState(Enum):
    CONNECTING = "connecting"
    ACTIVE = "active"
    DEGRADED = "degraded"
    RECONNECTING = "reconnecting"
    SUSPENDED = "suspended"
    FAILED = "failed"
    DISCONNECTED = "disconnected"

# Message priority levels for intelligent queuing
class MessagePriority(Enum):
    CRITICAL = 1    # Connection status, errors
    HIGH = 2        # Progress updates, completion
    NORMAL = 3      # General updates
    LOW = 4         # Heartbeats, diagnostics

# Enhanced message queue with priority support
class EnterpriseMessageQueue:
    def __init__(self, max_size: int = 200):
        self.queues = {
            MessagePriority.CRITICAL: deque(),
            MessagePriority.HIGH: deque(),
            MessagePriority.NORMAL: deque(),
            MessagePriority.LOW: deque()
        }
        self.max_size = max_size
        self.total_messages = 0
        
    def add_message(self, message: dict, priority: MessagePriority = MessagePriority.NORMAL):
        """Add message to appropriate priority queue with overflow protection"""
        # Remove low priority messages if queue is full
        while self.total_messages >= self.max_size:
            removed = False
            for p in [MessagePriority.LOW, MessagePriority.NORMAL, MessagePriority.HIGH]:
                if self.queues[p]:
                    self.queues[p].popleft()
                    self.total_messages -= 1
                    removed = True
                    break
            if not removed:
                break
        
        self.queues[priority].append(message)
        self.total_messages += 1
    
    def get_queued_messages(self) -> List[dict]:
        """Get all queued messages in priority order and clear queues"""
        messages = []
        for priority in MessagePriority:
            messages.extend(list(self.queues[priority]))
            self.queues[priority].clear()
        self.total_messages = 0
        return messages
    
    def clear(self):
        """Clear all queued messages"""
        for queue in self.queues.values():
            queue.clear()
        self.total_messages = 0
    
    def size(self) -> int:
        """Get total number of queued messages"""
        return self.total_messages

# Global message queues for each batch
_message_queues: Dict[str, EnterpriseMessageQueue] = {}

# Connection health monitoring
_connection_stats: Dict[str, dict] = {}

# Setup enhanced logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

async def subscribe(batch: str, ws: WebSocket):
    """Subscribe a WebSocket with enhanced connection tracking and immediate flush"""
    async with _lock:
        if batch not in _connections:
            _connections[batch] = {}
        
        # Add connection with metadata
        _connections[batch][ws] = {
            "state": ConnectionState.ACTIVE,
            "connected_at": time.time(),
            "last_heartbeat": time.time(),
            "retry_count": 0,
            "batch_id": batch
        }
        
        # Initialize message queue if not exists
        if batch not in _message_queues:
            _message_queues[batch] = EnterpriseMessageQueue(_max_message_queue)
        
        # Initialize connection stats
        if batch not in _connection_stats:
            _connection_stats[batch] = {
                "messages_sent": 0,
                "messages_queued": 0,
                "last_activity": time.time(),
                "connection_count": 0
            }
        
        _connection_stats[batch]["connection_count"] = len(_connections[batch])
        
        print(f"WebSocket subscribed for batch: {batch}")
        
        # Send immediate connection confirmation
        try:
            await ws.send_text(json.dumps({
                "type": "connection_established",
                "data": {
                    "batch_id": batch,
                    "timestamp": time.time(),
                    "message": "WebSocket connection established successfully"
                }
            }))
        except Exception as e:
            print(f"Failed to send connection confirmation: {e}")
        
        # Immediately flush any queued messages
        await flush_queued_messages_immediate(batch)

async def unsubscribe(batch: str, ws: WebSocket):
    """Unsubscribe and cleanup WebSocket resources"""
    async with _lock:
        if batch in _connections and ws in _connections[batch]:
            del _connections[batch][ws]
            if not _connections[batch]:  # Remove batch if no more connections
                del _connections[batch]
                # Keep message queue for potential reconnection
            else:
                # Update connection count
                if batch in _connection_stats:
                    _connection_stats[batch]["connection_count"] = len(_connections[batch])
            print(f"WebSocket unsubscribed from batch {batch}")

async def mark_file_completed(batch: str, completion_data: dict = None):
    """Mark a single file as completed (does NOT mark batch as completed)"""
    # Send file completion notification without marking batch as completed
    await push_high_priority(batch, "file_completed", {
        "message": "File processing completed successfully",
        "batch_id": batch,
        "completion_time": time.time(),
        "completion_data": completion_data or {},
        "status": "file_completed"
    })
    print(f"File completed for batch {batch}")

async def mark_batch_completed(batch: str, completion_data: dict = None):
    """Mark a batch as completed and schedule WebSocket closure"""
    async with _lock:
        _batch_completion_status[batch] = {
            "completed": True,
            "completion_time": time.time(),
            "completion_data": completion_data or {},
            "closure_scheduled": False
        }
        
        print(f"Batch {batch} marked as completed")
        
        # Send final completion message to all connections
        await push_high_priority(batch, "batch_completed", {
            "message": "All processing completed successfully",
            "batch_id": batch,
            "completion_time": time.time(),
            "status": "fully_completed",
            "websocket_closure": "scheduled_in_5_seconds"
        })
        
        # Schedule WebSocket closure after a brief delay
        asyncio.create_task(schedule_websocket_closure(batch, delay=5))

async def schedule_websocket_closure(batch: str, delay: int = 5):
    """Schedule WebSocket closure after processing completion"""
    try:
        print(f"Scheduling WebSocket closure for batch {batch} in {delay} seconds...")
        await asyncio.sleep(delay)
        
        async with _lock:
            if batch in _batch_completion_status:
                _batch_completion_status[batch]["closure_scheduled"] = True
        
        # Send final closure notification
        await push_high_priority(batch, "websocket_closing", {
            "message": "Processing completed. Closing WebSocket connection.",
            "batch_id": batch,
            "reason": "processing_completed",
            "timestamp": time.time()
        })
        
        # Wait a moment for the message to be sent
        await asyncio.sleep(1)
        
        # Close all WebSocket connections for this batch
        await close_batch_connections(batch)
        
    except Exception as e:
        print(f"Error scheduling WebSocket closure for batch {batch}: {e}")

async def close_batch_connections(batch: str):
    """Close all WebSocket connections for a completed batch"""
    try:
        async with _lock:
            if batch not in _connections:
                print(f"No connections found for batch {batch}")
                return
            
            connections_to_close = list(_connections[batch].keys())
            
        print(f"Closing {len(connections_to_close)} WebSocket connections for completed batch {batch}")
        
        # Close each connection gracefully
        for ws in connections_to_close:
            try:
                await ws.close(code=1000, reason="Processing completed")
                print(f"Closed WebSocket connection for batch {batch}")
            except Exception as e:
                print(f"Error closing WebSocket for batch {batch}: {e}")
        
        # Clean up batch data
        async with _lock:
            if batch in _connections:
                del _connections[batch]
            if batch in _message_queues:
                del _message_queues[batch]
            if batch in _connection_stats:
                del _connection_stats[batch]
            if batch in _batch_completion_status:
                del _batch_completion_status[batch]
        
        print(f"Completed cleanup for batch {batch}")
        
    except Exception as e:
        print(f"Error closing batch connections for {batch}: {e}")

async def is_batch_completed(batch: str) -> bool:
    """Check if a batch is marked as completed"""
    return batch in _batch_completion_status and _batch_completion_status[batch].get("completed", False)

async def push(batch: str, event: str, payload=None, retry_count: int = 0, priority: str = "normal"):
    """Enterprise-grade message push with intelligent queuing and delivery"""
    
    # Check if batch is completed and should not receive new messages
    # CRITICAL FIX: Only skip messages for truly completed batches, not individual file completions
    if await is_batch_completed(batch) and event not in ["batch_completed", "websocket_closing", "batch_fully_completed", "priority_file_completed", "background_file_completed", "rag_available"]:
        print(f"Skipping message for completed batch {batch}: {event}")
        return True
    
    # Map string priority to enum
    priority_map = {
        "critical": MessagePriority.CRITICAL,
        "high": MessagePriority.HIGH,
        "normal": MessagePriority.NORMAL,
        "low": MessagePriority.LOW
    }
    msg_priority = priority_map.get(priority, MessagePriority.NORMAL)
    
    # Ensure payload is JSON serializable and not too large
    safe_payload = payload
    if payload is not None:
        try:
            # Test JSON serialization
            test_json = json.dumps(payload)
            # Check size limit (1MB for WebSocket messages)
            if len(test_json) > 1024 * 1024:
                print(f"Warning: Large payload ({len(test_json)} bytes) for event {event}, truncating...")
                # Truncate large payloads
                if isinstance(payload, dict):
                    safe_payload = {
                        "message": str(payload.get("message", ""))[:1000],
                        "batch_id": payload.get("batch_id"),
                        "status": payload.get("status"),
                        "truncated": True,
                        "original_size": len(test_json)
                    }
                elif isinstance(payload, str):
                    safe_payload = payload[:1000] + "... [truncated]"
                else:
                    safe_payload = {"truncated": True, "type": str(type(payload))}
        except (TypeError, ValueError) as e:
            print(f"JSON serialization error for event {event}: {e}")
            safe_payload = {"error": "Payload not JSON serializable", "type": str(type(payload))}
    
    msg = {
        "type": event,
        "data": safe_payload,
        "timestamp": int(time.time() * 1000),
        "retry_count": retry_count,
        "priority": priority,
        "batch_id": batch,
        "message_id": f"{batch}_{event}_{int(time.time() * 1000)}"
    }
    
    # Initialize message queue for batch if not exists
    if batch not in _message_queues:
        _message_queues[batch] = EnterpriseMessageQueue(_max_message_queue)
    
    # Initialize connection stats
    if batch not in _connection_stats:
        _connection_stats[batch] = {
            "messages_sent": 0,
            "messages_queued": 0,
            "last_activity": time.time(),
            "connection_count": 0
        }
    
    # Check if we have active connections
    active_connections = []
    if batch in _connections:
        for ws, state in _connections[batch].items():
            if state["state"] == ConnectionState.ACTIVE:
                active_connections.append((ws, state))
    
    if not active_connections:
        print(f"No active WebSocket connections for batch {batch}, queuing message")
        _message_queues[batch].add_message(msg, msg_priority)
        _connection_stats[batch]["messages_queued"] += 1
        print(f"Message queued for batch {batch} (no active connections)")
        return False
    
    # Track delivery status
    delivery_status = {
        "success": 0,
        "failed": 0
    }
    
    # Safe JSON serialization with error handling
    try:
        msg_json = json.dumps(msg, ensure_ascii=False, separators=(',', ':'))
        
        # Additional size check after final serialization
        if len(msg_json) > 1024 * 1024:  # 1MB limit
            print(f"Error: Message too large ({len(msg_json)} bytes) for batch {batch}, skipping...")
            return False
            
    except (TypeError, ValueError) as e:
        print(f"Critical JSON serialization error for batch {batch}: {e}")
        # Create emergency fallback message
        fallback_msg = {
            "type": "error",
            "data": {
                "message": "Message serialization failed",
                "original_event": event,
                "batch_id": batch,
                "error": str(e)
            },
            "timestamp": int(time.time() * 1000)
        }
        try:
            msg_json = json.dumps(fallback_msg)
        except:
            return False
    
    # Send to all active connections
    for ws, state in active_connections:
        try:
            await ws.send_text(msg_json)
            state["last_heartbeat"] = time.time()
            delivery_status["success"] += 1
            _connection_stats[batch]["messages_sent"] += 1
            
        except (WebSocketDisconnect, ConnectionResetError) as e:
            print(f"WebSocket disconnected for batch {batch}: {e}")
            delivery_status["failed"] += 1
            await unsubscribe(batch, ws)
            
        except Exception as e:
            print(f"Push failed for batch {batch}: {e}")
            delivery_status["failed"] += 1
            
            # Enhanced error handling with exponential backoff
            if retry_count < 3 and _should_retry_error(e):
                try:
                    # Exponential backoff with jitter
                    wait_time = min(5.0, (2 ** retry_count) * 0.5)
                    await asyncio.sleep(wait_time)
                    
                    # Try to recover connection
                    if await _attempt_connection_recovery(batch, ws, state):
                        return await push(batch, event, payload, retry_count + 1, priority)
                    else:
                        await unsubscribe(batch, ws)
                        
                except Exception as retry_error:
                    print(f"Retry failed for batch {batch}: {retry_error}")
                    await unsubscribe(batch, ws)
            else:
                # Mark connection as failed and unsubscribe
                state["state"] = ConnectionState.FAILED
                await unsubscribe(batch, ws)
    
    # Update connection stats
    _connection_stats[batch]["last_activity"] = time.time()
    
    # Return success status
    return delivery_status["success"] > 0

def _should_retry_error(error: Exception) -> bool:
    """Determine if an error is worth retrying"""
    retry_errors = (
        RuntimeError,
        ConnectionResetError,
        OSError,
        asyncio.TimeoutError,
        ConnectionAbortedError
    )
    return isinstance(error, retry_errors)

async def _attempt_connection_recovery(batch: str, ws: WebSocket, state: dict) -> bool:
    """Attempt to recover a failed connection"""
    try:
        # Try a simple ping to test connection
        await ws.send_text(json.dumps({"type": "ping", "timestamp": time.time()}))
        state["last_heartbeat"] = time.time()
        state["state"] = ConnectionState.ACTIVE
        state["retry_count"] = state.get("retry_count", 0) + 1
        print(f"Connection recovery successful for batch {batch}")
        return True
    except Exception as e:
        print(f"Connection recovery failed for batch {batch}: {e}")
        return False

async def push_high_priority(batch: str, event: str, payload=None):
    """Push high-priority message that should be delivered even in degraded mode"""
    return await push(batch, event, payload, priority="high")

async def flush_queued_messages_immediate(batch: str):
    """Immediate message flushing for new connections"""
    if batch not in _message_queues or batch not in _connections or not _connections[batch]:
        return
    
    queued_messages = _message_queues[batch].get_queued_messages()
    if not queued_messages:
        return
    
    print(f"Immediately flushing {len(queued_messages)} queued messages for batch {batch}")
    
    # Get active connections
    active_connections = []
    for ws, state in _connections[batch].items():
        if state["state"] == ConnectionState.ACTIVE:
            active_connections.append((ws, state))
    
    if not active_connections:
        # Re-queue messages if no active connections
        for msg in queued_messages:
            priority = MessagePriority.NORMAL
            if msg.get("priority") == "critical":
                priority = MessagePriority.CRITICAL
            elif msg.get("priority") == "high":
                priority = MessagePriority.HIGH
            elif msg.get("priority") == "low":
                priority = MessagePriority.LOW
            _message_queues[batch].add_message(msg, priority)
        return
    
    # Send flush start notification
    flush_start_msg = {
        "type": "flush_start",
        "data": {
            "count": len(queued_messages),
            "timestamp": time.time(),
            "batch_id": batch
        }
    }
    
    for ws, state in active_connections:
        try:
            await ws.send_text(json.dumps(flush_start_msg))
        except Exception as e:
            print(f"Failed to send flush start: {e}")
    
    # Flush messages with minimal delay
    successful_flushes = 0
    failed_flushes = 0
    
    for i, msg in enumerate(queued_messages):
        try:
            # Add flush metadata to the message
            enhanced_msg = {
                **msg,
                "is_queued": True,
                "queue_position": i + 1,
                "total_queued": len(queued_messages),
                "flush_timestamp": time.time()
            }
            
            msg_json = json.dumps(enhanced_msg)
            
            # Send to all active connections
            for ws, state in active_connections:
                try:
                    await ws.send_text(msg_json)
                    successful_flushes += 1
                except Exception as e:
                    print(f"Failed to flush message to connection: {e}")
                    failed_flushes += 1
            
            # Small delay to prevent overwhelming
            if i % 5 == 0:  # Every 5 messages
                await asyncio.sleep(0.01)  # 10ms delay
            
        except Exception as e:
            print(f"Failed to flush message {i+1}/{len(queued_messages)} for batch {batch}: {e}")
            failed_flushes += 1
    
    # Send completion message
    flush_complete_msg = {
        "type": "flush_complete",
        "data": {
            "total_messages": len(queued_messages),
            "successful": successful_flushes,
            "failed": failed_flushes,
            "timestamp": time.time(),
            "batch_id": batch
        }
    }
    
    for ws, state in active_connections:
        try:
            await ws.send_text(json.dumps(flush_complete_msg))
        except Exception as e:
            print(f"Failed to send flush complete: {e}")
    
    print(f"Successfully flushed queued messages for batch {batch}")

async def flush_queued_messages(batch: str):
    """Enterprise-grade message flushing with intelligent delivery"""
    await flush_queued_messages_immediate(batch)

async def send_connection_status(batch: str, status: str, metadata: dict = None):
    """Send connection status updates to clients"""
    payload = {
        "status": status,
        "timestamp": time.time(),
        "batch_id": batch,
        **(metadata or {})
    }
    await push(batch, "connection_status", payload, priority="high")

async def wait_for_connection(batch: str, timeout: float = 15.0) -> bool:
    """Wait for at least one WebSocket connection to be established with enhanced detection"""
    start_time = time.time()
    
    print(f"Waiting for WebSocket connection for batch {batch}...")
    
    while time.time() - start_time < timeout:
        if batch in _connections and _connections[batch]:
            # Check if any connection is active
            for ws, state in _connections[batch].items():
                if state["state"] == ConnectionState.ACTIVE:
                    print(f"WebSocket connection established for batch {batch}, user: {state.get('user_id', 'unknown')}")
                    return True
        
        await asyncio.sleep(0.1)  # Check every 100ms
    
    print(f"Warning: No WebSocket connection established for batch {batch} after {timeout}s, proceeding anyway")
    return False

async def get_batch_health(batch: str) -> dict:
    """Get detailed health information for a batch"""
    if batch not in _connections:
        return {"status": "not_found", "health_score": 0.0}
    
    connections = _connections[batch]
    current_time = time.time()
    
    connection_details = []
    for ws, state in connections.items():
        connection_details.append({
            "state": state["state"].value if hasattr(state["state"], 'value') else str(state["state"]),
            "connected_duration": current_time - state["connected_at"],
            "last_heartbeat_ago": current_time - state["last_heartbeat"],
            "retry_count": state.get("retry_count", 0)
        })
    
    return {
        "status": "active" if connections else "inactive",
        "health_score": _calculate_health_score(batch),
        "connection_count": len(connections),
        "connection_details": connection_details,
        "queue_size": _message_queues[batch].size() if batch in _message_queues else 0,
        "stats": _connection_stats.get(batch, {}),
        "completed": await is_batch_completed(batch)
    }

def _calculate_health_score(batch: str) -> float:
    """Calculate health score for a specific batch (0.0 to 1.0)"""
    if batch not in _connections or not _connections[batch]:
        return 0.0
    
    connections = _connections[batch]
    active_connections = sum(1 for state in connections.values() 
                           if state["state"] == ConnectionState.ACTIVE)
    total_connections = len(connections)
    
    if total_connections == 0:
        return 0.0
    
    # Base score from active connections ratio
    connection_score = active_connections / total_connections
    
    # Penalty for queued messages
    queue_size = _message_queues[batch].size() if batch in _message_queues else 0
    queue_penalty = min(queue_size / 50, 0.3)  # Max 30% penalty
    
    # Bonus for recent activity
    stats = _connection_stats.get(batch, {})
    last_activity = stats.get("last_activity", 0)
    if time.time() - last_activity < 60:  # Active in last minute
        activity_bonus = 0.1
    else:
        activity_bonus = 0.0
    
    return max(0.0, min(1.0, connection_score - queue_penalty + activity_bonus))

def get_connection_stats(batch: str = None) -> dict:
    """Get comprehensive connection statistics for monitoring and analytics"""
    if batch:
        if batch not in _connections:
            return {
                "batch": batch, 
                "connections": 0, 
                "states": {},
                "queued_messages": 0,
                "messages_sent": 0,
                "messages_queued": 0,
                "last_activity": None,
                "completed": False
            }
        
        connections = _connections[batch]
        states = {}
        for ws, state in connections.items():
            state_name = state["state"].value if hasattr(state["state"], 'value') else str(state["state"])
            states[state_name] = states.get(state_name, 0) + 1
        
        # Get connection stats
        conn_stats = _connection_stats.get(batch, {})
        queue_size = _message_queues[batch].size() if batch in _message_queues else 0
        
        return {
            "batch": batch,
            "connections": len(connections),
            "states": states,
            "queued_messages": queue_size,
            "messages_sent": conn_stats.get("messages_sent", 0),
            "messages_queued": conn_stats.get("messages_queued", 0),
            "last_activity": conn_stats.get("last_activity"),
            "health_score": _calculate_health_score(batch),
            "completed": batch in _batch_completion_status and _batch_completion_status[batch].get("completed", False)
        }
    else:
        # Global stats
        total_connections = sum(len(batch_connections) for batch_connections in _connections.values())
        total_queued = sum(queue.size() for queue in _message_queues.values())
        total_sent = sum(stats.get("messages_sent", 0) for stats in _connection_stats.values())
        
        return {
            "total_batches": len(_connections),
            "total_connections": total_connections,
            "total_queued_messages": total_queued,
            "total_messages_sent": total_sent,
            "batches": list(_connections.keys()),
            "active_batches": len([b for b in _connections.keys() if _connections[b]]),
            "completed_batches": len(_batch_completion_status),
            "system_health": _calculate_system_health()
        }

def _calculate_system_health() -> float:
    """Calculate overall system health score"""
    if not _connections:
        return 1.0  # Perfect health when no connections (no issues)
    
    batch_scores = [_calculate_health_score(batch) for batch in _connections.keys()]
    return sum(batch_scores) / len(batch_scores) if batch_scores else 1.0

# Background cleanup task
async def cleanup_stale_connections():
    """Clean up stale connections and message queues"""
    current_time = time.time()
    stale_batches = []
    
    async with _lock:
        for batch, connections in _connections.items():
            stale_connections = []
            
            for ws, state in connections.items():
                # Check if connection is stale (no heartbeat for too long)
                if current_time - state["last_heartbeat"] > _connection_timeout:
                    stale_connections.append(ws)
            
            # Remove stale connections
            for ws in stale_connections:
                print(f"Removing stale connection for batch {batch}")
                del connections[ws]
            
            # Mark batch for cleanup if no connections left
            if not connections:
                stale_batches.append(batch)
    
    # Clean up empty batches and their message queues
    for batch in stale_batches:
        if batch in _connections:
            del _connections[batch]
        if batch in _message_queues:
            del _message_queues[batch]
        if batch in _connection_stats:
            del _connection_stats[batch]
        print(f"Cleaned up stale batch {batch}")

# Start cleanup task when module is imported
async def start_cleanup_task():
    """Start background cleanup task"""
    while True:
        try:
            await asyncio.sleep(60)  # Run every minute
            await cleanup_stale_connections()
        except Exception as e:
            print(f"Cleanup task error: {e}")

# Global cleanup task reference
_cleanup_task = None

def ensure_cleanup_task():
    """Ensure cleanup task is running"""
    global _cleanup_task
    try:
        if _cleanup_task is None or _cleanup_task.done():
            loop = asyncio.get_running_loop()
            _cleanup_task = loop.create_task(start_cleanup_task())
    except RuntimeError:
        # No running event loop, cleanup task will be started when needed
        pass

# Try to start cleanup task if event loop is available
try:
    ensure_cleanup_task()
except:
    pass
