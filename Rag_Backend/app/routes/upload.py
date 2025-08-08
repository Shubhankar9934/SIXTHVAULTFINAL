from fastapi import APIRouter, File, UploadFile, BackgroundTasks, Query, Depends
from uuid import uuid4
import asyncio
import time
import os
from typing import List, Dict, Any
from app.utils.broadcast import (
    push, wait_for_connection, send_connection_status, 
    push_high_priority, get_batch_health, mark_batch_completed
)
from app.utils.s3_storage import save_upload_file, s3_service, check_s3_health
from app.services.pipeline import lightning_process_file, enhanced_bulk_process_file
from app.deps import get_current_user
from app.config import settings
from app.services.processing_service import ProcessingService, ProcessingStatus
from app.utils.s3_storage import save_upload_file, s3_service, check_s3_health, cleanup_file_resources

router = APIRouter(tags=["upload"])

# Global processing queue for bulk operations
_processing_queues = {}
_resource_allocator = {"gpu_usage": 0.0, "cpu_usage": 0.0, "active_priority_tasks": 0}
_batch_tracking = {}  # Track batch progress: {batch_id: {"total_files": int, "completed_files": int, "files": []}}

def initialize_batch_tracking(batch_id: str, total_files: int, file_list: List[str]):
    """Initialize batch tracking for proper completion management"""
    global _batch_tracking
    _batch_tracking[batch_id] = {
        "total_files": total_files,
        "completed_files": 0,
        "files": file_list,
        "start_time": time.time(),
        "priority_completed": False,
        "background_completed": False
    }

async def mark_file_completed(batch_id: str, file_index: int, is_priority: bool = False):
    """Mark a file as completed and check if batch is done"""
    global _batch_tracking
    
    if batch_id not in _batch_tracking:
        return False
    
    batch_info = _batch_tracking[batch_id]
    batch_info["completed_files"] += 1
    
    if is_priority:
        batch_info["priority_completed"] = True
    
    # Check if all files are completed
    if batch_info["completed_files"] >= batch_info["total_files"]:
        batch_info["background_completed"] = True
        
        # Send final batch completion and mark batch as completed
        total_duration = time.time() - batch_info["start_time"]
        
        # Use the proper batch completion function that handles WebSocket closure
        await mark_batch_completed(batch_id, {
            "total_files": batch_info["total_files"],
            "completed_files": batch_info["completed_files"],
            "total_duration": total_duration,
            "files": batch_info["files"]
        })
        
        # Clean up tracking
        del _batch_tracking[batch_id]
        return True
    
    return False

class ResourceAllocator:
    """Smart resource allocation for priority vs background processing"""
    
    @staticmethod
    def allocate_for_priority():
        """Allocate maximum resources for priority file processing"""
        return {
            "gpu_allocation": 1.0,
            "cpu_cores": 8,
            "memory_limit": "4GB",
            "concurrent_tasks": 8,
            "timeout_strategy": "unlimited",
            "provider": "bedrock",
            "model": "anthropic.claude-3-haiku-20240307-v1:0"
        }
    
    @staticmethod
    def allocate_for_background():
        """Allocate conservative resources for background processing"""
        return {
            "gpu_allocation": 0.25,
            "cpu_cores": 2,
            "memory_limit": "1GB", 
            "concurrent_tasks": 4,
            "timeout_strategy": "adaptive",
            "provider": "bedrock",
            "model": "anthropic.claude-3-haiku-20240307-v1:0"
        }

async def calculate_batch_eta(files: List[UploadFile]) -> Dict[str, int]:
    """Calculate estimated processing times for bulk upload with scalable batching"""
    total_size = sum(f.size for f in files)
    file_count = len(files)
    
    # Estimation based on file size and count
    # Priority file: 15-20 seconds
    priority_eta = 18
    
    # Smart batching for background files based on count
    if file_count <= 5:
        # Small batch: Process 2-3 files in parallel
        parallel_factor = 2
        background_eta = max(25, ((file_count - 1) // parallel_factor + 1) * 20)
    elif file_count <= 20:
        # Medium batch: Process 4-5 files in parallel
        parallel_factor = 4
        background_eta = max(30, ((file_count - 1) // parallel_factor + 1) * 18)
    else:
        # Large batch: Process 6-8 files in parallel with optimized timing
        parallel_factor = 6
        background_eta = max(40, ((file_count - 1) // parallel_factor + 1) * 15)
    
    # Total batch time with overlap optimization
    total_eta = priority_eta + background_eta
    
    return {
        "priority_eta": priority_eta,
        "background_eta": background_eta,
        "total_eta": total_eta,
        "rag_ready_time": priority_eta + 5,  # RAG available 5s after first file
        "parallel_factor": parallel_factor,
        "estimated_parallel_batches": (file_count - 1) // parallel_factor + 1 if file_count > 1 else 0
    }

async def enterprise_push(batch: str, event: str, payload=None, priority: str = "normal"):
    """Enterprise-grade message push with comprehensive error handling"""
    try:
        success = await push(batch, event, payload, priority=priority)
        if not success:
            print(f"Message queued for batch {batch} (no active connections)")
        return success
    except Exception as e:
        print(f"Failed to send WebSocket message for batch {batch}: {e}")
        # Continue processing even if WebSocket fails
        return False

async def priority_process_file(user_id: str, batch: str, path: str, file_index: int = 0):
    """Priority processing for first file with dedicated resources"""
    processing_start_time = time.time()
    
    # Mark priority processing start
    global _resource_allocator
    _resource_allocator["active_priority_tasks"] += 1
    _resource_allocator["gpu_usage"] = 1.0
    
    # Send priority processing notification
    await enterprise_push(batch, "priority_processing_started", {
        "message": "🚀 Priority processing initiated - First file gets dedicated resources",
        "batch_id": batch,
        "file_index": file_index,
        "provider": "bedrock",
        "model": "anthropic.claude-3-haiku-20240307-v1:0",
        "resource_allocation": "maximum",
        "estimated_completion": 18,
        "start_time": processing_start_time
    }, priority="critical")
    
    try:
        # Use enhanced processing with priority resources
        await lightning_process_file(user_id, batch, path, "bedrock", "anthropic.claude-3-haiku-20240307-v1:0")
        
        # Send RAG availability notification
        processing_duration = time.time() - processing_start_time
        await enterprise_push(batch, "rag_available", {
            "message": "🎉 RAG is now available! First document processed and searchable",
            "batch_id": batch,
            "file_index": file_index,
            "duration": processing_duration,
            "status": "rag_ready",
            "search_enabled": True
        }, priority="critical")
        
        # Send priority file completion and update batch tracking
        await enterprise_push(batch, "priority_file_completed", {
            "message": "✅ Priority file processing completed - RAG is ready",
            "batch_id": batch,
            "file_index": file_index,
            "duration": processing_duration,
            "status": "priority_completed"
        }, priority="critical")
        
        # Mark priority file as completed in batch tracking
        await mark_file_completed(batch, file_index, is_priority=True)
        
        # Release priority resources
        _resource_allocator["active_priority_tasks"] -= 1
        if _resource_allocator["active_priority_tasks"] == 0:
            _resource_allocator["gpu_usage"] = 0.25  # Switch to background allocation
        
        print(f"🚀 Priority file processing completed for {path} in {processing_duration:.2f}s - RAG AVAILABLE")
        
    except Exception as e:
        # Release resources on error
        _resource_allocator["active_priority_tasks"] -= 1
        if _resource_allocator["active_priority_tasks"] == 0:
            _resource_allocator["gpu_usage"] = 0.25
            
        processing_duration = time.time() - processing_start_time
        await enterprise_push(batch, "priority_processing_error", {
            "message": f"Priority processing failed: {str(e)}",
            "batch_id": batch,
            "file_index": file_index,
            "duration": processing_duration,
            "error": str(e),
            "status": "error"
        }, priority="critical")
        
        print(f"❌ Priority processing failed for {path} after {processing_duration:.2f}s: {e}")
        raise

async def background_process_file(user_id: str, batch: str, path: str, file_index: int):
    """Background processing for remaining files with conservative resources"""
    processing_start_time = time.time()
    
    # Send background processing notification
    await enterprise_push(batch, "background_processing_started", {
        "message": f"Background processing file {file_index}",
        "batch_id": batch,
        "file_index": file_index,
        "provider": "bedrock",
        "model": "anthropic.claude-3-haiku-20240307-v1:0",
        "resource_allocation": "background",
        "start_time": processing_start_time
    }, priority="normal")
    
    try:
        # Use background resources for processing
        await lightning_process_file(user_id, batch, path, "bedrock", "anthropic.claude-3-haiku-20240307-v1:0")
        
        processing_duration = time.time() - processing_start_time
        await enterprise_push(batch, "background_file_completed", {
            "message": f"Background file {file_index} processed and added to RAG",
            "batch_id": batch,
            "file_index": file_index,
            "duration": processing_duration,
            "status": "completed"
        }, priority="normal")
        
        # Mark background file as completed in batch tracking
        await mark_file_completed(batch, file_index, is_priority=False)
        
        print(f"Background file processing completed for {path} in {processing_duration:.2f}s")
        
    except Exception as e:
        processing_duration = time.time() - processing_start_time
        await enterprise_push(batch, "background_processing_error", {
            "message": f"Background file {file_index} processing failed: {str(e)}",
            "batch_id": batch,
            "file_index": file_index,
            "duration": processing_duration,
            "error": str(e),
            "status": "error"
        }, priority="high")
        
        print(f"❌ Background processing failed for {path} after {processing_duration:.2f}s: {e}")
        raise

async def enhanced_process_file(user_id: str, batch: str, file_info: Dict[str, Any]):
    """Enterprise-grade file processing with intelligent WebSocket management and S3 integration"""
    processing_start_time = time.time()
    
    # Extract file information
    local_path = file_info["local_path"]
    s3_key = file_info["s3_key"]
    s3_bucket = file_info["s3_bucket"]
    filename = file_info["filename"]
    file_size = file_info["file_size"]
    content_type = file_info["content_type"]
    
    # Generate document ID for tracking
    doc_id = f"{filename}_{batch}"
    
    # Register document in processing tracker with S3 metadata
    try:
        await ProcessingService.register_processing_document(
            doc_id=doc_id,
            filename=filename,
            batch_id=batch,
            owner_id=user_id,
            file_path=local_path,
            s3_key=s3_key,
            s3_bucket=s3_bucket,
            file_size=file_size,
            content_type=content_type
        )
        await ProcessingService.update_processing_status(doc_id, ProcessingStatus.UPLOADING, 0, "uploading")
    except Exception as e:
        print(f"Failed to register processing document {doc_id}: {e}")
    
    # Send initial processing notification
    await enterprise_push(batch, "processing_started", {
        "message": "File processing initiated with AWS Bedrock",
        "batch_id": batch,
        "document_id": doc_id,
        "filename": filename,
        "provider": "bedrock",
        "model": "anthropic.claude-3-haiku-20240307-v1:0",
        "start_time": processing_start_time,
        "s3_key": s3_key,
        "file_size": file_size
    }, priority="high")
    
    # Wait for WebSocket connection with enhanced timeout
    print(f"Waiting for WebSocket connection for batch {batch}...")
    connection_established = await wait_for_connection(batch, timeout=15.0)
    
    if connection_established:
        # Get connection health information
        health_info = await get_batch_health(batch)
        await send_connection_status(batch, "connected", {
            "message": "Enterprise WebSocket connection established",
            "batch_id": batch,
            "health_score": health_info.get("health_score", 1.0),
            "connection_count": health_info.get("connection_count", 0)
        })
        print(f"WebSocket connection established for batch {batch} (health: {health_info.get('health_score', 1.0):.2f})")
    else:
        await send_connection_status(batch, "no_connection", {
            "message": "Processing without WebSocket connection - messages will be queued",
            "batch_id": batch,
            "fallback_mode": True
        })
        print(f"Starting processing for batch {batch} in fallback mode (no WebSocket)")
    
    try:
        # Check for cancellation before starting
        if await ProcessingService.is_cancellation_requested(doc_id):
            print(f"Processing cancelled before start for {doc_id}")
            return
        
        # Update status to processing
        await ProcessingService.update_processing_status(doc_id, ProcessingStatus.PROCESSING, 10, "processing")
        
        # Start the actual file processing with enhanced monitoring - Lightning Fast
        await lightning_process_file(user_id, batch, local_path, "bedrock", "anthropic.claude-3-haiku-20240307-v1:0")
        
        # Check for cancellation after processing
        if await ProcessingService.is_cancellation_requested(doc_id):
            print(f"Processing cancelled after completion for {doc_id}")
            return
        
        # Update status to completed
        await ProcessingService.update_processing_status(doc_id, ProcessingStatus.COMPLETED, 100, "completed")
        
        # Remove from processing tracker (document is now in database)
        await ProcessingService.remove_processing_document(doc_id, user_id)
        
        # Clean up temporary local file (keep S3 file)
        await cleanup_file_resources(local_path=local_path)
        
        # Send completion notification with comprehensive data for immediate frontend display
        processing_duration = time.time() - processing_start_time
        await enterprise_push(batch, "processing_completed", {
            "message": "File processing completed successfully with AI analysis",
            "batch_id": batch,
            "document_id": doc_id,
            "filename": filename,
            "duration": processing_duration,
            "status": "success",
            "s3_key": s3_key,
            "ai_analysis_ready": True,  # Flag to indicate AI data is available
            "progress": 100
        }, priority="high")
        
        print(f"File processing completed for batch {batch} in {processing_duration:.2f}s")
        
    except Exception as e:
        # Update status to error
        try:
            await ProcessingService.update_processing_status(doc_id, ProcessingStatus.ERROR, 0, "error", str(e))
        except:
            pass
        
        # Clean up resources on error
        await cleanup_file_resources(s3_key=s3_key, local_path=local_path)
        
        # Send error notification
        processing_duration = time.time() - processing_start_time
        await enterprise_push(batch, "processing_error", {
            "message": f"File processing failed: {str(e)}",
            "batch_id": batch,
            "document_id": doc_id,
            "filename": filename,
            "duration": processing_duration,
            "error": str(e),
            "status": "error"
        }, priority="critical")
        
        print(f"File processing failed for batch {batch} after {processing_duration:.2f}s: {e}")
        raise  # Re-raise the exception for proper error handling

@router.post("/upload")
async def upload(
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    user = Depends(get_current_user),
    sequential: bool = Query(default=False, description="Process files sequentially (one at a time)")
):
    """Enhanced file upload endpoint with sequential processing option"""
    batch = str(uuid4())
    upload_start_time = time.time()
    
    # Validate files before processing
    valid_files = []
    invalid_files = []
    
    for f in files:
        if f.size > 100 * 1024 * 1024:  # 100MB limit
            invalid_files.append({"filename": f.filename, "reason": "File too large (max 100MB)"})
        elif not f.content_type or f.content_type not in [
            "application/pdf", 
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain", 
            "application/rtf"
        ]:
            invalid_files.append({"filename": f.filename, "reason": "Unsupported file type"})
        else:
            valid_files.append(f)
    
    # Send initial upload notification with processing mode
    processing_mode = "sequential" if sequential else "parallel"
    await enterprise_push(batch, "upload_started", {
        "message": f"File upload initiated with {processing_mode} processing",
        "batch_id": batch,
        "total_files": len(files),
        "valid_files": len(valid_files),
        "invalid_files": len(invalid_files),
        "invalid_file_details": invalid_files,
        "processing_mode": processing_mode,
        "provider": "bedrock",
        "model": "anthropic.claude-3-haiku-20240307-v1:0",
        "user_id": str(user.id),
        "upload_time": upload_start_time
    }, priority="high")
    
    if not valid_files:
        # No valid files to process
        await enterprise_push(batch, "upload_error", {
            "message": "No valid files to process",
            "batch_id": batch,
            "error": "All files failed validation",
            "invalid_files": invalid_files
        }, priority="critical")
        
        return {
            "batch_id": batch, 
            "count": 0, 
            "valid_files": 0,
            "invalid_files": len(invalid_files),
            "error": "No valid files to process",
            "details": invalid_files
        }
    
    # Choose processing strategy based on sequential flag
    if sequential:
        # SEQUENTIAL PROCESSING: Process one file at a time
        await sequential_file_processing(background, batch, str(user.id), valid_files)
    else:
        # PARALLEL PROCESSING: Original behavior
        await parallel_file_processing(background, batch, str(user.id), valid_files)
    
    # Send upload completion summary
    upload_duration = time.time() - upload_start_time
    await enterprise_push(batch, "upload_completed", {
        "message": f"File upload phase completed ({processing_mode} mode)",
        "batch_id": batch,
        "total_files": len(files),
        "valid_files": len(valid_files),
        "processing_mode": processing_mode,
        "upload_duration": upload_duration,
        "next_phase": "processing"
    }, priority="high")
    
    print(f"Upload completed for batch {batch}: {len(valid_files)} files queued for {processing_mode} processing")
    
    return {
        "batch_id": batch, 
        "count": len(files),
        "valid_files": len(valid_files),
        "processing_mode": processing_mode,
        "upload_duration": upload_duration,
        "status": "success"
    }

async def sequential_file_processing(background: BackgroundTasks, batch: str, user_id: str, valid_files: list[UploadFile]):
    """Process files sequentially - one at a time with proper progress updates"""
    
    await enterprise_push(batch, "sequential_processing_started", {
        "message": f"🔄 Starting sequential processing for {len(valid_files)} files",
        "batch_id": batch,
        "total_files": len(valid_files),
        "processing_mode": "sequential",
        "estimated_time_per_file": "30-60 seconds"
    }, priority="high")
    
    # Initialize sequential batch tracking
    file_names = [f.filename for f in valid_files]
    initialize_batch_tracking(batch, len(valid_files), file_names)
    
    # Add sequential processing task
    background.add_task(process_files_sequentially, user_id, batch, valid_files)

async def parallel_file_processing(background: BackgroundTasks, batch: str, user_id: str, valid_files: list[UploadFile]):
    """Process files in parallel - original behavior"""
    
    await enterprise_push(batch, "queued", {
        "message": f"Files queued for parallel processing",
        "batch_id": batch,
        "files": [f.filename for f in valid_files],
        "processing_mode": "parallel",
        "provider": "bedrock",
        "model": "anthropic.claude-3-haiku-20240307-v1:0"
    }, priority="normal")

    # Process each valid file in parallel (original behavior)
    processed_files = 0
    failed_files = 0
    
    for i, f in enumerate(valid_files):
        try:
            # Save file with progress tracking
            await enterprise_push(batch, "file_upload_progress", {
                "message": f"Uploading file {i+1}/{len(valid_files)}",
                "batch_id": batch,
                "filename": f.filename,
                "file_index": i + 1,
                "total_files": len(valid_files),
                "progress": int((i / len(valid_files)) * 20)  # 20% for upload phase
            }, priority="normal")
            
            file_info = await save_upload_file(str(user_id), batch, f)
            
            # Add processing task in parallel
            background.add_task(enhanced_process_file, user_id, batch, file_info)
            processed_files += 1
            
            # Send file uploaded confirmation
            await enterprise_push(batch, "file_uploaded", {
                "message": f"File uploaded successfully: {f.filename}",
                "batch_id": batch,
                "filename": f.filename,
                "file_path": file_info["local_path"],
                "s3_key": file_info["s3_key"],
                "file_size": f.size,
                "file_index": i + 1,
                "total_files": len(valid_files)
            }, priority="normal")
            
        except Exception as e:
            failed_files += 1
            await enterprise_push(batch, "file_upload_error", {
                "message": f"Failed to upload file: {f.filename}",
                "batch_id": batch,
                "filename": f.filename,
                "error": str(e),
                "file_index": i + 1,
                "total_files": len(valid_files)
            }, priority="high")
            
            print(f"Failed to upload file {f.filename} for batch {batch}: {e}")

async def process_files_sequentially(user_id: str, batch: str, files: list[UploadFile]):
    """Process files one at a time in sequence"""
    
    for i, file in enumerate(files):
        current_file_num = i + 1
        
        try:
            # Send file processing start notification
            await enterprise_push(batch, "sequential_file_started", {
                "message": f"📁 Processing file {current_file_num}/{len(files)}: {file.filename}",
                "batch_id": batch,
                "filename": file.filename,
                "file_index": current_file_num,
                "total_files": len(files),
                "progress": int((i / len(files)) * 100)
            }, priority="high")
            
            # Save file
            file_info = await save_upload_file(user_id, batch, file)
            
            # Send file uploaded notification
            await enterprise_push(batch, "file_uploaded", {
                "message": f"File uploaded successfully: {file.filename}",
                "batch_id": batch,
                "filename": file.filename,
                "file_path": file_info["local_path"],
                "s3_key": file_info["s3_key"],
                "file_size": file.size,
                "file_index": current_file_num,
                "total_files": len(files)
            }, priority="normal")
            
            # Process this file synchronously (wait for completion)
            await process_single_file_sync(user_id, batch, file_info, current_file_num, len(files))
            
            # Send file completion notification
            await enterprise_push(batch, "sequential_file_completed", {
                "message": f"✅ File {current_file_num}/{len(files)} completed: {file.filename}",
                "batch_id": batch,
                "filename": file.filename,
                "file_index": current_file_num,
                "total_files": len(files),
                "progress": int((current_file_num / len(files)) * 100)
            }, priority="high")
            
            # Mark file as completed in batch tracking
            await mark_file_completed(batch, i, is_priority=(i == 0))
            
            print(f"✅ Sequential processing completed for file {current_file_num}/{len(files)}: {file.filename}")
            
            # Brief pause between files for better UX (except for last file)
            if i < len(files) - 1:
                await asyncio.sleep(1)
                
        except Exception as e:
            # Send file error notification
            await enterprise_push(batch, "sequential_file_error", {
                "message": f"❌ File {current_file_num}/{len(files)} failed: {file.filename}",
                "batch_id": batch,
                "filename": file.filename,
                "file_index": current_file_num,
                "total_files": len(files),
                "error": str(e)
            }, priority="critical")
            
            print(f"❌ Sequential processing failed for file {current_file_num}/{len(files)}: {file.filename} - {e}")
            continue
    
    # Send sequential processing completion
    await enterprise_push(batch, "sequential_processing_completed", {
        "message": f"🎉 Sequential processing completed for all {len(files)} files",
        "batch_id": batch,
        "total_files": len(files),
        "processing_mode": "sequential"
    }, priority="critical")

async def process_single_file_sync(user_id: str, batch: str, file_info: Dict[str, Any], file_index: int, total_files: int):
    """Process a single file synchronously and wait for completion"""
    processing_start_time = time.time()
    
    # Extract file information
    local_path = file_info["local_path"]
    s3_key = file_info["s3_key"]
    s3_bucket = file_info["s3_bucket"]
    filename = file_info["filename"]
    file_size = file_info["file_size"]
    content_type = file_info["content_type"]
    
    # Generate document ID for tracking
    doc_id = f"{filename}_{batch}_{file_index}"
    
    try:
        # Register document in processing tracker
        await ProcessingService.register_processing_document(
            doc_id=doc_id,
            filename=filename,
            batch_id=batch,
            owner_id=user_id,
            file_path=local_path,
            s3_key=s3_key,
            s3_bucket=s3_bucket,
            file_size=file_size,
            content_type=content_type
        )
        
        # Update to processing status
        await ProcessingService.update_processing_status(doc_id, ProcessingStatus.PROCESSING, 10, "processing")
        
        # Send processing start notification
        await enterprise_push(batch, "processing", {
            "message": f"🔄 AI processing file {file_index}/{total_files}: {filename}",
            "batch_id": batch,
            "filename": filename,
            "file_index": file_index,
            "total_files": total_files,
            "progress": 30,
            "provider": "bedrock",
            "model": "anthropic.claude-3-haiku-20240307-v1:0"
        }, priority="high")
        
        # Process the file with lightning fast processing
        result = await lightning_process_file(user_id, batch, local_path, "bedrock", "anthropic.claude-3-haiku-20240307-v1:0")
        
        # Update to completed status
        await ProcessingService.update_processing_status(doc_id, ProcessingStatus.COMPLETED, 100, "completed")
        
        # Send completion notification with AI data
        processing_duration = time.time() - processing_start_time
        await enterprise_push(batch, "completed", {
            "message": f"✅ AI processing completed for {filename}",
            "batch_id": batch,
            "filename": filename,
            "file": filename,  # For frontend compatibility
            "file_index": file_index,
            "total_files": total_files,
            "progress": 100,
            "duration": processing_duration,
            "doc_id": doc_id,
            "ai_analysis_complete": True
        }, priority="critical")
        
        # Clean up processing tracker
        await ProcessingService.remove_processing_document(doc_id, user_id)
        
        # Clean up temporary local file (keep S3 file)
        await cleanup_file_resources(local_path=local_path)
        
        print(f"✅ Synchronous processing completed for {filename} in {processing_duration:.2f}s")
        
    except Exception as e:
        # Update status to error
        try:
            await ProcessingService.update_processing_status(doc_id, ProcessingStatus.ERROR, 0, "error", str(e))
        except:
            pass
        
        # Clean up resources on error
        await cleanup_file_resources(s3_key=s3_key, local_path=local_path)
        
        # Send error notification
        processing_duration = time.time() - processing_start_time
        await enterprise_push(batch, "error", {
            "message": f"❌ Processing failed for {filename}: {str(e)}",
            "batch_id": batch,
            "filename": filename,
            "file": filename,  # For frontend compatibility
            "file_index": file_index,
            "total_files": total_files,
            "error": str(e),
            "duration": processing_duration
        }, priority="critical")
        
        print(f"❌ Synchronous processing failed for {filename} after {processing_duration:.2f}s: {e}")
        raise

@router.post("/upload/bulk")
async def bulk_upload_enhanced(
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    user = Depends(get_current_user),
):
    """Enhanced bulk upload with two-tier processing strategy"""
    batch = str(uuid4())
    upload_start_time = time.time()
    
    # Validate files
    valid_files = []
    invalid_files = []
    
    for f in files:
        if f.size > 100 * 1024 * 1024:  # 100MB limit
            invalid_files.append({"filename": f.filename, "reason": "File too large (max 100MB)"})
        elif not f.content_type or f.content_type not in [
            "application/pdf", 
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain", 
            "application/rtf"
        ]:
            invalid_files.append({"filename": f.filename, "reason": "Unsupported file type"})
        else:
            valid_files.append(f)
    
    if not valid_files:
        return {
            "batch_id": batch, 
            "count": 0, 
            "valid_files": 0,
            "invalid_files": len(invalid_files),
            "error": "No valid files to process",
            "details": invalid_files
        }
    
    # Calculate ETAs for user feedback
    eta_info = await calculate_batch_eta(valid_files)
    
    # Send enhanced bulk upload notification with ETA
    await enterprise_push(batch, "bulk_upload_started", {
        "message": "🚀 Enhanced bulk processing initiated",
        "batch_id": batch,
        "total_files": len(files),
        "valid_files": len(valid_files),
        "invalid_files": len(invalid_files),
        "processing_strategy": "two_tier_hybrid",
        "eta_info": eta_info,
        "immediate_feedback": f"First document ready in {eta_info['priority_eta']}s, RAG available in {eta_info['rag_ready_time']}s, full batch in {eta_info['total_eta']}s",
        "provider": "bedrock",
        "user_id": str(user.id),
        "upload_time": upload_start_time
    }, priority="critical")
    
    # Save all files first
    file_paths = []
    for i, f in enumerate(valid_files):
        try:
            file_info = await save_upload_file(str(user.id), batch, f)
            # Extract the local_path from the file_info dictionary
            local_path = file_info["local_path"]
            file_paths.append((local_path, f.filename, i))
            
            await enterprise_push(batch, "file_saved", {
                "message": f"File {i+1}/{len(valid_files)} saved",
                "batch_id": batch,
                "filename": f.filename,
                "file_index": i + 1,
                "total_files": len(valid_files)
            }, priority="normal")
            
        except Exception as e:
            await enterprise_push(batch, "file_save_error", {
                "message": f"Failed to save file: {f.filename}",
                "batch_id": batch,
                "filename": f.filename,
                "error": str(e),
                "file_index": i + 1
            }, priority="high")
            continue
    
    if not file_paths:
        await enterprise_push(batch, "bulk_upload_error", {
            "message": "Failed to save any files",
            "batch_id": batch,
            "error": "File saving failed"
        }, priority="critical")
        
        return {
            "batch_id": batch,
            "count": len(files),
            "valid_files": len(valid_files),
            "processed_files": 0,
            "failed_files": len(valid_files),
            "status": "failed"
        }
    
    # Initialize batch tracking for proper completion management
    file_names = [filename for _, filename, _ in file_paths]
    initialize_batch_tracking(batch, len(file_paths), file_names)
    
    # TWO-TIER PROCESSING STRATEGY
    
    # TIER 1: Priority processing for first file
    if file_paths:
        priority_path, priority_filename, _ = file_paths[0]
        
        await enterprise_push(batch, "tier1_initiated", {
            "message": "🚀 TIER 1: Priority processing starting for immediate RAG availability",
            "batch_id": batch,
            "priority_file": priority_filename,
            "resource_allocation": "maximum",
            "estimated_completion": eta_info['priority_eta']
        }, priority="critical")
        
        # Start priority processing immediately
        background.add_task(priority_process_file, str(user.id), batch, priority_path, 0)
    
    # TIER 2: Background processing for remaining files
    if len(file_paths) > 1:
        background_files = file_paths[1:]
        
        await enterprise_push(batch, "tier2_initiated", {
            "message": f"📦 TIER 2: Background processing queued for {len(background_files)} files",
            "batch_id": batch,
            "background_files": len(background_files),
            "resource_allocation": "background",
            "estimated_completion": eta_info['background_eta']
        }, priority="normal")
        
        # Start background processing with delay to allow priority to complete
        async def delayed_background_processing():
            # Wait for priority file to start
            await asyncio.sleep(2)
            
            # Use smart batching based on file count
            parallel_factor = eta_info.get('parallel_factor', 3)
            batch_size = min(parallel_factor, len(background_files))
            
            await enterprise_push(batch, "background_batching_started", {
                "message": f"🔄 Starting background processing with {parallel_factor}-file parallel batches",
                "batch_id": batch,
                "total_background_files": len(background_files),
                "parallel_factor": parallel_factor,
                "estimated_batches": eta_info.get('estimated_parallel_batches', 1)
            }, priority="normal")
            
            # Process background files in optimized parallel batches
            for i in range(0, len(background_files), batch_size):
                batch_chunk = background_files[i:i + batch_size]
                batch_number = (i // batch_size) + 1
                total_batches = (len(background_files) + batch_size - 1) // batch_size
                
                await enterprise_push(batch, "background_batch_started", {
                    "message": f"📦 Starting background batch {batch_number}/{total_batches} ({len(batch_chunk)} files)",
                    "batch_id": batch,
                    "batch_number": batch_number,
                    "total_batches": total_batches,
                    "files_in_batch": len(batch_chunk)
                }, priority="normal")
                
                # Start this batch of files in parallel
                for path, filename, file_index in batch_chunk:
                    background.add_task(background_process_file, str(user.id), batch, path, file_index + 1)
                
                # Adaptive delay between batches based on batch size
                if i + batch_size < len(background_files):
                    # Shorter delays for larger batches to maintain throughput
                    delay = max(2, 8 - parallel_factor)  # 2-6 second delay
                    await asyncio.sleep(delay)
        
        # Start delayed background processing
        background.add_task(delayed_background_processing)
    
    # Send final bulk upload confirmation
    upload_duration = time.time() - upload_start_time
    await enterprise_push(batch, "bulk_upload_queued", {
        "message": "🎯 Enhanced bulk processing pipeline initiated successfully",
        "batch_id": batch,
        "total_files": len(files),
        "valid_files": len(valid_files),
        "priority_file": file_paths[0][1] if file_paths else None,
        "background_files": len(file_paths) - 1 if len(file_paths) > 1 else 0,
        "upload_duration": upload_duration,
        "eta_info": eta_info,
        "next_steps": [
            f"Priority file processing ({eta_info['priority_eta']}s)",
            f"RAG availability ({eta_info['rag_ready_time']}s)",
            f"Background files processing ({eta_info['background_eta']}s)",
            f"Full batch completion ({eta_info['total_eta']}s)"
        ]
    }, priority="critical")
    
    print(f"🚀 Enhanced bulk upload initiated for batch {batch}: {len(valid_files)} files")
    print(f"   Priority: 1 file -> RAG ready in {eta_info['rag_ready_time']}s")
    print(f"   Background: {len(file_paths) - 1 if len(file_paths) > 1 else 0} files -> Complete in {eta_info['total_eta']}s")
    
    return {
        "batch_id": batch,
        "count": len(files),
        "valid_files": len(valid_files),
        "processed_files": len(file_paths),
        "failed_files": len(valid_files) - len(file_paths),
        "invalid_files": len(invalid_files),
        "upload_duration": upload_duration,
        "processing_strategy": "two_tier_hybrid",
        "eta_info": eta_info,
        "status": "queued",
        "immediate_feedback": f"First document ready in {eta_info['priority_eta']}s, RAG available in {eta_info['rag_ready_time']}s, full batch in {eta_info['total_eta']}s"
    }
