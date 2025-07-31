import boto3
import aiofiles
import asyncio
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import UploadFile, HTTPException
from typing import Optional, Dict, Any
import os
import tempfile
from pathlib import Path
from uuid import uuid4
import logging
from datetime import datetime, timedelta
from app.config import settings

logger = logging.getLogger(__name__)

class S3StorageService:
    """Enterprise-grade S3 storage service for SixthVault"""
    
    def __init__(self):
        self.bucket_name = settings.s3_bucket_name
        self.region = settings.aws_region
        self.signed_url_expiry = settings.s3_signed_url_expiry
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=self.region,
                use_ssl=settings.s3_use_ssl,
                endpoint_url=settings.s3_endpoint_url
            )
            
            # Test connection
            self._test_connection()
            logger.info(f"S3 client initialized successfully for bucket: {self.bucket_name}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            raise HTTPException(status_code=500, detail="AWS credentials not configured")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise HTTPException(status_code=500, detail=f"S3 initialization failed: {str(e)}")
    
    def _test_connection(self):
        """Test S3 connection and bucket access"""
        try:
            # Try to list objects in the bucket (this tests both connection and permissions)
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Successfully connected to S3 bucket: {self.bucket_name}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise HTTPException(status_code=500, detail=f"S3 bucket '{self.bucket_name}' not found")
            elif error_code == '403':
                raise HTTPException(status_code=500, detail=f"Access denied to S3 bucket '{self.bucket_name}'")
            else:
                raise HTTPException(status_code=500, detail=f"S3 connection error: {str(e)}")
    
    def _generate_s3_key(self, user_id: str, batch_id: str, filename: str) -> str:
        """Generate S3 object key with proper organization"""
        # Sanitize filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-").rstrip()
        if not safe_filename:
            safe_filename = f"file_{uuid4().hex[:8]}"
        
        return f"documents/{user_id}/{batch_id}/{safe_filename}"
    
    async def upload_file(self, user_id: str, batch_id: str, file: UploadFile) -> Dict[str, Any]:
        """
        Upload file to S3 with progress tracking and error handling
        
        Returns:
            Dict containing S3 key, URL, metadata, etc.
        """
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        s3_key = self._generate_s3_key(user_id, batch_id, file.filename)
        
        try:
            # Create a temporary file to handle the upload
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file_path = temp_file.name
            
            try:
                # Read file content in chunks
                content = await file.read()
                temp_file.write(content)
                temp_file.flush()
                temp_file.close()  # Explicitly close the file handle
                
                # Reset file position for potential re-use
                await file.seek(0)
                
                # Upload to S3 with metadata
                extra_args = {
                    'Metadata': {
                        'user_id': user_id,
                        'batch_id': batch_id,
                        'original_filename': file.filename,
                        'content_type': file.content_type or 'application/octet-stream',
                        'upload_timestamp': str(asyncio.get_event_loop().time())
                    },
                    'ContentType': file.content_type or 'application/octet-stream',
                    'ServerSideEncryption': 'AES256'  # Enable encryption
                }
                
                # Perform the upload
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.s3_client.upload_file(
                        temp_file_path,
                        self.bucket_name,
                        s3_key,
                        ExtraArgs=extra_args
                    )
                )
                
            finally:
                # Clean up temp file - now it's properly closed
                try:
                    os.unlink(temp_file_path)
                except OSError as e:
                    logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")
                
                # Generate file info
                file_info = {
                    's3_key': s3_key,
                    's3_bucket': self.bucket_name,
                    's3_region': self.region,
                    'filename': file.filename,
                    'content_type': file.content_type,
                    'size': len(content),
                    'user_id': user_id,
                    'batch_id': batch_id,
                    'url': f"s3://{self.bucket_name}/{s3_key}",
                    'public_url': None  # We use signed URLs for security
                }
                
                logger.info(f"Successfully uploaded file to S3: {s3_key}")
                return file_info
                
        except ClientError as e:
            logger.error(f"S3 upload failed for {s3_key}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload file to S3: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during S3 upload: {e}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    async def generate_signed_url(self, s3_key: str, expiry: Optional[int] = None) -> str:
        """Generate a signed URL for secure file access"""
        try:
            expiry_time = expiry or self.signed_url_expiry
            
            signed_url = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.bucket_name, 'Key': s3_key},
                    ExpiresIn=expiry_time
                )
            )
            
            logger.debug(f"Generated signed URL for {s3_key} (expires in {expiry_time}s)")
            return signed_url
            
        except ClientError as e:
            logger.error(f"Failed to generate signed URL for {s3_key}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate file access URL: {str(e)}")
    
    async def download_file(self, s3_key: str, local_path: str) -> str:
        """Download file from S3 to local path"""
        try:
            # Ensure directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.download_file(
                    self.bucket_name,
                    s3_key,
                    local_path
                )
            )
            
            logger.info(f"Downloaded {s3_key} to {local_path}")
            return local_path
            
        except ClientError as e:
            logger.error(f"S3 download failed for {s3_key}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")
    
    async def delete_file(self, s3_key: str) -> bool:
        """Delete file from S3"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.delete_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
            )
            
            logger.info(f"Deleted file from S3: {s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"S3 delete failed for {s3_key}: {e}")
            return False
    
    async def file_exists(self, s3_key: str) -> bool:
        """Check if file exists in S3"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.head_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
            )
            return True
        except ClientError:
            return False
    
    async def get_file_metadata(self, s3_key: str) -> Dict[str, Any]:
        """Get file metadata from S3"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.head_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
            )
            
            return {
                'size': response.get('ContentLength', 0),
                'last_modified': response.get('LastModified'),
                'content_type': response.get('ContentType'),
                'metadata': response.get('Metadata', {}),
                'etag': response.get('ETag', '').strip('"')
            }
            
        except ClientError as e:
            logger.error(f"Failed to get metadata for {s3_key}: {e}")
            raise HTTPException(status_code=404, detail="File not found")
    
    async def list_user_files(self, user_id: str, batch_id: Optional[str] = None) -> list:
        """List files for a user, optionally filtered by batch"""
        try:
            prefix = f"documents/{user_id}/"
            if batch_id:
                prefix += f"{batch_id}/"
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix
                )
            )
            
            files = []
            for obj in response.get('Contents', []):
                files.append({
                    's3_key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag'].strip('"')
                })
            
            return files
            
        except ClientError as e:
            logger.error(f"Failed to list files for user {user_id}: {e}")
            return []
    
    async def create_temp_download_path(self, s3_key: str) -> str:
        """Create a temporary local path for downloading S3 files for processing"""
        # Extract filename from S3 key
        filename = os.path.basename(s3_key)
        
        # Create temp directory if it doesn't exist
        temp_dir = Path(settings.upload_dir) / "temp" / "s3_downloads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique temp filename but preserve original name for display
        # Use UUID for uniqueness but keep original filename structure
        temp_filename = f"{uuid4().hex}_{filename}"
        temp_path = temp_dir / temp_filename
        
        return str(temp_path)
    
    async def cleanup_user_files(self, user_id: str, batch_id: Optional[str] = None) -> Dict[str, Any]:
        """Clean up all files for a user, optionally filtered by batch"""
        try:
            files_to_delete = await self.list_user_files(user_id, batch_id)
            
            deleted_count = 0
            failed_count = 0
            
            for file_info in files_to_delete:
                try:
                    success = await self.delete_file(file_info['s3_key'])
                    if success:
                        deleted_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete S3 file {file_info['s3_key']}: {e}")
                    failed_count += 1
            
            return {
                "deleted_count": deleted_count,
                "failed_count": failed_count,
                "total_files": len(files_to_delete)
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup user files: {e}")
            return {"deleted_count": 0, "failed_count": 0, "total_files": 0, "error": str(e)}
    
    async def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """Clean up old temporary download files"""
        temp_dir = Path(settings.upload_dir) / "temp" / "s3_downloads"
        
        if not temp_dir.exists():
            return 0
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        try:
            for temp_file in temp_dir.iterdir():
                if temp_file.is_file():
                    file_time = datetime.fromtimestamp(temp_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        try:
                            temp_file.unlink()
                            cleaned_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete temp file {temp_file}: {e}")
            
            logger.info(f"Cleaned up {cleaned_count} old temporary files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
            return 0

# Global S3 service instance
s3_service = S3StorageService()

# Enhanced compatibility functions for existing code
async def save_upload_file(user_id: str, batch: str, f: UploadFile) -> Dict[str, Any]:
    """
    Save uploaded file to S3 and return file information for processing
    
    This function maintains compatibility with existing upload logic
    while using S3 storage and providing comprehensive file metadata.
    """
    try:
        # Upload to S3
        file_info = await s3_service.upload_file(user_id, batch, f)
        s3_key = file_info['s3_key']
        
        # For processing, we need a local file path
        # Download to temp location for processing
        temp_path = await s3_service.create_temp_download_path(s3_key)
        await s3_service.download_file(s3_key, temp_path)
        
        # Return comprehensive file information
        result = {
            "local_path": temp_path,
            "s3_key": s3_key,
            "s3_bucket": file_info['s3_bucket'],
            "filename": file_info['filename'],
            "file_size": file_info['size'],
            "content_type": file_info['content_type'],
            "user_id": user_id,
            "batch_id": batch
        }
        
        logger.info(f"File uploaded to S3 and downloaded to temp path: {temp_path}")
        logger.info(f"S3 key: {s3_key}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to save upload file: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

async def cleanup_file_resources(s3_key: Optional[str] = None, local_path: Optional[str] = None, 
                                user_id: Optional[str] = None, batch_id: Optional[str] = None) -> Dict[str, bool]:
    """
    Comprehensive cleanup of file resources (S3, local temp files)
    """
    cleanup_results = {
        "s3_deleted": False,
        "local_deleted": False,
        "batch_cleaned": False
    }
    
    try:
        # Clean up S3 file
        if s3_key:
            try:
                cleanup_results["s3_deleted"] = await s3_service.delete_file(s3_key)
                logger.info(f"S3 file cleanup: {s3_key} - {'success' if cleanup_results['s3_deleted'] else 'failed'}")
            except Exception as e:
                logger.error(f"Failed to delete S3 file {s3_key}: {e}")
        
        # Clean up local temp file
        if local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
                cleanup_results["local_deleted"] = True
                logger.info(f"Local temp file deleted: {local_path}")
            except Exception as e:
                logger.error(f"Failed to delete local temp file {local_path}: {e}")
        
        # Clean up batch if specified
        if user_id and batch_id:
            try:
                batch_cleanup = await s3_service.cleanup_user_files(user_id, batch_id)
                cleanup_results["batch_cleaned"] = batch_cleanup["deleted_count"] > 0
                logger.info(f"Batch cleanup for user {user_id}, batch {batch_id}: {batch_cleanup}")
            except Exception as e:
                logger.error(f"Failed to cleanup batch {batch_id} for user {user_id}: {e}")
        
        return cleanup_results
        
    except Exception as e:
        logger.error(f"Error during file resource cleanup: {e}")
        return cleanup_results

# Background cleanup task for temp files
async def start_temp_file_cleanup():
    """Background task to clean up old temporary files"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            await s3_service.cleanup_temp_files(24)  # Clean files older than 24 hours
        except Exception as e:
            logger.error(f"Temp file cleanup error: {e}")

# Start cleanup task
try:
    import asyncio
    loop = asyncio.get_running_loop()
    loop.create_task(start_temp_file_cleanup())
except RuntimeError:
    pass  # No event loop running

# Health check function
async def check_s3_health() -> Dict[str, Any]:
    """Check S3 service health"""
    try:
        # Test bucket access
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: s3_service.s3_client.head_bucket(Bucket=s3_service.bucket_name)
        )
        
        return {
            "status": "healthy",
            "bucket": s3_service.bucket_name,
            "region": s3_service.region,
            "message": "S3 service is operational"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "bucket": s3_service.bucket_name,
            "region": s3_service.region,
            "error": str(e),
            "message": "S3 service is not accessible"
        }
