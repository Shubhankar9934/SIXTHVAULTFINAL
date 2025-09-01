#!/usr/bin/env python3
"""
Database Migration Script: Add File Metadata to Documents Table
==============================================================

This script adds the following columns to the documents table:
- file_size: INTEGER (stores file size in bytes)
- content_type: VARCHAR (stores MIME type)
- s3_key: VARCHAR (stores S3 object key)
- s3_bucket: VARCHAR (stores S3 bucket name)

It also populates existing documents with file size data where possible.
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String
from sqlalchemy.exc import OperationalError
import logging

# Add the app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config import settings
from app.database import engine
from sqlmodel import Session, select
from app.models import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_column_exists(engine, table_name: str, column_name: str) -> bool:
    """Check if a column exists in the table"""
    try:
        with engine.connect() as conn:
            # Query the information schema to check if column exists
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = :table_name AND column_name = :column_name
            """), {"table_name": table_name, "column_name": column_name})
            
            return result.fetchone() is not None
    except Exception as e:
        logger.error(f"Error checking column existence: {e}")
        return False

def add_file_metadata_columns():
    """Add file metadata columns to the documents table"""
    logger.info("🔄 Starting database migration: Adding file metadata columns to documents table")
    
    try:
        with engine.connect() as conn:
            # Check which columns need to be added
            columns_to_add = []
            
            if not check_column_exists(engine, "documents", "file_size"):
                columns_to_add.append("file_size INTEGER DEFAULT 0")
                logger.info("📊 Will add file_size column")
            else:
                logger.info("✅ file_size column already exists")
            
            if not check_column_exists(engine, "documents", "content_type"):
                columns_to_add.append("content_type VARCHAR(255)")
                logger.info("📄 Will add content_type column")
            else:
                logger.info("✅ content_type column already exists")
            
            if not check_column_exists(engine, "documents", "s3_key"):
                columns_to_add.append("s3_key VARCHAR(512)")
                logger.info("☁️ Will add s3_key column")
            else:
                logger.info("✅ s3_key column already exists")
            
            if not check_column_exists(engine, "documents", "s3_bucket"):
                columns_to_add.append("s3_bucket VARCHAR(255)")
                logger.info("🪣 Will add s3_bucket column")
            else:
                logger.info("✅ s3_bucket column already exists")
            
            # Add columns that don't exist
            for column_def in columns_to_add:
                try:
                    sql = f"ALTER TABLE documents ADD COLUMN {column_def}"
                    logger.info(f"🔧 Executing: {sql}")
                    conn.execute(text(sql))
                    conn.commit()
                    logger.info(f"✅ Successfully added column: {column_def}")
                except Exception as e:
                    logger.error(f"❌ Failed to add column {column_def}: {e}")
                    raise
            
            if not columns_to_add:
                logger.info("✅ All required columns already exist")
            else:
                logger.info(f"✅ Successfully added {len(columns_to_add)} new columns")
                
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

def populate_existing_file_sizes():
    """Populate file sizes for existing documents"""
    logger.info("🔄 Populating file sizes for existing documents")
    
    try:
        with Session(engine) as session:
            # Get all documents that don't have file_size set
            documents = session.exec(
                select(Document).where(
                    (Document.file_size == None) | (Document.file_size == 0)
                )
            ).all()
            
            logger.info(f"📊 Found {len(documents)} documents without file size data")
            
            updated_count = 0
            error_count = 0
            
            for doc in documents:
                try:
                    # Calculate file size if file exists
                    file_size = 0
                    content_type = None
                    
                    if os.path.exists(doc.path):
                        file_size = os.path.getsize(doc.path)
                        
                        # Determine content type from file extension
                        file_ext = Path(doc.filename).suffix.lower()
                        content_type = "application/pdf" if file_ext == ".pdf" else \
                                      "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if file_ext == ".docx" else \
                                      "text/plain" if file_ext == ".txt" else \
                                      "application/rtf" if file_ext == ".rtf" else \
                                      "application/octet-stream"
                        
                        # Update the document
                        doc.file_size = file_size
                        if hasattr(doc, 'content_type'):
                            doc.content_type = content_type
                        
                        session.add(doc)
                        updated_count += 1
                        
                        logger.info(f"📊 Updated {doc.filename}: {file_size} bytes ({content_type})")
                    else:
                        logger.warning(f"⚠️ File not found for document {doc.filename} at path: {doc.path}")
                        error_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ Error processing document {doc.filename}: {e}")
                    error_count += 1
            
            # Commit all changes
            if updated_count > 0:
                session.commit()
                logger.info(f"✅ Successfully updated {updated_count} documents with file size data")
            
            if error_count > 0:
                logger.warning(f"⚠️ {error_count} documents could not be updated (files not found)")
                
    except Exception as e:
        logger.error(f"❌ Failed to populate file sizes: {e}")
        raise

def main():
    """Run the complete migration"""
    logger.info("🚀 Starting Document File Size Migration")
    logger.info("=" * 60)
    
    try:
        # Step 1: Add new columns
        add_file_metadata_columns()
        
        # Step 2: Populate existing data
        populate_existing_file_sizes()
        
        logger.info("=" * 60)
        logger.info("🎉 Migration completed successfully!")
        logger.info("📊 File sizes should now display correctly in both /documents and /admin pages")
        logger.info("🔄 Please restart the FastAPI backend to ensure all changes take effect")
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"💥 Migration failed: {e}")
        logger.error("🔧 Please check the error above and fix any issues before retrying")
        sys.exit(1)

if __name__ == "__main__":
    main()
