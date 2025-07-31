#!/usr/bin/env python3
"""
Database Reset Script for SixthVault
This script drops all existing tables and recreates them from the current models.
Use with caution - this will delete all data!

Usage:
    python reset_database.py --confirm
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import SQLModel
from app.database import engine, get_database_url
from app.config import settings

# Import all models to ensure they're registered with SQLModel.metadata
from app.database import User, TempUser, UserToken
from app.models import (
    Document, ProcessingDocument, AICuration, CurationSettings,
    DocumentCurationMapping, CurationGenerationHistory, AISummary,
    SummarySettings, DocumentSummaryMapping, SummaryGenerationHistory
)

def print_banner():
    """Print a banner for the script."""
    print("=" * 60)
    print("🗃️  SIXTHVAULT DATABASE RESET SCRIPT")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔗 Database: {get_database_url()}")
    print("=" * 60)

def get_all_table_names():
    """Get all table names that should exist based on our models."""
    return [
        'temp_users',
        'users', 
        'user_tokens',
        'document',
        'processing_documents',
        'ai_curations',
        'curation_settings',
        'document_curation_mapping',
        'curation_generation_history',
        'ai_summaries',
        'summary_settings',
        'document_summary_mapping',
        'summary_generation_history'
    ]

def drop_all_tables():
    """Drop all existing tables from the database."""
    print("🗑️  DROPPING ALL EXISTING TABLES")
    print("-" * 40)
    
    try:
        with engine.begin() as conn:
            # Get existing table names
            inspector = inspect(engine)
            existing_tables = inspector.get_table_names()
            
            if not existing_tables:
                print("   ℹ️  No tables found to drop.")
                return
            
            print(f"   📋 Found {len(existing_tables)} tables: {', '.join(existing_tables)}")
            
            # Drop tables in reverse dependency order to handle foreign keys
            tables_to_drop = [
                'summary_generation_history',
                'document_summary_mapping', 
                'summary_settings',
                'ai_summaries',
                'curation_generation_history',
                'document_curation_mapping',
                'curation_settings', 
                'ai_curations',
                'processing_documents',
                'document',
                'user_tokens',
                'users',
                'temp_users'
            ]
            
            # Drop each table with CASCADE to handle any remaining dependencies
            for table_name in tables_to_drop:
                if table_name in existing_tables:
                    try:
                        conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                        print(f"   ✅ Dropped: {table_name}")
                    except Exception as e:
                        print(f"   ⚠️  Warning dropping {table_name}: {e}")
            
            # Drop any remaining tables that weren't in our list
            remaining_tables = set(existing_tables) - set(tables_to_drop)
            for table_name in remaining_tables:
                try:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                    print(f"   ✅ Dropped (extra): {table_name}")
                except Exception as e:
                    print(f"   ⚠️  Warning dropping {table_name}: {e}")
                    
        print("   🎉 All tables dropped successfully!")
        
    except SQLAlchemyError as e:
        print(f"   ❌ Error dropping tables: {e}")
        raise

def create_all_tables():
    """Create all tables based on current models."""
    print("\n🏗️  CREATING ALL TABLES FROM MODELS")
    print("-" * 40)
    
    try:
        # Create all tables using SQLModel metadata
        SQLModel.metadata.create_all(engine)
        print("   🎉 All tables created successfully!")
        
    except SQLAlchemyError as e:
        print(f"   ❌ Error creating tables: {e}")
        raise

def verify_database_structure():
    """Verify the database structure after reset."""
    print("\n🔍 VERIFYING DATABASE STRUCTURE")
    print("-" * 40)
    
    try:
        inspector = inspect(engine)
        actual_tables = set(inspector.get_table_names())
        expected_tables = set(get_all_table_names())
        
        print(f"   📊 Total tables created: {len(actual_tables)}")
        
        # Check each expected table
        for table in expected_tables:
            if table in actual_tables:
                columns = inspector.get_columns(table)
                indexes = inspector.get_indexes(table)
                foreign_keys = inspector.get_foreign_keys(table)
                print(f"   ✅ {table}: {len(columns)} columns, {len(indexes)} indexes, {len(foreign_keys)} FKs")
            else:
                print(f"   ❌ Missing table: {table}")
        
        # Check for unexpected tables
        unexpected = actual_tables - expected_tables
        if unexpected:
            print(f"   ⚠️  Unexpected tables: {', '.join(unexpected)}")
        
        # Verify foreign key relationships
        print("\n   🔗 Foreign Key Relationships:")
        for table in actual_tables:
            fks = inspector.get_foreign_keys(table)
            if fks:
                for fk in fks:
                    print(f"      {table}.{fk['constrained_columns'][0]} -> {fk['referred_table']}.{fk['referred_columns'][0]}")
            
    except Exception as e:
        print(f"   ❌ Error verifying database: {e}")

def test_database_connection():
    """Test database connection and basic operations."""
    print("\n🧪 TESTING DATABASE CONNECTION")
    print("-" * 40)
    
    try:
        with engine.begin() as conn:
            # Test basic query
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"   ✅ PostgreSQL Version: {version}")
            
            # Test table count
            result = conn.execute(text("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'"))
            table_count = result.fetchone()[0]
            print(f"   ✅ Tables in public schema: {table_count}")
            
            # Test a simple insert/select/delete on users table
            test_id = "test-user-id"
            conn.execute(text("""
                INSERT INTO users (id, email, password_hash, first_name, last_name, verified, role, is_admin, is_active) 
                VALUES (:id, 'test@example.com', 'test_hash', 'Test', 'User', false, 'user', false, true)
            """), {"id": test_id})
            
            result = conn.execute(text("SELECT email FROM users WHERE id = :id"), {"id": test_id})
            email = result.fetchone()[0]
            print(f"   ✅ Test insert/select: {email}")
            
            conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": test_id})
            print("   ✅ Test delete completed")
            
    except Exception as e:
        print(f"   ❌ Database test failed: {e}")
        raise

def main():
    """Main function to reset the database."""
    print_banner()
    
    # Check for confirmation flag
    if len(sys.argv) < 2 or sys.argv[1] != "--confirm":
        print("⚠️  WARNING: This will delete ALL existing data!")
        print("📝 This script will:")
        print("   • Drop all existing tables")
        print("   • Recreate tables from current models")
        print("   • Verify the new structure")
        print("   • Run basic tests")
        print("\n❓ To proceed, run:")
        print("   python reset_database.py --confirm")
        print("\n🛑 Exiting without changes.")
        return
    
    try:
        # Step 1: Drop all existing tables
        drop_all_tables()
        
        # Step 2: Create all tables from models
        create_all_tables()
        
        # Step 3: Verify the structure
        verify_database_structure()
        
        # Step 4: Test the database
        test_database_connection()
        
        print("\n" + "=" * 60)
        print("🎉 DATABASE RESET COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"✅ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("📋 Next steps:")
        print("   • Run your application to verify everything works")
        print("   • Create initial admin user if needed")
        print("   • Import any required seed data")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ DATABASE RESET FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        print("🔧 Please check:")
        print("   • Database connection settings")
        print("   • PostgreSQL server is running")
        print("   • Database permissions")
        print("   • Model definitions are correct")
        sys.exit(1)

if __name__ == "__main__":
    main()
