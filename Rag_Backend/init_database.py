#!/usr/bin/env python3
"""
Database Initialization Script for SixthVault RAG Backend

This script ensures all database tables are created properly and populates
initial data including tenants and default settings.

Usage:
    python init_database.py [--reset]
    
Options:
    --reset    Drop all existing tables and recreate them (USE WITH CAUTION)
"""

import sys
import argparse
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.database import engine, init_db, reset_database, get_session
from app.config import settings
from sqlmodel import Session, select
import uuid
from datetime import datetime, timedelta

def create_initial_tenants():
    """Create initial tenant data"""
    print("🏢 Creating initial tenants...")
    
    from app.tenant_models import Tenant, TenantSettings, INITIAL_TENANTS, DEFAULT_TENANT_CONFIGS
    
    with Session(engine) as session:
        # Check if tenants already exist
        existing_tenants = session.exec(select(Tenant)).all()
        if existing_tenants:
            print(f"   ✅ Found {len(existing_tenants)} existing tenants, skipping creation")
            return
        
        # Create initial tenants
        created_count = 0
        for tenant_data in INITIAL_TENANTS:
            tenant_type = tenant_data["tenant_type"]
            config = DEFAULT_TENANT_CONFIGS.get(tenant_type, {})
            
            tenant = Tenant(
                slug=tenant_data["slug"],
                name=tenant_data["name"],
                tenant_type=tenant_type,
                primary_color=tenant_data.get("primary_color"),
                secondary_color=tenant_data.get("secondary_color"),
                max_users=config.get("max_users"),
                max_storage_gb=config.get("max_storage_gb"),
                max_documents=config.get("max_documents"),
                allowed_file_types=config.get("allowed_file_types", []),
                features=config.get("features", {}),
                is_active=True
            )
            
            session.add(tenant)
            created_count += 1
            
            # Create default tenant settings
            tenant_settings = TenantSettings(
                tenant_id=tenant.id,
                ai_provider="groq",
                ai_model="llama-3.1-8b-instant",
                allow_user_uploads=True,
                auto_process_documents=True
            )
            session.add(tenant_settings)
        
        session.commit()
        print(f"   ✅ Created {created_count} tenants with default settings")

def create_default_user_settings():
    """Create default user settings for existing users"""
    print("👤 Creating default user settings...")
    
    from app.database import User
    from app.models import ConversationSettings, CurationSettings, SummarySettings
    
    with Session(engine) as session:
        # Get all users without conversation settings
        users = session.exec(select(User)).all()
        
        settings_created = 0
        for user in users:
            # Check if conversation settings exist
            existing_conv_settings = session.exec(
                select(ConversationSettings).where(ConversationSettings.owner_id == user.id)
            ).first()
            
            if not existing_conv_settings:
                conv_settings = ConversationSettings(
                    owner_id=user.id,
                    auto_save_conversations=True,
                    auto_generate_titles=True,
                    max_conversations=100,
                    auto_archive_after_days=30,
                    default_provider="groq",
                    default_model="llama-3.1-8b-instant"
                )
                session.add(conv_settings)
                settings_created += 1
            
            # Check if curation settings exist
            existing_cur_settings = session.exec(
                select(CurationSettings).where(CurationSettings.owner_id == user.id)
            ).first()
            
            if not existing_cur_settings:
                cur_settings = CurationSettings(
                    owner_id=user.id,
                    auto_refresh=True,
                    on_add="incremental",
                    on_delete="auto_clean",
                    change_threshold=15,
                    max_curations=4,
                    min_documents_per_curation=2
                )
                session.add(cur_settings)
            
            # Check if summary settings exist
            existing_sum_settings = session.exec(
                select(SummarySettings).where(SummarySettings.owner_id == user.id)
            ).first()
            
            if not existing_sum_settings:
                sum_settings = SummarySettings(
                    owner_id=user.id,
                    auto_refresh=True,
                    on_add="incremental",
                    on_delete="auto_clean",
                    change_threshold=15,
                    max_summaries=8,
                    min_documents_per_summary=1,
                    default_summary_type="auto",
                    include_individual=True,
                    include_combined=True
                )
                session.add(sum_settings)
        
        session.commit()
        if settings_created > 0:
            print(f"   ✅ Created default settings for {settings_created} users")
        else:
            print("   ✅ All users already have default settings")

def verify_database_structure():
    """Verify that all required tables exist"""
    print("🔍 Verifying database structure...")
    
    from sqlalchemy import inspect
    
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    # Expected tables
    expected_tables = [
        'tenants', 'users', 'temp_users', 'user_tokens', 'tenant_users', 
        'tenant_invitations', 'tenant_settings', 'tenant_analytics', 'tenant_audit_logs',
        'document', 'processing_documents',
        'ai_curations', 'curation_settings', 'document_curation_mapping', 'curation_generation_history',
        'ai_summaries', 'summary_settings', 'document_summary_mapping', 'summary_generation_history',
        'conversations', 'messages', 'conversation_settings'
    ]
    
    missing_tables = [table for table in expected_tables if table not in existing_tables]
    
    if missing_tables:
        print(f"   ❌ Missing tables: {missing_tables}")
        return False
    else:
        print(f"   ✅ All {len(expected_tables)} required tables exist")
        return True

def test_database_connection():
    """Test database connection and basic operations"""
    print("🔌 Testing database connection...")
    
    try:
        with Session(engine) as session:
            # Test basic query
            result = session.exec(select(1)).first()
            if result == 1:
                print("   ✅ Database connection successful")
                return True
            else:
                print("   ❌ Database connection test failed")
                return False
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        return False

def main():
    """Main initialization function"""
    parser = argparse.ArgumentParser(description="Initialize SixthVault database")
    parser.add_argument("--reset", action="store_true", 
                       help="Reset database (drop and recreate all tables)")
    args = parser.parse_args()
    
    print("🚀 SixthVault Database Initialization")
    print("=" * 50)
    
    # Test database connection first
    if not test_database_connection():
        print("❌ Cannot proceed without database connection")
        sys.exit(1)
    
    try:
        if args.reset:
            print("⚠️  RESETTING DATABASE (dropping all tables)...")
            response = input("Are you sure? This will delete ALL data! (yes/no): ")
            if response.lower() != 'yes':
                print("❌ Database reset cancelled")
                sys.exit(1)
            
            reset_database()
            print("✅ Database reset completed")
        else:
            print("📊 Initializing database tables...")
            init_db()
            print("✅ Database initialization completed")
        
        # Verify database structure
        if not verify_database_structure():
            print("❌ Database structure verification failed")
            sys.exit(1)
        
        # Create initial data
        create_initial_tenants()
        create_default_user_settings()
        
        print("\n" + "=" * 50)
        print("🎉 Database initialization completed successfully!")
        print("=" * 50)
        print("✅ All tables created")
        print("✅ Initial tenants created")
        print("✅ Default settings configured")
        print("✅ Database ready for use")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
