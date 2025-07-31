#!/usr/bin/env python3
"""
Database Manager Script for SixthVault
Provides various database management utilities including backup, restore, and migration.

Usage:
    python db_manager.py <command> [options]

Commands:
    reset --confirm          Reset database (drop and recreate all tables)
    backup <filename>        Backup database to SQL file
    restore <filename>       Restore database from SQL file
    status                   Show database status and table information
    clean                    Clean up orphaned records and optimize
    migrate                  Run any pending migrations
    seed                     Seed database with initial data
"""

import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime
import json

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import SQLModel
from app.database import engine, get_database_url, get_session
from app.config import settings

# Import all models
from app.database import User, TempUser, UserToken
from app.models import (
    Document, ProcessingDocument, AICuration, CurationSettings,
    DocumentCurationMapping, CurationGenerationHistory, AISummary,
    SummarySettings, DocumentSummaryMapping, SummaryGenerationHistory
)

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"🗃️  {title}")
    print("=" * 60)

def get_db_connection_params():
    """Extract database connection parameters."""
    return {
        'host': settings.postgres_host,
        'port': settings.postgres_port,
        'user': settings.postgres_user,
        'password': settings.postgres_password,
        'database': settings.postgres_db
    }

def backup_database(filename):
    """Backup database to SQL file using pg_dump."""
    print_header("DATABASE BACKUP")
    
    if not filename.endswith('.sql'):
        filename += '.sql'
    
    backup_path = Path(filename)
    if backup_path.exists():
        response = input(f"File {filename} exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Backup cancelled.")
            return
    
    params = get_db_connection_params()
    
    # Set PGPASSWORD environment variable
    env = os.environ.copy()
    env['PGPASSWORD'] = params['password']
    
    cmd = [
        'pg_dump',
        '-h', params['host'],
        '-p', str(params['port']),
        '-U', params['user'],
        '-d', params['database'],
        '--clean',
        '--create',
        '--if-exists',
        '-f', str(backup_path)
    ]
    
    try:
        print(f"📦 Creating backup: {backup_path}")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            size = backup_path.stat().st_size
            print(f"✅ Backup completed successfully!")
            print(f"📁 File: {backup_path}")
            print(f"📏 Size: {size:,} bytes")
        else:
            print(f"❌ Backup failed: {result.stderr}")
            
    except FileNotFoundError:
        print("❌ pg_dump not found. Please install PostgreSQL client tools.")
    except Exception as e:
        print(f"❌ Backup error: {e}")

def restore_database(filename):
    """Restore database from SQL file using psql."""
    print_header("DATABASE RESTORE")
    
    backup_path = Path(filename)
    if not backup_path.exists():
        print(f"❌ Backup file not found: {filename}")
        return
    
    print("⚠️  WARNING: This will replace ALL existing data!")
    response = input("Are you sure you want to continue? (y/N): ")
    if response.lower() != 'y':
        print("Restore cancelled.")
        return
    
    params = get_db_connection_params()
    
    # Set PGPASSWORD environment variable
    env = os.environ.copy()
    env['PGPASSWORD'] = params['password']
    
    cmd = [
        'psql',
        '-h', params['host'],
        '-p', str(params['port']),
        '-U', params['user'],
        '-d', params['database'],
        '-f', str(backup_path)
    ]
    
    try:
        print(f"📥 Restoring from: {backup_path}")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Restore completed successfully!")
        else:
            print(f"❌ Restore failed: {result.stderr}")
            
    except FileNotFoundError:
        print("❌ psql not found. Please install PostgreSQL client tools.")
    except Exception as e:
        print(f"❌ Restore error: {e}")

def show_database_status():
    """Show comprehensive database status."""
    print_header("DATABASE STATUS")
    
    try:
        with engine.begin() as conn:
            # Database version and info
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"🐘 PostgreSQL Version: {version.split(',')[0]}")
            
            # Database size
            result = conn.execute(text("""
                SELECT pg_size_pretty(pg_database_size(current_database()))
            """))
            db_size = result.fetchone()[0]
            print(f"💾 Database Size: {db_size}")
            
            # Connection info
            params = get_db_connection_params()
            print(f"🔗 Host: {params['host']}:{params['port']}")
            print(f"🗄️  Database: {params['database']}")
            print(f"👤 User: {params['user']}")
            
        # Table information
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        print(f"\n📋 Tables ({len(tables)}):")
        print("-" * 40)
        
        with engine.begin() as conn:
            for table in sorted(tables):
                # Get row count
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.fetchone()[0]
                
                # Get table size
                result = conn.execute(text(f"""
                    SELECT pg_size_pretty(pg_total_relation_size('{table}'))
                """))
                size = result.fetchone()[0]
                
                # Get column count
                columns = inspector.get_columns(table)
                col_count = len(columns)
                
                print(f"   📊 {table:<25} {count:>8,} rows  {col_count:>2} cols  {size:>8}")
        
        # Recent activity
        print(f"\n🕒 Recent Activity:")
        print("-" * 40)
        
        with engine.begin() as conn:
            # Check for recent users
            result = conn.execute(text("""
                SELECT COUNT(*) FROM users 
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """))
            recent_users = result.fetchone()[0]
            print(f"   👥 New users (24h): {recent_users}")
            
            # Check for recent documents
            if 'document' in tables:
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM document 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """))
                recent_docs = result.fetchone()[0]
                print(f"   📄 New documents (24h): {recent_docs}")
            
            # Check for processing jobs
            if 'processing_documents' in tables:
                result = conn.execute(text("""
                    SELECT status, COUNT(*) 
                    FROM processing_documents 
                    GROUP BY status
                """))
                processing_stats = result.fetchall()
                if processing_stats:
                    print(f"   ⚙️  Processing jobs:")
                    for status, count in processing_stats:
                        print(f"      {status}: {count}")
                        
    except Exception as e:
        print(f"❌ Error getting database status: {e}")

def clean_database():
    """Clean up orphaned records and optimize database."""
    print_header("DATABASE CLEANUP")
    
    try:
        with engine.begin() as conn:
            cleanup_count = 0
            
            # Clean up expired temp users
            result = conn.execute(text("""
                DELETE FROM temp_users 
                WHERE verification_code_expires_at < NOW()
            """))
            expired_temp = result.rowcount
            cleanup_count += expired_temp
            if expired_temp > 0:
                print(f"   🗑️  Removed {expired_temp} expired temp users")
            
            # Clean up expired user tokens
            result = conn.execute(text("""
                DELETE FROM user_tokens 
                WHERE expires_at < NOW()
            """))
            expired_tokens = result.rowcount
            cleanup_count += expired_tokens
            if expired_tokens > 0:
                print(f"   🗑️  Removed {expired_tokens} expired tokens")
            
            # Clean up orphaned processing documents
            result = conn.execute(text("""
                DELETE FROM processing_documents 
                WHERE status IN ('completed', 'error', 'cancelled') 
                AND updated_at < NOW() - INTERVAL '7 days'
            """))
            old_processing = result.rowcount
            cleanup_count += old_processing
            if old_processing > 0:
                print(f"   🗑️  Removed {old_processing} old processing records")
            
            # Vacuum and analyze
            conn.execute(text("VACUUM ANALYZE"))
            print("   🧹 Database vacuumed and analyzed")
            
            if cleanup_count == 0:
                print("   ✨ Database is already clean!")
            else:
                print(f"   ✅ Cleanup completed: {cleanup_count} records removed")
                
    except Exception as e:
        print(f"❌ Cleanup error: {e}")

def seed_database():
    """Seed database with initial data."""
    print_header("DATABASE SEEDING")
    
    try:
        # Check if we already have an admin user
        with get_session() as session:
            admin_exists = session.exec(
                text("SELECT COUNT(*) FROM users WHERE is_admin = true")
            ).first()[0]
            
            if admin_exists > 0:
                print("   ℹ️  Admin user already exists, skipping seed.")
                return
        
        print("   🌱 No admin user found, would you like to create one?")
        response = input("   Create admin user? (y/N): ")
        
        if response.lower() == 'y':
            print("   📝 Please run the create_initial_admin.py script instead.")
            print("   Command: python create_initial_admin.py")
        else:
            print("   ⏭️  Skipping admin user creation.")
            
    except Exception as e:
        print(f"❌ Seeding error: {e}")

def reset_database():
    """Reset database by calling the reset script."""
    print_header("DATABASE RESET")
    
    try:
        # Import and run the reset function
        from reset_database import main as reset_main
        
        # Temporarily modify sys.argv to include --confirm
        original_argv = sys.argv.copy()
        sys.argv = ['reset_database.py', '--confirm']
        
        reset_main()
        
        # Restore original argv
        sys.argv = original_argv
        
    except Exception as e:
        print(f"❌ Reset error: {e}")

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1].lower()
    
    if command == 'reset':
        if len(sys.argv) > 2 and sys.argv[2] == '--confirm':
            reset_database()
        else:
            print("⚠️  Use 'reset --confirm' to reset the database")
    
    elif command == 'backup':
        if len(sys.argv) < 3:
            filename = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
        else:
            filename = sys.argv[2]
        backup_database(filename)
    
    elif command == 'restore':
        if len(sys.argv) < 3:
            print("❌ Please specify backup file: python db_manager.py restore <filename>")
            return
        restore_database(sys.argv[2])
    
    elif command == 'status':
        show_database_status()
    
    elif command == 'clean':
        clean_database()
    
    elif command == 'seed':
        seed_database()
    
    elif command == 'migrate':
        print("🚧 Migration functionality not implemented yet.")
        print("   Use 'reset --confirm' to recreate tables from current models.")
    
    else:
        print(f"❌ Unknown command: {command}")
        print(__doc__)

if __name__ == "__main__":
    main()
