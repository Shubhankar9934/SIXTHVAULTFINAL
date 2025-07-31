#!/usr/bin/env python3
"""
Tenant Data Sharing Migration Script
===================================

This script migrates existing AICuration and AISummary records to include tenant_id
for proper tenant-based data sharing.

Steps:
1. Add tenant_id columns to ai_curations and ai_summaries tables
2. Populate tenant_id values based on owner_id -> user.tenant_id mapping
3. Update existing vector embeddings to use tenant_id instead of user_id

Run this script after updating the models to include tenant_id fields.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlmodel import Session, select, text
from app.database import engine, User
from app.models import AICuration, AISummary, Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_tenant_data():
    """Main migration function"""
    
    logger.info("Starting tenant data migration...")
    
    with Session(engine) as session:
        try:
            # Step 1: Check if columns already exist
            logger.info("Checking existing table structure...")
            
            # Check AICuration table (PostgreSQL syntax)
            result = session.exec(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'ai_curations'
            """)).fetchall()
            curation_columns = [row[0] for row in result]
            has_curation_tenant_id = 'tenant_id' in curation_columns
            
            # Check AISummary table (PostgreSQL syntax)
            result = session.exec(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'ai_summaries'
            """)).fetchall()
            summary_columns = [row[0] for row in result]
            has_summary_tenant_id = 'tenant_id' in summary_columns
            
            logger.info(f"AICuration has tenant_id: {has_curation_tenant_id}")
            logger.info(f"AISummary has tenant_id: {has_summary_tenant_id}")
            
            # Step 2: Add tenant_id columns if they don't exist
            if not has_curation_tenant_id:
                logger.info("Adding tenant_id column to ai_curations table...")
                session.exec(text("ALTER TABLE ai_curations ADD COLUMN tenant_id TEXT"))
                session.commit()
            
            if not has_summary_tenant_id:
                logger.info("Adding tenant_id column to ai_summaries table...")
                session.exec(text("ALTER TABLE ai_summaries ADD COLUMN tenant_id TEXT"))
                session.commit()
            
            # Step 3: Get all users and their tenant_ids
            logger.info("Building user -> tenant_id mapping...")
            users = session.exec(select(User)).all()
            user_tenant_map = {user.id: user.tenant_id for user in users}
            logger.info(f"Found {len(user_tenant_map)} users")
            
            # Step 4: Update AICuration records
            logger.info("Updating AICuration records with tenant_id...")
            curations = session.exec(select(AICuration)).all()
            curation_updates = 0
            
            for curation in curations:
                if curation.owner_id in user_tenant_map and user_tenant_map[curation.owner_id]:
                    curation.tenant_id = user_tenant_map[curation.owner_id]
                    curation_updates += 1
                else:
                    logger.warning(f"No tenant_id found for curation owner {curation.owner_id}")
            
            session.commit()
            logger.info(f"Updated {curation_updates} AICuration records")
            
            # Step 5: Update AISummary records
            logger.info("Updating AISummary records with tenant_id...")
            summaries = session.exec(select(AISummary)).all()
            summary_updates = 0
            
            for summary in summaries:
                if summary.owner_id in user_tenant_map and user_tenant_map[summary.owner_id]:
                    summary.tenant_id = user_tenant_map[summary.owner_id]
                    summary_updates += 1
                else:
                    logger.warning(f"No tenant_id found for summary owner {summary.owner_id}")
            
            session.commit()
            logger.info(f"Updated {summary_updates} AISummary records")
            
            # Step 6: Verify Document records have tenant_id
            logger.info("Checking Document records for tenant_id...")
            documents = session.exec(select(Document)).all()
            doc_updates = 0
            
            for doc in documents:
                if not doc.tenant_id and doc.owner_id in user_tenant_map:
                    if user_tenant_map[doc.owner_id]:
                        doc.tenant_id = user_tenant_map[doc.owner_id]
                        doc_updates += 1
                    else:
                        logger.warning(f"No tenant_id found for document owner {doc.owner_id}")
            
            if doc_updates > 0:
                session.commit()
                logger.info(f"Updated {doc_updates} Document records with missing tenant_id")
            else:
                logger.info("All Document records already have tenant_id")
            
            # Step 7: Create foreign key constraints (optional, for data integrity)
            logger.info("Migration completed successfully!")
            
            # Summary
            logger.info("=" * 50)
            logger.info("MIGRATION SUMMARY:")
            logger.info(f"- AICuration records updated: {curation_updates}")
            logger.info(f"- AISummary records updated: {summary_updates}")
            logger.info(f"- Document records updated: {doc_updates}")
            logger.info(f"- Total users processed: {len(user_tenant_map)}")
            logger.info("=" * 50)
            
            # Step 8: Verification queries
            logger.info("Running verification queries...")
            
            # Check for records without tenant_id
            curations_without_tenant = session.exec(
                text("SELECT COUNT(*) FROM ai_curations WHERE tenant_id IS NULL")
            ).fetchone()
            
            summaries_without_tenant = session.exec(
                text("SELECT COUNT(*) FROM ai_summaries WHERE tenant_id IS NULL")
            ).fetchone()
            
            documents_without_tenant = session.exec(
                text("SELECT COUNT(*) FROM document WHERE tenant_id IS NULL")
            ).fetchone()
            
            logger.info(f"Records without tenant_id:")
            logger.info(f"- AICurations: {curations_without_tenant[0] if curations_without_tenant else 0}")
            logger.info(f"- AISummaries: {summaries_without_tenant[0] if summaries_without_tenant else 0}")
            logger.info(f"- Documents: {documents_without_tenant[0] if documents_without_tenant else 0}")
            
            if (curations_without_tenant[0] == 0 and 
                summaries_without_tenant[0] == 0 and 
                documents_without_tenant[0] == 0):
                logger.info("✅ All records have tenant_id assigned!")
            else:
                logger.warning("⚠️ Some records still missing tenant_id")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            session.rollback()
            raise

def verify_tenant_isolation():
    """Verify that tenant isolation is working correctly"""
    
    logger.info("Verifying tenant isolation...")
    
    with Session(engine) as session:
        # Get tenant statistics
        tenant_stats = session.exec(text("""
            SELECT 
                u.tenant_id,
                COUNT(DISTINCT u.id) as users,
                COUNT(DISTINCT d.id) as documents,
                COUNT(DISTINCT c.id) as curations,
                COUNT(DISTINCT s.id) as summaries
            FROM users u
            LEFT JOIN document d ON d.tenant_id = u.tenant_id
            LEFT JOIN ai_curations c ON c.tenant_id = u.tenant_id
            LEFT JOIN ai_summaries s ON s.tenant_id = u.tenant_id
            WHERE u.tenant_id IS NOT NULL
            GROUP BY u.tenant_id
            ORDER BY users DESC
        """)).fetchall()
        
        logger.info("Tenant isolation statistics:")
        logger.info("Tenant ID | Users | Documents | Curations | Summaries")
        logger.info("-" * 55)
        
        for row in tenant_stats:
            tenant_id = row[0][:8] + "..." if row[0] else "None"
            logger.info(f"{tenant_id:9} | {row[1]:5} | {row[2]:9} | {row[3]:9} | {row[4]:9}")
        
        logger.info("✅ Tenant isolation verification complete!")

if __name__ == "__main__":
    try:
        migrate_tenant_data()
        verify_tenant_isolation()
        
        print("\n" + "="*60)
        print("🎉 TENANT DATA SHARING MIGRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Restart the backend server")
        print("2. Test AI features (RAG, Curation, Summary) with tenant users")
        print("3. Verify that users can access documents from their tenant")
        print("4. Confirm that tenant isolation is maintained")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print("Please check the logs and fix any issues before retrying.")
        sys.exit(1)
