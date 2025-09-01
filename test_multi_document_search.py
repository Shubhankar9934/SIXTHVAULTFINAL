#!/usr/bin/env python3
"""
Test script to reproduce and fix the multi-document search issue.
This script will help identify why only the first document is being used in responses.
"""

import asyncio
import sys
import os

# Add the Rag_Backend directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_backend_dir = os.path.join(current_dir, 'Rag_Backend')
sys.path.insert(0, rag_backend_dir)

# Now we can import the modules
try:
    from app.utils.qdrant_store import search
    from app.database import engine
    from sqlmodel import Session, select
    from app.models import Document
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the SIXTHVAULTFINAL directory")
    print("And that the Rag_Backend directory contains the app modules")
    sys.exit(1)

async def test_multi_document_search():
    """Test multi-document search functionality"""
    print("🔍 Testing Multi-Document Search Issue")
    print("=" * 50)
    
    # Test with a sample tenant and query
    test_tenant_id = "test-tenant"
    test_query = "What is important information?"
    
    # First, let's see what documents exist in the database
    print("\n1. Checking documents in database...")
    try:
        with Session(engine) as session:
            statement = select(Document.id, Document.filename, Document.path).limit(10)
            documents = list(session.exec(statement))
            
            if not documents:
                print("❌ No documents found in database. Please upload some documents first.")
                return
            
            print(f"✅ Found {len(documents)} documents:")
            for doc in documents:
                print(f"   - ID: {doc.id[:8]}... | File: {doc.filename} | Path: {doc.path}")
    
    except Exception as e:
        print(f"❌ Database error: {e}")
        return
    
    # Test 1: Search with no document filter (should return results from all documents)
    print("\n2. Testing search without document filter...")
    try:
        results_all = search(test_tenant_id, test_query, k=10, document_ids=None)
        print(f"✅ Found {len(results_all)} results without filter")
        
        if results_all:
            # Show which documents the results come from
            doc_sources = set()
            for result in results_all:
                path = result.get('path', 'unknown')
                doc_sources.add(path)
            print(f"   Results come from {len(doc_sources)} different documents:")
            for source in sorted(doc_sources):
                print(f"     - {source}")
    
    except Exception as e:
        print(f"❌ Search without filter failed: {e}")
        return
    
    # Test 2: Search with multiple document IDs
    if len(documents) >= 2:
        print("\n3. Testing search with multiple document IDs...")
        test_doc_ids = [documents[0].id, documents[1].id]
        print(f"   Using document IDs: {[doc_id[:8]+'...' for doc_id in test_doc_ids]}")
        
        try:
            results_filtered = search(test_tenant_id, test_query, k=10, document_ids=test_doc_ids)
            print(f"✅ Found {len(results_filtered)} results with document filter")
            
            if results_filtered:
                # Show which documents the results come from
                doc_sources = set()
                for result in results_filtered:
                    path = result.get('path', 'unknown')
                    doc_sources.add(path)
                    
                print(f"   Results come from {len(doc_sources)} different documents:")
                for source in sorted(doc_sources):
                    print(f"     - {source}")
                
                # Check if we're getting results from both documents
                expected_paths = set()
                with Session(engine) as session:
                    for doc_id in test_doc_ids:
                        statement = select(Document.path).where(Document.id == doc_id)
                        path = session.exec(statement).first()
                        if path:
                            expected_paths.add(path)
                
                print(f"   Expected paths: {sorted(expected_paths)}")
                print(f"   Actual paths: {sorted(doc_sources)}")
                
                if len(doc_sources) == 1 and len(expected_paths) > 1:
                    print("❌ BUG CONFIRMED: Only getting results from one document!")
                    print("   This confirms the multi-document search issue.")
                elif len(doc_sources) == len(expected_paths):
                    print("✅ Multi-document search working correctly!")
                else:
                    print("⚠️  Partial results - some documents may not have matching content")
            
        except Exception as e:
            print(f"❌ Search with filter failed: {e}")
            return
    
    else:
        print("\n3. Skipping multi-document test - need at least 2 documents")
    
    # Test 3: Test the database query used in the search function
    print("\n4. Testing document path lookup (used in search filter)...")
    try:
        test_doc_ids = [doc.id for doc in documents[:2]]
        
        with Session(engine) as session:
            statement = select(Document.path).where(Document.id.in_(test_doc_ids))
            document_paths = list(session.exec(statement))
            
            print(f"✅ Path lookup for {len(test_doc_ids)} documents returned {len(document_paths)} paths:")
            for path in document_paths:
                print(f"   - {path}")
                
            if len(document_paths) != len(test_doc_ids):
                print("❌ Path lookup issue: Not all document IDs returned paths!")
    
    except Exception as e:
        print(f"❌ Database path lookup failed: {e}")
    
    print("\n" + "=" * 50)
    print("🔍 Multi-Document Search Test Complete")

if __name__ == "__main__":
    asyncio.run(test_multi_document_search())
