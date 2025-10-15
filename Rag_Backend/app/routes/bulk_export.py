"""
Bulk Export Routes for Document Summaries
Handles bulk export of transcript summaries to Excel format
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import pandas as pd
import io
from datetime import datetime
import re

from app.database import get_session
from app.deps import get_current_user
from app.models import User, Document, AISummary
from app.services.summarise import _enhanced_summarizer

router = APIRouter()

def detect_transcript_content(text: str) -> bool:
    """
    Detect if content is transcript/customer feedback content using the same logic as summarize service
    """
    # Create a simple transcript detection function without relying on private methods
    text_lower = text.lower()
    
    # Primary transcript indicators
    primary_indicators = [
        'transcript', 'interview', 'feedback', 'client', 'customer', 
        'satisfaction', 'rating', 'recommendation', 'experience'
    ]
    
    # Secondary indicators
    secondary_indicators = [
        'said', 'mentioned', 'expressed', 'highlighted', 'concerns',
        'satisfied', 'improvement', 'service', 'quality', 'partnership'
    ]
    
    # Count indicators
    primary_score = sum(2 for word in primary_indicators if word in text_lower)
    secondary_score = sum(1 for word in secondary_indicators if word in text_lower)
    
    # Look for feedback phrases
    feedback_phrases = [
        'likelihood of recommendation', 'out of 10', 'rating the', 
        'client is satisfied', 'areas for improvement', 'room for improvement',
        'generally satisfied', 'positive experience', 'sees room for',
        'would like to see', 'concerns about', 'client expressed'
    ]
    phrase_score = sum(3 for phrase in feedback_phrases if phrase in text_lower)
    
    total_score = primary_score + secondary_score + phrase_score
    
    # Additional heuristics
    has_rating_pattern = bool(re.search(r'\d+\s*out of\s*\d+', text_lower))
    has_client_mentions = text_lower.count('client') + text_lower.count('customer') >= 2
    has_feedback_structure = 'satisfaction' in text_lower and 'improvement' in text_lower
    
    # Final decision
    return (
        total_score >= 5 or
        has_rating_pattern or
        (has_client_mentions and has_feedback_structure)
    )

def clean_for_excel(text: str) -> str:
    """Clean text content for Excel export"""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Replace markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Replace markdown bold/italic
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Replace bullet points
    text = re.sub(r'^[-*+•]\s+', '• ', text, flags=re.MULTILINE)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_sections_from_summary(summary: str) -> dict:
    """Extract structured sections from comprehensive transcript summary"""
    sections = {
        'transcript_summary': '',
        'company_name': '',
        'overall_sentiment': '',
        'sentiment_quote': '',
        'reasons_happy': '',
        'things_improve': '',
        'agreement_share': '',
        'issues_stories': ''
    }
    
    if not summary:
        return sections
    
    # Updated patterns to match actual format without ### and **
    section_patterns = {
        'transcript_summary': r'A\.\s*Summary of the Transcript\s*(.*?)(?=B\.\s*Name of the Company|$)',
        'company_name': r'B\.\s*Name of the Company\s*(.*?)(?=C\.\s*Overall Sentiment|$)',
        'overall_sentiment': r'C\.\s*Overall Sentiment\s*(.*?)(?=D\.\s*Reasons Which Make Them Happy|$)',
        'reasons_happy': r'D\.\s*Reasons Which Make Them Happy\s*(.*?)(?=E\.\s*Things That Can Be Improved|$)',
        'things_improve': r'E\.\s*Things That Can Be Improved\s*(.*?)(?=F\.\s*Agreement to Share Information|$)',
        'agreement_share': r'F\.\s*Agreement to Share Information\s*(.*?)(?=G\.\s*Summary of Issues and Stories Mentioned|$)',
        'issues_stories': r'G\.\s*Summary of Issues and Stories Mentioned\s*(.*?)$'
    }
    
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, summary, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            sections[section_name] = clean_for_excel(content)
    
    # Extract sentiment quote from overall sentiment section
    if sections['overall_sentiment']:
        sentiment_quote_match = re.search(r'Respondent Quote:\s*"([^"]*)"', sections['overall_sentiment'])
        if sentiment_quote_match:
            sections['sentiment_quote'] = sentiment_quote_match.group(1).strip()
    
    # Extract company name more specifically
    if sections['company_name']:
        # Look for company name after "The client company mentioned in the transcript is"
        company_match = re.search(r'The client company mentioned in the transcript is\s*([^.]*)', sections['company_name'])
        if company_match:
            sections['company_name'] = company_match.group(1).strip()
    
    return sections

@router.get("/bulk-export/transcript-summaries")
async def export_transcript_summaries(
    format: str = Query("excel", description="Export format (excel/csv)"),
    include_all: bool = Query(False, description="Include all documents or only transcripts"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Export transcript summaries in bulk to Excel or CSV format
    """
    try:
        # Get all user's documents
        query = db.query(Document).filter(Document.owner_id == current_user.id)
        
        if not include_all:
            # Filter only transcript documents
            all_docs = query.all()
            transcript_docs = []
            
            for doc in all_docs:
                is_transcript = False
                
                # Check document content for transcript indicators
                if doc.summary and detect_transcript_content(doc.summary):
                    is_transcript = True
                elif doc.insight and detect_transcript_content(doc.insight):
                    is_transcript = True
                
                if is_transcript:
                    transcript_docs.append(doc)
            
            documents = transcript_docs
        else:
            documents = query.all()
        
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found for export")
        
        # Prepare export data
        export_data = []
        
        for doc in documents:
            try:
                # Basic document info with safe attribute access
                row_data = {
                    'Document ID': getattr(doc, 'id', 'Unknown'),
                    'Document Name': getattr(doc, 'filename', 'Unknown'),
                    'Upload Date': doc.created_at.strftime('%Y-%m-%d %H:%M:%S') if getattr(doc, 'created_at', None) else '',
                    'File Size (MB)': round(getattr(doc, 'file_size', 0) / (1024 * 1024), 2) if getattr(doc, 'file_size', 0) else 0,
                    'Language': 'Unknown',  # Language not available in current model
                }
            except Exception as e:
                # Fallback if document has issues
                row_data = {
                    'Document ID': str(getattr(doc, 'id', 'Error')),
                    'Document Name': f'Error accessing document: {str(e)}',
                    'Upload Date': '',
                    'File Size (MB)': 0,
                    'Language': 'Unknown',
                }
            
            # Get related AI summaries - using proper JSON array search
            ai_summaries = db.query(AISummary).filter(
                AISummary.owner_id == current_user.id
            ).all()
            
            # Filter summaries that contain this document ID
            ai_summaries = [s for s in ai_summaries if s.document_ids and doc.id in s.document_ids]
            
            # Determine if this is a transcript based on content
            is_transcript = False
            if doc.summary and detect_transcript_content(doc.summary):
                is_transcript = True
            elif doc.insight and detect_transcript_content(doc.insight):
                is_transcript = True
            
            row_data['Is Transcript'] = 'Yes' if is_transcript else 'No'
            
            # Check document summary/insight first
            if doc.summary or doc.insight:
                summary_text = doc.summary or doc.insight or ''
                
                # Check if it's a structured transcript summary
                if 'Comprehensive Transcript Summary' in summary_text:
                    # Extract structured sections
                    sections = extract_sections_from_summary(summary_text)
                    
                    row_data.update({
                        'Summary Type': 'Comprehensive Transcript Analysis',
                        'Transcript Summary (150 words)': sections['transcript_summary'],
                        'Company Name': sections['company_name'],
                        'Overall Sentiment': clean_for_excel(sections['overall_sentiment']),
                        'Key Sentiment Quote': sections['sentiment_quote'],
                        'Reasons Happy': sections['reasons_happy'],
                        'Things to Improve': sections['things_improve'],
                        'Agreement to Share': sections['agreement_share'],
                        'Issues & Stories': sections['issues_stories'],
                        'Full Summary': clean_for_excel(summary_text)
                    })
                else:
                    # Regular summary
                    row_data.update({
                        'Summary Type': 'Regular Summary',
                        'Full Summary': clean_for_excel(summary_text),
                        'Transcript Summary (150 words)': clean_for_excel(summary_text[:500] + '...' if len(summary_text) > 500 else summary_text),
                        'Company Name': '',
                        'Overall Sentiment': '',
                        'Key Sentiment Quote': '',
                        'Reasons Happy': '',
                        'Things to Improve': '',
                        'Agreement to Share': '',
                        'Issues & Stories': ''
                    })
            
            # Check AI summaries for additional context
            elif ai_summaries:
                try:
                    # Use the most recent AI summary
                    latest_summary = max(ai_summaries, key=lambda x: x.created_at)
                    summary_type = getattr(latest_summary, 'summary_type', 'Unknown')
                    summary_content = getattr(latest_summary, 'summary_content', '')
                    
                    row_data.update({
                        'Summary Type': f'AI Summary ({summary_type})',
                        'Full Summary': clean_for_excel(summary_content) if summary_content else '',
                        'Transcript Summary (150 words)': clean_for_excel(summary_content[:500] + '...' if summary_content and len(summary_content) > 500 else summary_content or ''),
                        'Company Name': '',
                        'Overall Sentiment': '',
                        'Key Sentiment Quote': '',
                        'Reasons Happy': '',
                        'Things to Improve': '',
                        'Agreement to Share': '',
                        'Issues & Stories': ''
                    })
                except Exception as e:
                    # Fallback if AI summary has issues
                    row_data.update({
                        'Summary Type': 'AI Summary (Error)',
                        'Full Summary': f'Error accessing AI summary: {str(e)}',
                        'Transcript Summary (150 words)': '',
                        'Company Name': '',
                        'Overall Sentiment': '',
                        'Key Sentiment Quote': '',
                        'Reasons Happy': '',
                        'Things to Improve': '',
                        'Agreement to Share': '',
                        'Issues & Stories': ''
                    })
            
            else:
                # No analysis available
                row_data.update({
                    'Summary Type': 'Not Analyzed',
                    'Transcript Summary (150 words)': '',
                    'Full Summary': '',
                    'Company Name': '',
                    'Overall Sentiment': '',
                    'Key Sentiment Quote': '',
                    'Reasons Happy': '',
                    'Things to Improve': '',
                    'Agreement to Share': '',
                    'Issues & Stories': ''
                })
            
            # Add themes and keywords - using tags from Document model
            themes = []
            if hasattr(doc, 'tags') and doc.tags:
                themes = doc.tags
            row_data['Themes/Topics'] = ', '.join(themes) if themes else ''
            
            # Add demographics if available - using demo_tags from Document model
            demographics = []
            if hasattr(doc, 'demo_tags') and doc.demo_tags:
                demographics = doc.demo_tags
            row_data['Demographics'] = ', '.join(demographics) if demographics else ''
            
            # Add sentiment and reading level - not available in current model
            row_data['Document Sentiment'] = ''
            row_data['Reading Level'] = ''
            
            export_data.append(row_data)
        
        # Create DataFrame
        df = pd.DataFrame(export_data)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        doc_type = 'transcripts' if not include_all else 'all_documents'
        filename = f"sixthvault_{doc_type}_export_{timestamp}"
        
        if format.lower() == 'excel':
            # Create Excel file
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Document Summaries', index=False)
                
                # Summary sheet
                # Calculate proper metrics
                transcript_count = 0
                analyzed_count = 0
                
                for doc in documents:
                    # Check if transcript based on content only
                    if doc.summary and detect_transcript_content(doc.summary):
                        transcript_count += 1
                        analyzed_count += 1
                    elif doc.insight and detect_transcript_content(doc.insight):
                        transcript_count += 1
                        analyzed_count += 1
                    elif doc.summary or doc.insight:
                        analyzed_count += 1
                
                summary_data = {
                    'Metric': [
                        'Total Documents',
                        'Transcript Documents',
                        'Documents with Analysis',
                        'Export Date',
                        'Export Type'
                    ],
                    'Value': [
                        len(documents),
                        transcript_count,
                        analyzed_count,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Transcripts Only' if not include_all else 'All Documents'
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Export Summary', index=False)
                
                # Format sheets
                workbook = writer.book
                
                # Format main sheet
                worksheet = writer.sheets['Document Summaries']
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)  # Cap at 50 chars
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                # Format summary sheet
                summary_worksheet = writer.sheets['Export Summary']
                summary_worksheet.column_dimensions['A'].width = 25
                summary_worksheet.column_dimensions['B'].width = 30
            
            output.seek(0)
            
            headers = {
                'Content-Disposition': f'attachment; filename="{filename}.xlsx"'
            }
            
            return StreamingResponse(
                io.BytesIO(output.read()),
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                headers=headers
            )
        
        else:  # CSV format
            output = io.StringIO()
            df.to_csv(output, index=False, encoding='utf-8')
            output.seek(0)
            
            headers = {
                'Content-Disposition': f'attachment; filename="{filename}.csv"'
            }
            
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode('utf-8')),
                media_type='text/csv',
                headers=headers
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/bulk-export/available-documents")
async def get_available_documents_for_export(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Get summary of available documents for export
    """
    try:
        # Get all user's documents
        all_docs = db.query(Document).filter(Document.owner_id == current_user.id).all()
        
        transcript_count = 0
        analyzed_count = 0
        
        for doc in all_docs:
            is_transcript = False
            has_analysis = False
            
            # Check document content for transcript indicators
            if doc.summary and detect_transcript_content(doc.summary):
                is_transcript = True
                has_analysis = True
            elif doc.insight and detect_transcript_content(doc.insight):
                is_transcript = True
                has_analysis = True
            elif doc.summary or doc.insight:
                has_analysis = True
            
            # Check AI summaries for additional context - using proper JSON array search
            ai_summaries = db.query(AISummary).filter(
                AISummary.owner_id == current_user.id
            ).all()
            
            # Filter summaries that contain this document ID
            ai_summaries = [s for s in ai_summaries if s.document_ids and doc.id in s.document_ids]
            
            if ai_summaries:
                has_analysis = True
            
            if is_transcript:
                transcript_count += 1
            if has_analysis:
                analyzed_count += 1
        
        return {
            "total_documents": len(all_docs),
            "transcript_documents": transcript_count,
            "analyzed_documents": analyzed_count,
            "ready_for_export": transcript_count > 0 or analyzed_count > 0
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document summary: {str(e)}")
