from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import Optional, List
import time
from app.services.rag import lightning_answer, agentic_answer, AgenticConfig
from app.services.crewai_integration import crewai_rag_answer, get_crewai_modes, initialize_crewai_system
from app.services.conversation_service import conversation_service
from app.deps import get_current_user, get_current_user_with_tenant, get_current_tenant_id_dependency

router = APIRouter(tags=["query"])

class QueryIn(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None  # List of specific document IDs to search, None means all documents
    provider: Optional[str] = "ollama"  # AI provider selection
    model: Optional[str] = None  # Specific model selection
    conversation_id: Optional[str] = None  # Optional conversation ID to continue existing conversation
    save_conversation: bool = True  # Whether to save this query as a conversation
    mode: Optional[str] = "standard"  # RAG mode - "standard", "hybrid", "agentic", "crewai"
    
    # Tags filtering for enhanced context
    selected_tags: Optional[List[str]] = None  # Content and demographic tags selected by user
    content_tags: Optional[List[str]] = None  # Specific content tags filter
    demographic_tags: Optional[List[str]] = None  # Specific demographic tags filter
    
    # Agentic RAG specific settings
    agentic_config: Optional[dict] = None  # Configuration for agentic mode
    
    # CrewAI specific settings
    crew_type: Optional[str] = "standard"  # "standard" or "research"
    execution_mode: Optional[str] = "sequential"  # "sequential", "parallel", "hierarchical", "adaptive"

@router.post("/query")
async def query(
    body: QueryIn,
    hybrid: bool = Query(False),
    max_context: bool = Query(False, description="Use maximum context for very large documents"),
    user = Depends(get_current_user_with_tenant),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
):
    start_time = time.time()
    
    # Use selected provider or default to Ollama
    selected_provider = body.provider or "ollama"
    
    # Determine RAG mode and execute accordingly
    rag_mode = body.mode or "standard"
    reasoning_summary = None
    
    if rag_mode == "crewai":
        # Use CrewAI multi-agent mode
        print(f"🚀 Using CrewAI mode for query: {body.question[:50]}...")
        
        # Execute CrewAI RAG
        ans, src, reasoning_summary = await crewai_rag_answer(
            user_id=tenant_id,
            question=body.question,
            hybrid=hybrid,
            provider=selected_provider,
            max_context=max_context,
            document_ids=body.document_ids,
            selected_model=body.model,
            execution_mode=body.execution_mode or "sequential",
            crew_type=body.crew_type or "standard",
            selected_tags=body.selected_tags,
            content_tags=body.content_tags,
            demographic_tags=body.demographic_tags
        )
        
    elif rag_mode == "agentic":
        # Use Agentic RAG mode
        print(f"🤖 Using Agentic RAG mode for query: {body.question[:50]}...")
        
        # Parse agentic configuration
        agentic_config = None
        if body.agentic_config:
            try:
                agentic_config = AgenticConfig(**body.agentic_config)
            except Exception as e:
                print(f"Invalid agentic config: {e}, using defaults")
                agentic_config = AgenticConfig()
        
        # Execute agentic RAG
        ans, src, reasoning_summary = await agentic_answer(
            user_id=tenant_id,
            question=body.question,
            hybrid=hybrid,
            provider=selected_provider,
            max_context=max_context,
            document_ids=body.document_ids,
            selected_model=body.model,
            agentic_config=agentic_config,
            selected_tags=body.selected_tags,
            content_tags=body.content_tags,
            demographic_tags=body.demographic_tags
        )
        
    else:
        # Use standard lightning RAG (supports both standard and hybrid modes)
        print(f"⚡ Using {rag_mode} RAG mode for query: {body.question[:50]}...")
        
        ans, src = await lightning_answer(
            tenant_id,  # Use tenant_id from dependency for tenant-based data sharing
            body.question, 
            hybrid, 
            selected_provider,  # Use user-selected provider
            max_context,
            document_ids=body.document_ids,
            selected_model=body.model,  # Pass selected model
            selected_tags=body.selected_tags,
            content_tags=body.content_tags,
            demographic_tags=body.demographic_tags
        )
    
    response_time_ms = int((time.time() - start_time) * 1000)
    
    # Auto-save conversation if enabled
    conversation_id = None
    if body.save_conversation:
        try:
            # Get user's conversation settings
            settings = conversation_service.get_conversation_settings(user.id)
            
            if settings.auto_save_conversations:
                # Create or get existing conversation
                if body.conversation_id:
                    # Continue existing conversation
                    conversation = conversation_service.get_conversation_by_id(
                        body.conversation_id, user.id, tenant_id
                    )
                    if conversation:
                        conversation_id = conversation.id
                
                if not conversation_id:
                    # Create new conversation
                    conversation = conversation_service.create_conversation(
                        user_id=user.id,
                        tenant_id=tenant_id,
                        title=None,  # Will be auto-generated from first message
                        auto_generate_title=settings.auto_generate_titles
                    )
                    conversation_id = conversation.id
                
                # Add user message
                conversation_service.add_message(
                    conversation_id=conversation_id,
                    role="user",
                    content=body.question,
                    user_id=user.id,
                    tenant_id=tenant_id,
                    provider_used=selected_provider,
                    model_used=body.model,
                    document_ids=body.document_ids,
                    context_mode="hybrid" if hybrid else "pure_rag",
                    metadata={
                        "max_context": max_context,
                        "query_type": "rag_search"
                    }
                )
                
                # Add assistant response with enhanced metadata
                assistant_metadata = {
                    "max_context": max_context,
                    "query_type": "rag_response",
                    "source_count": len(src) if src else 0,
                    "rag_mode": rag_mode,
                    "provider": selected_provider,
                    "model": body.model
                }
                
                # Add agentic-specific metadata if available
                if reasoning_summary:
                    assistant_metadata.update({
                        "agentic_mode": True,
                        "reasoning_steps": len(reasoning_summary.get('reasoning_chain', [])),
                        "confidence_score": reasoning_summary.get('final_confidence', 0.0),
                        "iterations_performed": reasoning_summary.get('iterations_performed', 1),
                        "tools_used": reasoning_summary.get('tools_used', [])
                    })
                
                conversation_service.add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=ans,
                    user_id=user.id,
                    tenant_id=tenant_id,
                    provider_used=selected_provider,
                    model_used=body.model,
                    document_ids=body.document_ids,
                    sources=src,
                    context_mode="hybrid" if hybrid else "pure_rag",
                    metadata=assistant_metadata,
                    response_time_ms=response_time_ms
                )
                
        except Exception as e:
            # Log error but don't fail the query
            print(f"Failed to save conversation for user {user.id}: {e}")
    
    # Build response
    response = {
        "answer": ans, 
        "sources": src,
        "mode": rag_mode,
        "response_time_ms": response_time_ms
    }
    
    # Include conversation_id in response if conversation was saved
    if conversation_id:
        response["conversation_id"] = conversation_id
    
    # Include reasoning summary for agentic mode
    if reasoning_summary:
        response["reasoning_summary"] = reasoning_summary
    
    return response

# Add new endpoint for getting available RAG modes
@router.get("/modes")
async def get_rag_modes():
    """Get available RAG modes and their descriptions"""
    return {
        "modes": [
            {
                "name": "standard",
                "display_name": "Standard RAG",
                "description": "Fast, efficient retrieval-augmented generation using document context only",
                "features": ["Lightning-fast processing", "Two-stage retrieval", "Context-only responses"],
                "use_cases": ["Quick factual queries", "Document-specific questions", "High-speed processing"]
            },
            {
                "name": "hybrid", 
                "display_name": "Hybrid RAG",
                "description": "Combines document context with general knowledge for comprehensive answers",
                "features": ["Document + general knowledge", "Enhanced context", "Broader coverage"],
                "use_cases": ["Complex explanations", "Cross-domain questions", "Educational content"]
            },
            {
                "name": "agentic",
                "display_name": "Agentic RAG",
                "description": "Intelligent agent-based RAG with reasoning, query decomposition, and adaptive retrieval",
                "features": [
                    "Query analysis and decomposition",
                    "Multi-step reasoning",
                    "Adaptive retrieval strategies", 
                    "Self-reflection and validation",
                    "Confidence scoring",
                    "Reasoning chain tracking"
                ],
                "use_cases": [
                    "Complex multi-part questions",
                    "Research and analysis tasks",
                    "Comparative queries",
                    "Exploratory investigations"
                ],
                "config_options": {
                    "max_iterations": {"type": "int", "default": 3, "description": "Maximum reasoning iterations"},
                    "confidence_threshold": {"type": "float", "default": 0.7, "description": "Confidence threshold for retrieval"},
                    "enable_query_decomposition": {"type": "bool", "default": True, "description": "Enable complex query breakdown"},
                    "enable_self_reflection": {"type": "bool", "default": True, "description": "Enable self-reflection and validation"},
                    "reasoning_chain_visible": {"type": "bool", "default": False, "description": "Show reasoning steps in response"}
                }
            },
            {
                "name": "crewai",
                "display_name": "CrewAI Multi-Agent RAG",
                "description": "Advanced multi-agent collaboration with specialized roles, task delegation, and coordinated workflows",
                "features": [
                    "Multi-agent collaboration",
                    "Specialized agent roles (Researcher, Validator, Synthesizer, QA)",
                    "Task delegation and coordination",
                    "Multiple execution modes (Sequential, Parallel, Hierarchical, Adaptive)",
                    "Quality assurance and validation",
                    "Performance monitoring and optimization",
                    "Fallback mechanisms to standard RAG"
                ],
                "use_cases": [
                    "Complex research projects",
                    "Multi-faceted analysis tasks",
                    "Quality-critical responses",
                    "Collaborative problem solving",
                    "Enterprise-grade RAG applications"
                ],
                "config_options": {
                    "crew_type": {
                        "type": "string", 
                        "default": "standard", 
                        "options": ["standard", "research"],
                        "description": "Type of crew configuration"
                    },
                    "execution_mode": {
                        "type": "string", 
                        "default": "sequential", 
                        "options": ["sequential", "parallel", "hierarchical", "adaptive"],
                        "description": "How agents coordinate and execute tasks"
                    }
                },
                "agent_roles": [
                    {
                        "name": "researcher",
                        "description": "Retrieves and analyzes relevant information from multiple sources"
                    },
                    {
                        "name": "validator", 
                        "description": "Validates accuracy and consistency of retrieved information"
                    },
                    {
                        "name": "synthesizer",
                        "description": "Combines information from multiple sources into coherent responses"
                    },
                    {
                        "name": "quality_assurance",
                        "description": "Ensures final response meets quality standards"
                    },
                    {
                        "name": "router",
                        "description": "Coordinates workflow and routes tasks between agents"
                    }
                ]
            }
        ],
        "default_mode": "standard",
        "recommendations": {
            "simple_factual": "standard",
            "complex_explanatory": "hybrid", 
            "multi_part_research": "agentic",
            "comparative_analysis": "agentic"
        }
    }
