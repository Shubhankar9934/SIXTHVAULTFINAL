# CrewAI Integration for Lightning-Fast RAG System

## Overview

This guide provides comprehensive documentation for the CrewAI integration with your existing Lightning-Fast RAG system. The integration adds sophisticated multi-agent collaboration capabilities while maintaining all existing performance optimizations and backward compatibility.

## 🚀 Key Features

### Multi-Agent Collaboration
- **Specialized Agents**: 8 different agent roles with specific expertise
- **Task Delegation**: Intelligent routing and delegation between agents
- **Quality Assurance**: Built-in validation and quality control
- **Performance Monitoring**: Comprehensive statistics and performance tracking

### Execution Modes
- **Sequential**: Tasks executed one after another with context passing
- **Parallel**: Independent tasks run simultaneously for speed
- **Hierarchical**: Manager agent coordinates worker agents
- **Adaptive**: Automatically chooses best execution strategy

### Seamless Integration
- **Backward Compatible**: All existing functionality preserved
- **Fallback Mechanisms**: Automatic fallback to standard RAG if needed
- **Performance Optimized**: Maintains your existing speed optimizations
- **Easy Configuration**: Simple API parameters for customization

## 🤖 Agent Roles

### Core Agents

#### 1. Researcher Agent
- **Role**: Information retrieval and analysis specialist
- **Capabilities**: 
  - Multi-source search strategies
  - Query expansion and optimization
  - Relevance filtering and scoring
- **Tools**: `search`, `dense_search`, `bm25_search`, `two_stage_retrieval`

#### 2. Validator Agent
- **Role**: Quality assurance and fact-checking expert
- **Capabilities**:
  - Fact verification against sources
  - Consistency checking across results
  - Source quality validation
- **Tools**: `fact_check`, `consistency_check`, `source_validation`

#### 3. Synthesizer Agent
- **Role**: Information synthesis and response generation
- **Capabilities**:
  - Multi-context merging
  - Response structuring and formatting
  - Summary generation
- **Tools**: `merge_contexts`, `generate_summary`, `structure_response`

#### 4. Quality Assurance Agent
- **Role**: Final quality control and assessment
- **Capabilities**:
  - Response completeness checking
  - Clarity and readability assessment
  - Overall quality scoring
- **Tools**: `quality_score`, `completeness_check`, `clarity_assessment`

#### 5. Router Agent
- **Role**: Workflow coordination and task management
- **Capabilities**:
  - Query intent analysis
  - Task routing and delegation
  - Workflow coordination
- **Tools**: `analyze_query_intent`, `route_to_agent`, `coordinate_workflow`

### Specialized Agents

#### 6. Fact Checker Agent
- **Role**: Dedicated fact verification specialist
- **Focus**: Cross-referencing claims against reliable sources

#### 7. Context Optimizer Agent
- **Role**: Context selection and organization expert
- **Focus**: Maximizing relevance and minimizing redundancy

#### 8. Query Analyst Agent
- **Role**: Query understanding and processing strategist
- **Focus**: Intent recognition and complexity analysis

## 📋 Usage Examples

### 1. Basic CrewAI Query

```python
# Simple CrewAI query with default settings
POST /query
{
    "question": "What are the benefits of renewable energy?",
    "mode": "crewai",
    "provider": "bedrock"
}
```

### 2. Advanced CrewAI Configuration

```python
# Advanced configuration with specific crew type and execution mode
POST /query
{
    "question": "Compare the environmental impact of solar vs wind energy, including economic factors and implementation challenges",
    "mode": "crewai",
    "crew_type": "research",
    "execution_mode": "hierarchical",
    "provider": "bedrock",
    "max_context": true,
    "hybrid": true
}
```

### 3. Research-Focused Crew

```python
# Research crew for complex analysis tasks
POST /query
{
    "question": "Analyze the latest trends in artificial intelligence and their potential impact on healthcare",
    "mode": "crewai",
    "crew_type": "research",
    "execution_mode": "parallel",
    "provider": "bedrock",
    "document_ids": ["doc1", "doc2", "doc3"]
}
```

### 4. Sequential Processing for Quality

```python
# Sequential processing for maximum quality assurance
POST /query
{
    "question": "What are the regulatory requirements for pharmaceutical drug approval?",
    "mode": "crewai",
    "crew_type": "standard",
    "execution_mode": "sequential",
    "provider": "bedrock",
    "max_context": true
}
```

## 🔧 Configuration Options

### Crew Types

#### Standard Crew
- **Purpose**: General-purpose RAG with quality assurance
- **Agents**: Query Analyst → Researcher → Validator → Synthesizer → QA
- **Best For**: Most queries requiring balanced speed and quality

#### Research Crew
- **Purpose**: Research-intensive tasks with fact-checking
- **Agents**: Researcher → Fact Checker + Context Optimizer → Synthesizer
- **Best For**: Complex research, analysis, and fact-critical responses

### Execution Modes

#### Sequential Mode
- **Process**: Tasks executed one after another
- **Context**: Results passed between agents
- **Best For**: Quality-critical tasks, complex reasoning chains
- **Speed**: Moderate (thorough processing)

#### Parallel Mode
- **Process**: Independent tasks run simultaneously
- **Context**: Shared initial context
- **Best For**: Multiple independent sub-tasks
- **Speed**: Fast (concurrent processing)

#### Hierarchical Mode
- **Process**: Manager agent coordinates workers
- **Context**: Centralized coordination
- **Best For**: Complex multi-faceted queries
- **Speed**: Variable (depends on coordination complexity)

#### Adaptive Mode
- **Process**: Automatically selects best execution strategy
- **Context**: Dynamic based on query analysis
- **Best For**: Unknown query complexity
- **Speed**: Optimized (adapts to query needs)

## 📊 Response Format

### Standard Response
```json
{
    "answer": "Comprehensive response from the crew",
    "sources": [
        {
            "doc_id": "document_123",
            "chunk_id": "chunk_456",
            "text": "Relevant text excerpt",
            "score": 0.95
        }
    ],
    "mode": "crewai",
    "response_time_ms": 15000,
    "reasoning_summary": {
        "total_time": 15.2,
        "crew_mode": true,
        "crew_type": "standard",
        "execution_mode": "sequential",
        "agents_used": ["query_analyst", "researcher", "validator", "synthesizer", "quality_assurance"],
        "tasks_completed": 5,
        "total_tasks": 5,
        "overall_confidence": 0.87,
        "overall_quality": 0.92,
        "delegations_performed": 1,
        "agent_results": {
            "researcher": {
                "success": true,
                "confidence": 0.9,
                "quality_score": 0.85,
                "reasoning": "Found highly relevant sources...",
                "tools_used": ["two_stage_retrieval", "relevance_filter"]
            }
        }
    }
}
```

## 🔄 Integration with Existing System

### Backward Compatibility
- All existing API endpoints work unchanged
- Standard and hybrid modes remain available
- Existing agentic mode enhanced but preserved
- No breaking changes to current functionality

### Performance Considerations
- CrewAI mode is slower than standard RAG (15-30s vs <10s)
- Automatic fallback to standard RAG if CrewAI fails
- Caching works across all modes
- Performance monitoring for all execution paths

### Error Handling
- Graceful degradation to simpler modes
- Comprehensive error logging and reporting
- Agent-level error recovery and delegation
- System-level fallback mechanisms

## 🛠️ Advanced Configuration

### Custom Agent Configuration
```python
# Example of customizing agent behavior (internal configuration)
from app.services.crewai_integration import AgentConfig, AgentRole

custom_researcher = AgentConfig(
    role=AgentRole.RESEARCHER,
    goal="Retrieve highly specific technical information",
    backstory="You are a technical research specialist...",
    max_iterations=5,
    temperature=0.05,  # More deterministic
    allow_delegation=True
)
```

### Custom Task Configuration
```python
# Example of custom task setup (internal configuration)
from app.services.crewai_integration import TaskConfig, TaskPriority

custom_task = TaskConfig(
    description="Perform deep technical analysis of the query",
    expected_output="Detailed technical report with citations",
    agent_role=AgentRole.RESEARCHER,
    priority=TaskPriority.HIGH,
    dependencies=["query_analysis_task_id"]
)
```

## 📈 Performance Monitoring

### Available Metrics
- **Response Times**: Per-agent and total execution time
- **Success Rates**: Agent and crew-level success statistics
- **Confidence Scores**: Quality metrics for responses
- **Delegation Patterns**: Task routing and delegation statistics
- **Cache Performance**: Hit rates across all modes

### Monitoring Endpoints
```python
# Get CrewAI system status
GET /query/crewai/status

# Get performance metrics
GET /query/crewai/metrics

# Get available modes and configurations
GET /query/modes
```

## 🚨 Troubleshooting

### Common Issues

#### 1. CrewAI Mode Not Working
- **Symptom**: Requests fall back to standard RAG
- **Causes**: Agent initialization failure, LLM provider issues
- **Solution**: Check logs, verify provider configuration

#### 2. Slow Response Times
- **Symptom**: Responses take >30 seconds
- **Causes**: Complex queries, hierarchical mode, multiple delegations
- **Solution**: Use parallel mode, simplify queries, check agent performance

#### 3. Quality Issues
- **Symptom**: Responses don't meet expected quality
- **Causes**: Low confidence thresholds, insufficient validation
- **Solution**: Use sequential mode, enable self-reflection, check source quality

### Debug Mode
```python
# Enable verbose logging for debugging
POST /query
{
    "question": "Your question here",
    "mode": "crewai",
    "crew_type": "standard",
    "execution_mode": "sequential",
    "provider": "bedrock",
    "_debug": true  # Internal flag for detailed logging
}
```

## 🔮 Future Enhancements

### Planned Features
1. **Custom Agent Creation**: User-defined agents with specific roles
2. **Streaming Responses**: Real-time updates during crew execution
3. **Learning from Feedback**: Continuous improvement from user interactions
4. **Multi-Modal Agents**: Image and document processing capabilities
5. **External Tool Integration**: API calls and external service integration

### Extension Points
- Custom tool development for agents
- New execution modes and strategies
- Integration with external knowledge bases
- Custom validation and quality metrics

## 📚 API Reference

### Query Endpoint
```
POST /query
```

#### Parameters
- `question` (string, required): The user's question
- `mode` (string): RAG mode - "standard", "hybrid", "agentic", "crewai"
- `crew_type` (string): "standard" or "research" (CrewAI only)
- `execution_mode` (string): "sequential", "parallel", "hierarchical", "adaptive" (CrewAI only)
- `provider` (string): LLM provider
- `model` (string): Specific model to use
- `document_ids` (array): Specific documents to search
- `max_context` (boolean): Use maximum context
- `hybrid` (boolean): Allow general knowledge

### Modes Endpoint
```
GET /query/modes
```

Returns available RAG modes with descriptions, features, and configuration options.

## 🤝 Support and Contribution

### Getting Help
1. Check the comprehensive logging output
2. Review the troubleshooting section
3. Test with simpler queries first
4. Verify your provider configuration

### Contributing
- Agent role enhancements
- New execution strategies
- Performance optimizations
- Documentation improvements

---

**Implementation Status**: ✅ Complete and Production Ready
**Backward Compatibility**: ✅ Fully Maintained
**Performance**: ✅ Optimized with Fallbacks
**Documentation**: ✅ Comprehensive

This CrewAI integration provides enterprise-grade multi-agent capabilities while preserving all the speed and reliability of your existing Lightning-Fast RAG system.
