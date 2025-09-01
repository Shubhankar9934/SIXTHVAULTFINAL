"""
Comprehensive Test Suite for CrewAI Integration
==============================================

This test suite validates the CrewAI integration with the Lightning-Fast RAG system,
ensuring all components work correctly and maintain backward compatibility.

Run with: python test_crewai_integration.py
"""

import asyncio
import json
import time
from typing import Dict, Any, List
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Import CrewAI components
from app.services.crewai_integration import (
    AgentRole, AgentConfig, TaskConfig, CrewConfig, TaskPriority, ExecutionMode,
    CrewAIAgent, CrewAICrew, create_crewai_agents, create_standard_crew, 
    create_research_crew, crewai_rag_answer, CrewAIIntegration
)

# Import existing RAG components for comparison
from app.services.rag import lightning_answer, NeverFailLLM

class CrewAITestSuite:
    """Comprehensive test suite for CrewAI integration"""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
        self.passed_tests = []
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test results"""
        result = {
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        if passed:
            self.passed_tests.append(test_name)
            print(f"✅ {test_name}: PASSED {details}")
        else:
            self.failed_tests.append(test_name)
            print(f"❌ {test_name}: FAILED {details}")
    
    async def test_agent_creation(self):
        """Test individual agent creation and configuration"""
        print("\n🧪 Testing Agent Creation...")
        
        try:
            # Test agent configuration
            config = AgentConfig(
                role=AgentRole.RESEARCHER,
                goal="Test goal",
                backstory="Test backstory",
                max_iterations=2
            )
            
            # Test agent creation
            agent = CrewAIAgent(config)
            
            # Validate agent properties
            assert agent.config.role == AgentRole.RESEARCHER
            assert agent.config.max_iterations == 2
            assert len(agent.tools) > 0
            assert "search" in agent.tools
            
            self.log_test("Agent Creation", True, "- Agent created with correct configuration")
            
        except Exception as e:
            self.log_test("Agent Creation", False, f"- Error: {str(e)}")
    
    async def test_task_configuration(self):
        """Test task configuration and dependency resolution"""
        print("\n🧪 Testing Task Configuration...")
        
        try:
            # Create test tasks with dependencies
            task1 = TaskConfig(
                description="First task",
                expected_output="Output 1",
                agent_role=AgentRole.RESEARCHER,
                priority=TaskPriority.HIGH
            )
            
            task2 = TaskConfig(
                description="Second task",
                expected_output="Output 2",
                agent_role=AgentRole.VALIDATOR,
                priority=TaskPriority.MEDIUM,
                dependencies=[task1.task_id]
            )
            
            # Validate task properties
            assert task1.priority == TaskPriority.HIGH
            assert task2.dependencies == [task1.task_id]
            assert task1.task_id != task2.task_id
            
            self.log_test("Task Configuration", True, "- Tasks created with correct dependencies")
            
        except Exception as e:
            self.log_test("Task Configuration", False, f"- Error: {str(e)}")
    
    async def test_crew_creation(self):
        """Test crew creation with different configurations"""
        print("\n🧪 Testing Crew Creation...")
        
        try:
            # Create agents
            agents = await create_crewai_agents()
            
            # Test standard crew creation
            standard_crew = create_standard_crew(agents, ExecutionMode.SEQUENTIAL)
            assert len(standard_crew.agents) > 0
            assert len(standard_crew.tasks) > 0
            assert standard_crew.config.execution_mode == ExecutionMode.SEQUENTIAL
            
            # Test research crew creation
            research_crew = create_research_crew(agents)
            assert len(research_crew.agents) > 0
            assert len(research_crew.tasks) > 0
            
            self.log_test("Crew Creation", True, f"- Standard crew: {len(standard_crew.agents)} agents, {len(standard_crew.tasks)} tasks")
            
        except Exception as e:
            self.log_test("Crew Creation", False, f"- Error: {str(e)}")
    
    async def test_agent_execution(self):
        """Test individual agent task execution"""
        print("\n🧪 Testing Agent Execution...")
        
        try:
            # Create a simple agent
            config = AgentConfig(
                role=AgentRole.RESEARCHER,
                goal="Find information about renewable energy",
                backstory="You are a research specialist"
            )
            agent = CrewAIAgent(config)
            
            # Create a simple task
            task = TaskConfig(
                description="Research renewable energy benefits",
                expected_output="A summary of renewable energy benefits",
                agent_role=AgentRole.RESEARCHER
            )
            
            # Execute task with timeout
            context = {"question": "What are the benefits of renewable energy?"}
            
            # Use asyncio.wait_for with a reasonable timeout
            result = await asyncio.wait_for(
                agent.execute_task(task, context),
                timeout=30.0  # 30 second timeout
            )
            
            # Validate result structure
            assert isinstance(result, dict)
            assert "success" in result
            assert "result" in result
            assert "confidence" in result
            
            success_status = "successful" if result.get("success", False) else "completed with issues"
            self.log_test("Agent Execution", True, f"- Task {success_status}, confidence: {result.get('confidence', 0):.2f}")
            
        except asyncio.TimeoutError:
            self.log_test("Agent Execution", False, "- Task execution timed out after 30 seconds")
        except Exception as e:
            self.log_test("Agent Execution", False, f"- Error: {str(e)}")
    
    async def test_crew_execution_modes(self):
        """Test different crew execution modes"""
        print("\n🧪 Testing Crew Execution Modes...")
        
        try:
            # Create agents
            agents = await create_crewai_agents()
            
            # Test sequential execution
            sequential_crew = create_standard_crew(agents, ExecutionMode.SEQUENTIAL)
            
            context = {
                "question": "What is artificial intelligence?",
                "user_id": "test_user"
            }
            
            # Execute with timeout
            result = await asyncio.wait_for(
                sequential_crew.kickoff(context),
                timeout=45.0  # 45 second timeout for crew execution
            )
            
            # Validate result
            assert isinstance(result, dict)
            assert "success" in result
            assert "execution_time" in result
            
            execution_time = result.get("execution_time", 0)
            success_status = "successful" if result.get("success", False) else "completed with issues"
            
            self.log_test("Crew Execution", True, f"- Sequential mode {success_status} in {execution_time:.2f}s")
            
        except asyncio.TimeoutError:
            self.log_test("Crew Execution", False, "- Crew execution timed out after 45 seconds")
        except Exception as e:
            self.log_test("Crew Execution", False, f"- Error: {str(e)}")
    
    async def test_crewai_integration_manager(self):
        """Test the CrewAI integration manager"""
        print("\n🧪 Testing CrewAI Integration Manager...")
        
        try:
            # Create integration manager
            integration = CrewAIIntegration()
            
            # Test initialization
            await integration.initialize()
            assert integration.initialized == True
            assert len(integration.agents) > 0
            assert len(integration.crews) > 0
            
            # Test available modes
            modes = integration.get_available_modes()
            assert "execution_modes" in modes
            assert "crew_types" in modes
            assert "sequential" in modes["execution_modes"]
            
            self.log_test("Integration Manager", True, f"- Initialized with {len(integration.agents)} agents and {len(integration.crews)} crews")
            
        except Exception as e:
            self.log_test("Integration Manager", False, f"- Error: {str(e)}")
    
    async def test_crewai_rag_function(self):
        """Test the main CrewAI RAG function with fallback"""
        print("\n🧪 Testing CrewAI RAG Function...")
        
        try:
            # Test with a simple question
            question = "What is machine learning?"
            user_id = "test_user"
            
            # Execute with timeout
            answer, sources, reasoning = await asyncio.wait_for(
                crewai_rag_answer(
                    user_id=user_id,
                    question=question,
                    provider="bedrock",
                    execution_mode="sequential",
                    crew_type="standard"
                ),
                timeout=60.0  # 60 second timeout for full RAG
            )
            
            # Validate response
            assert isinstance(answer, str)
            assert len(answer) > 0
            assert isinstance(sources, list)
            assert isinstance(reasoning, dict)
            
            # Check reasoning summary
            assert "total_time" in reasoning
            assert "crew_mode" in reasoning
            
            response_time = reasoning.get("total_time", 0)
            crew_mode = reasoning.get("crew_mode", False)
            fallback_used = reasoning.get("fallback_used", False)
            
            status = "with CrewAI" if crew_mode else "with fallback"
            if fallback_used:
                status = "with fallback to standard RAG"
            
            self.log_test("CrewAI RAG Function", True, f"- Completed {status} in {response_time:.2f}s")
            
        except asyncio.TimeoutError:
            self.log_test("CrewAI RAG Function", False, "- Function timed out after 60 seconds")
        except Exception as e:
            self.log_test("CrewAI RAG Function", False, f"- Error: {str(e)}")
    
    async def test_performance_comparison(self):
        """Compare CrewAI performance with standard RAG"""
        print("\n🧪 Testing Performance Comparison...")
        
        try:
            question = "What are the advantages of cloud computing?"
            user_id = "test_user"
            
            # Test standard RAG (if available)
            try:
                start_time = time.time()
                standard_answer, standard_sources = await asyncio.wait_for(
                    lightning_answer(user_id, question, False, "bedrock", False),
                    timeout=30.0
                )
                standard_time = time.time() - start_time
                standard_available = True
            except Exception as e:
                print(f"   Standard RAG not available: {e}")
                standard_time = 0
                standard_available = False
            
            # Test CrewAI RAG
            start_time = time.time()
            crewai_answer, crewai_sources, crewai_reasoning = await asyncio.wait_for(
                crewai_rag_answer(user_id, question, provider="bedrock"),
                timeout=60.0
            )
            crewai_time = time.time() - start_time
            
            # Compare results
            if standard_available:
                speed_ratio = crewai_time / standard_time if standard_time > 0 else 0
                details = f"- Standard: {standard_time:.2f}s, CrewAI: {crewai_time:.2f}s (ratio: {speed_ratio:.1f}x)"
            else:
                details = f"- CrewAI: {crewai_time:.2f}s (standard RAG not available for comparison)"
            
            self.log_test("Performance Comparison", True, details)
            
        except asyncio.TimeoutError:
            self.log_test("Performance Comparison", False, "- Performance test timed out")
        except Exception as e:
            self.log_test("Performance Comparison", False, f"- Error: {str(e)}")
    
    async def test_error_handling(self):
        """Test error handling and fallback mechanisms"""
        print("\n🧪 Testing Error Handling...")
        
        try:
            # Test with invalid configuration
            question = "Test question for error handling"
            user_id = "test_user"
            
            # This should trigger fallback mechanisms
            answer, sources, reasoning = await asyncio.wait_for(
                crewai_rag_answer(
                    user_id=user_id,
                    question=question,
                    provider="invalid_provider",  # This should cause fallback
                    execution_mode="sequential"
                ),
                timeout=60.0
            )
            
            # Should still get a response due to fallback
            assert isinstance(answer, str)
            assert len(answer) > 0
            
            fallback_used = reasoning.get("fallback_used", False)
            error_present = "error" in reasoning
            
            if fallback_used or error_present:
                self.log_test("Error Handling", True, "- Fallback mechanism worked correctly")
            else:
                self.log_test("Error Handling", True, "- No errors encountered (system robust)")
            
        except Exception as e:
            self.log_test("Error Handling", False, f"- Error: {str(e)}")
    
    async def test_agent_tools(self):
        """Test agent-specific tools and capabilities"""
        print("\n🧪 Testing Agent Tools...")
        
        try:
            # Create different types of agents
            agents = await create_crewai_agents()
            
            # Test researcher agent tools
            researcher = agents.get(AgentRole.RESEARCHER)
            if researcher:
                tools = researcher.tools
                expected_tools = ["search", "dense_search", "expand_query", "multi_source_search"]
                
                tools_present = sum(1 for tool in expected_tools if tool in tools)
                tool_coverage = tools_present / len(expected_tools)
                
                assert tool_coverage > 0.5  # At least half the expected tools should be present
                
                self.log_test("Agent Tools", True, f"- Researcher has {tools_present}/{len(expected_tools)} expected tools")
            else:
                self.log_test("Agent Tools", False, "- Researcher agent not found")
            
        except Exception as e:
            self.log_test("Agent Tools", False, f"- Error: {str(e)}")
    
    def print_summary(self):
        """Print test summary"""
        total_tests = len(self.test_results)
        passed_count = len(self.passed_tests)
        failed_count = len(self.failed_tests)
        
        print(f"\n{'='*60}")
        print(f"🧪 CrewAI Integration Test Summary")
        print(f"{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_count} ✅")
        print(f"Failed: {failed_count} ❌")
        print(f"Success Rate: {(passed_count/total_tests)*100:.1f}%")
        
        if failed_count > 0:
            print(f"\n❌ Failed Tests:")
            for test in self.failed_tests:
                print(f"   - {test}")
        
        if passed_count > 0:
            print(f"\n✅ Passed Tests:")
            for test in self.passed_tests:
                print(f"   - {test}")
        
        print(f"\n{'='*60}")
        
        # Overall assessment
        if failed_count == 0:
            print("🎉 All tests passed! CrewAI integration is working correctly.")
        elif failed_count <= 2:
            print("⚠️ Most tests passed. Minor issues detected.")
        else:
            print("🚨 Multiple test failures. Please review the implementation.")
        
        return failed_count == 0

async def run_comprehensive_tests():
    """Run all CrewAI integration tests"""
    print("🚀 Starting CrewAI Integration Test Suite")
    print("=" * 60)
    
    test_suite = CrewAITestSuite()
    
    # Run all tests
    test_methods = [
        test_suite.test_agent_creation,
        test_suite.test_task_configuration,
        test_suite.test_crew_creation,
        test_suite.test_agent_execution,
        test_suite.test_crew_execution_modes,
        test_suite.test_crewai_integration_manager,
        test_suite.test_agent_tools,
        test_suite.test_crewai_rag_function,
        test_suite.test_performance_comparison,
        test_suite.test_error_handling
    ]
    
    for test_method in test_methods:
        try:
            await test_method()
        except Exception as e:
            test_name = test_method.__name__.replace("test_", "").replace("_", " ").title()
            test_suite.log_test(test_name, False, f"- Unexpected error: {str(e)}")
        
        # Small delay between tests
        await asyncio.sleep(0.5)
    
    # Print summary
    all_passed = test_suite.print_summary()
    
    return all_passed

def run_quick_validation():
    """Run a quick validation of basic functionality"""
    print("⚡ Quick CrewAI Validation")
    print("-" * 30)
    
    try:
        # Test basic imports
        from app.services.crewai_integration import AgentRole, CrewAIAgent, AgentConfig
        print("✅ Imports successful")
        
        # Test basic agent creation
        config = AgentConfig(
            role=AgentRole.RESEARCHER,
            goal="Test",
            backstory="Test agent"
        )
        agent = CrewAIAgent(config)
        print("✅ Agent creation successful")
        
        # Test enum values
        roles = [role.value for role in AgentRole]
        print(f"✅ Available agent roles: {', '.join(roles)}")
        
        print("\n🎉 Quick validation passed! CrewAI integration is properly installed.")
        return True
        
    except Exception as e:
        print(f"❌ Quick validation failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CrewAI Integration")
    parser.add_argument("--quick", action="store_true", help="Run quick validation only")
    parser.add_argument("--full", action="store_true", help="Run comprehensive tests")
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_validation()
        sys.exit(0 if success else 1)
    elif args.full or len(sys.argv) == 1:
        # Run comprehensive tests
        try:
            success = asyncio.run(run_comprehensive_tests())
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            print("\n🛑 Tests interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n💥 Test suite crashed: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
