#!/usr/bin/env python3
"""
Test script to verify CrewAI integration with Agentic RAG
========================================================

This script tests the complete integration flow:
1. Standard RAG mode
2. Agentic RAG mode  
3. CrewAI multi-agent mode
4. Integration between Agentic RAG and CrewAI

Usage:
    python test_crewai_agentic_integration.py
"""

import asyncio
import sys
import os
import time
import json
from typing import Dict, Any, List, Tuple

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Rag_Backend'))

# Import the services we need to test
from app.services.rag import lightning_answer, agentic_answer, AgenticConfig
from app.services.crewai_integration import crewai_rag_answer, initialize_crewai_system, get_crewai_modes
from app.services.llm_factory import NeverFailLLM

class IntegrationTester:
    """Test class for CrewAI + Agentic RAG integration"""
    
    def __init__(self):
        self.test_results = []
        self.test_user_id = "test_tenant_integration"
        self.test_question = "What are the main benefits of artificial intelligence in healthcare?"
        
    def log_test(self, test_name: str, success: bool, details: Dict[str, Any]):
        """Log test results"""
        result = {
            "test_name": test_name,
            "success": success,
            "timestamp": time.time(),
            "details": details
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details.get("error"):
            print(f"   Error: {details['error']}")
        if details.get("execution_time"):
            print(f"   Time: {details['execution_time']:.2f}s")
        print()
    
    async def test_standard_rag(self) -> bool:
        """Test 1: Standard RAG mode"""
        print("🔄 Testing Standard RAG Mode...")
        
        try:
            start_time = time.time()
            
            answer, sources = await lightning_answer(
                user_id=self.test_user_id,
                question=self.test_question,
                hybrid=False,
                provider="bedrock",
                max_context=False,
                mode="standard"
            )
            
            execution_time = time.time() - start_time
            
            success = bool(answer and len(answer.strip()) > 10)
            
            self.log_test("Standard RAG Mode", success, {
                "execution_time": execution_time,
                "answer_length": len(answer) if answer else 0,
                "sources_count": len(sources) if sources else 0,
                "answer_preview": answer[:100] + "..." if answer and len(answer) > 100 else answer
            })
            
            return success
            
        except Exception as e:
            self.log_test("Standard RAG Mode", False, {
                "error": str(e),
                "execution_time": time.time() - start_time
            })
            return False
    
    async def test_agentic_rag(self) -> bool:
        """Test 2: Agentic RAG mode"""
        print("🔄 Testing Agentic RAG Mode...")
        
        try:
            start_time = time.time()
            
            # Configure agentic settings
            agentic_config = AgenticConfig(
                max_iterations=2,
                confidence_threshold=0.7,
                enable_query_decomposition=True,
                enable_self_reflection=True,
                reasoning_chain_visible=True
            )
            
            answer, sources, reasoning_summary = await agentic_answer(
                user_id=self.test_user_id,
                question=self.test_question,
                hybrid=False,
                provider="bedrock",
                max_context=False,
                agentic_config=agentic_config
            )
            
            execution_time = time.time() - start_time
            
            success = bool(
                answer and len(answer.strip()) > 10 and
                reasoning_summary and
                reasoning_summary.get('agentic_mode') is True
            )
            
            self.log_test("Agentic RAG Mode", success, {
                "execution_time": execution_time,
                "answer_length": len(answer) if answer else 0,
                "sources_count": len(sources) if sources else 0,
                "reasoning_steps": len(reasoning_summary.get('reasoning_chain', [])) if reasoning_summary else 0,
                "confidence_score": reasoning_summary.get('final_confidence', 0) if reasoning_summary else 0,
                "iterations": reasoning_summary.get('iterations_performed', 0) if reasoning_summary else 0,
                "answer_preview": answer[:100] + "..." if answer and len(answer) > 100 else answer
            })
            
            return success
            
        except Exception as e:
            self.log_test("Agentic RAG Mode", False, {
                "error": str(e),
                "execution_time": time.time() - start_time
            })
            return False
    
    async def test_crewai_mode(self) -> bool:
        """Test 3: CrewAI multi-agent mode"""
        print("🔄 Testing CrewAI Multi-Agent Mode...")
        
        try:
            start_time = time.time()
            
            # Initialize CrewAI system
            await initialize_crewai_system()
            
            answer, sources, reasoning_summary = await crewai_rag_answer(
                user_id=self.test_user_id,
                question=self.test_question,
                hybrid=False,
                provider="bedrock",
                max_context=False,
                execution_mode="sequential",
                crew_type="standard"
            )
            
            execution_time = time.time() - start_time
            
            success = bool(
                answer and len(answer.strip()) > 10 and
                reasoning_summary and
                reasoning_summary.get('crew_mode') is True
            )
            
            self.log_test("CrewAI Multi-Agent Mode", success, {
                "execution_time": execution_time,
                "answer_length": len(answer) if answer else 0,
                "sources_count": len(sources) if sources else 0,
                "agents_used": reasoning_summary.get('agents_used', []) if reasoning_summary else [],
                "tasks_completed": reasoning_summary.get('tasks_completed', 0) if reasoning_summary else 0,
                "execution_mode": reasoning_summary.get('execution_mode', 'unknown') if reasoning_summary else 'unknown',
                "answer_preview": answer[:100] + "..." if answer and len(answer) > 100 else answer
            })
            
            return success
            
        except Exception as e:
            self.log_test("CrewAI Multi-Agent Mode", False, {
                "error": str(e),
                "execution_time": time.time() - start_time
            })
            return False
    
    async def test_crewai_research_mode(self) -> bool:
        """Test 4: CrewAI research-focused mode"""
        print("🔄 Testing CrewAI Research Mode...")
        
        try:
            start_time = time.time()
            
            answer, sources, reasoning_summary = await crewai_rag_answer(
                user_id=self.test_user_id,
                question=self.test_question,
                hybrid=False,
                provider="bedrock",
                max_context=True,
                execution_mode="parallel",
                crew_type="research"
            )
            
            execution_time = time.time() - start_time
            
            success = bool(
                answer and len(answer.strip()) > 10 and
                reasoning_summary and
                reasoning_summary.get('crew_mode') is True and
                reasoning_summary.get('crew_type') == 'research'
            )
            
            self.log_test("CrewAI Research Mode", success, {
                "execution_time": execution_time,
                "answer_length": len(answer) if answer else 0,
                "sources_count": len(sources) if sources else 0,
                "crew_type": reasoning_summary.get('crew_type', 'unknown') if reasoning_summary else 'unknown',
                "execution_mode": reasoning_summary.get('execution_mode', 'unknown') if reasoning_summary else 'unknown',
                "answer_preview": answer[:100] + "..." if answer and len(answer) > 100 else answer
            })
            
            return success
            
        except Exception as e:
            self.log_test("CrewAI Research Mode", False, {
                "error": str(e),
                "execution_time": time.time() - start_time
            })
            return False
    
    async def test_mode_availability(self) -> bool:
        """Test 5: Check available modes"""
        print("🔄 Testing Mode Availability...")
        
        try:
            start_time = time.time()
            
            modes = get_crewai_modes()
            
            execution_time = time.time() - start_time
            
            success = bool(
                modes and
                'execution_modes' in modes and
                'crew_types' in modes and
                len(modes['execution_modes']) > 0 and
                len(modes['crew_types']) > 0
            )
            
            self.log_test("Mode Availability Check", success, {
                "execution_time": execution_time,
                "execution_modes": modes.get('execution_modes', []) if modes else [],
                "crew_types": modes.get('crew_types', []) if modes else [],
                "available_crews": modes.get('available_crews', []) if modes else [],
                "available_agents": modes.get('available_agents', []) if modes else []
            })
            
            return success
            
        except Exception as e:
            self.log_test("Mode Availability Check", False, {
                "error": str(e),
                "execution_time": time.time() - start_time
            })
            return False
    
    async def test_integration_consistency(self) -> bool:
        """Test 6: Integration consistency between modes"""
        print("🔄 Testing Integration Consistency...")
        
        try:
            start_time = time.time()
            
            # Test the same question with different modes
            test_question = "What is machine learning?"
            
            # Standard RAG
            standard_answer, standard_sources = await lightning_answer(
                user_id=self.test_user_id,
                question=test_question,
                hybrid=False,
                provider="bedrock",
                mode="standard"
            )
            
            # Agentic RAG
            agentic_answer_text, agentic_sources, agentic_reasoning = await agentic_answer(
                user_id=self.test_user_id,
                question=test_question,
                hybrid=False,
                provider="bedrock",
                agentic_config=AgenticConfig(max_iterations=1)
            )
            
            # CrewAI
            crewai_answer_text, crewai_sources, crewai_reasoning = await crewai_rag_answer(
                user_id=self.test_user_id,
                question=test_question,
                hybrid=False,
                provider="bedrock",
                execution_mode="sequential",
                crew_type="standard"
            )
            
            execution_time = time.time() - start_time
            
            # Check that all modes produced valid responses
            success = bool(
                standard_answer and len(standard_answer.strip()) > 10 and
                agentic_answer_text and len(agentic_answer_text.strip()) > 10 and
                crewai_answer_text and len(crewai_answer_text.strip()) > 10 and
                agentic_reasoning.get('agentic_mode') is True and
                crewai_reasoning.get('crew_mode') is True
            )
            
            self.log_test("Integration Consistency", success, {
                "execution_time": execution_time,
                "standard_answer_length": len(standard_answer) if standard_answer else 0,
                "agentic_answer_length": len(agentic_answer_text) if agentic_answer_text else 0,
                "crewai_answer_length": len(crewai_answer_text) if crewai_answer_text else 0,
                "agentic_mode_confirmed": agentic_reasoning.get('agentic_mode') if agentic_reasoning else False,
                "crew_mode_confirmed": crewai_reasoning.get('crew_mode') if crewai_reasoning else False,
                "all_modes_functional": success
            })
            
            return success
            
        except Exception as e:
            self.log_test("Integration Consistency", False, {
                "error": str(e),
                "execution_time": time.time() - start_time
            })
            return False
    
    async def test_error_handling(self) -> bool:
        """Test 7: Error handling and fallbacks"""
        print("🔄 Testing Error Handling...")
        
        try:
            start_time = time.time()
            
            # Test with invalid parameters
            answer, sources, reasoning = await crewai_rag_answer(
                user_id=self.test_user_id,
                question="",  # Empty question
                hybrid=False,
                provider="bedrock",
                execution_mode="invalid_mode",  # Invalid mode
                crew_type="invalid_type"  # Invalid type
            )
            
            execution_time = time.time() - start_time
            
            # Should handle errors gracefully and still return something
            success = bool(
                answer is not None and  # Should not crash
                reasoning is not None and
                reasoning.get('crew_mode') is True  # Should maintain crew mode even with errors
            )
            
            self.log_test("Error Handling", success, {
                "execution_time": execution_time,
                "graceful_degradation": success,
                "answer_provided": answer is not None,
                "reasoning_provided": reasoning is not None,
                "crew_mode_maintained": reasoning.get('crew_mode') if reasoning else False
            })
            
            return success
            
        except Exception as e:
            # Even exceptions should be handled gracefully
            self.log_test("Error Handling", False, {
                "error": str(e),
                "execution_time": time.time() - start_time,
                "note": "System should handle errors gracefully without crashing"
            })
            return False
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("🎯 CREWAI + AGENTIC RAG INTEGRATION TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print()
        
        # Print detailed results
        for result in self.test_results:
            status = "✅" if result['success'] else "❌"
            print(f"{status} {result['test_name']}")
            
            if result['details'].get('execution_time'):
                print(f"   ⏱️  Time: {result['details']['execution_time']:.2f}s")
            
            if result['success']:
                # Print key success metrics
                details = result['details']
                if 'answer_length' in details:
                    print(f"   📝 Answer: {details['answer_length']} chars")
                if 'sources_count' in details:
                    print(f"   📚 Sources: {details['sources_count']}")
                if 'reasoning_steps' in details:
                    print(f"   🧠 Reasoning: {details['reasoning_steps']} steps")
                if 'agents_used' in details:
                    print(f"   🤖 Agents: {len(details['agents_used'])}")
            else:
                # Print error info
                if 'error' in result['details']:
                    print(f"   ❌ Error: {result['details']['error']}")
            
            print()
        
        # Overall assessment
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! CrewAI + Agentic RAG integration is working correctly.")
        elif passed_tests >= total_tests * 0.8:
            print("⚠️  Most tests passed. Integration is mostly functional with minor issues.")
        else:
            print("🚨 Multiple test failures. Integration needs attention.")
        
        print("\n" + "="*60)

async def main():
    """Main test execution"""
    print("🚀 Starting CrewAI + Agentic RAG Integration Tests")
    print("="*60)
    print()
    
    tester = IntegrationTester()
    
    # Run all tests
    tests = [
        tester.test_standard_rag(),
        tester.test_agentic_rag(),
        tester.test_crewai_mode(),
        tester.test_crewai_research_mode(),
        tester.test_mode_availability(),
        tester.test_integration_consistency(),
        tester.test_error_handling()
    ]
    
    # Execute tests
    await asyncio.gather(*tests, return_exceptions=True)
    
    # Print summary
    tester.print_summary()

if __name__ == "__main__":
    # Run the integration tests
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        sys.exit(1)
