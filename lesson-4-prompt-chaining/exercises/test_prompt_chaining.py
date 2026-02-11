#!/usr/bin/env python3
"""
Lesson 4: Prompt Chaining - Comprehensive Test Suite

This test suite validates all TODOs 1-24 for Lesson 4, ensuring students
implement proper sequential and conditional chaining with BI integration.

Test Coverage:
- TODO 1: Sequential chain step execution
- TODO 2: Context flow management  
- TODO 3: Chain quality validation
- TODO 4: Conditional branching logic
- TODO 5: Adaptive prompt selection
- TODO 6: Multi-path reasoning
- TODO 7: Complete BI report chain
- TODO 8: Advanced error recovery

Usage:
    python test_prompt_chaining.py              # Test all TODOs
    python test_prompt_chaining.py --todo 17    # Test specific TODO
    python test_prompt_chaining.py --verbose    # Detailed output
    python test_prompt_chaining.py --integration # Full integration test

Requirements:
- PROJECT_ID environment variable set
- Vertex AI API access configured
- All solution files present in lesson-4-prompt-chaining/exercises/solution/

Author: Noble Ackerson (Udacity)
Date: 2025
"""

import os
import sys
import time
import unittest
import argparse
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Add solution path for testing
solution_path = os.path.join(os.path.dirname(__file__), "solution")
sys.path.insert(0, solution_path)

# Add starter path for TODO validation
starter_path = os.path.join(os.path.dirname(__file__), "starter")
sys.path.insert(0, starter_path)

try:
    from solution.sequential_chain import (
        SequentialChain as SolutionSequentialChain,
        ChainContext, ChainStep, StepResult, ChainResult,
        ChainStepType, ValidationLevel
    )
    from solution.conditional_chain import (
        ConditionalChain as SolutionConditionalChain,
        BranchingDecision, ReasoningPath, PathResult, SynthesizedResult
    )
    from solution.bi_chain_agent import (
        BusinessIntelligenceChain as SolutionBIChain,
        BusinessScenario, BIReport, BIReportMetrics
    )
    SOLUTIONS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Solution files not found: {e}")
    SOLUTIONS_AVAILABLE = False

try:
    from starter.sequential_chain import SequentialChain as StarterSequentialChain
    from starter.conditional_chain import ConditionalChain as StarterConditionalChain  
    from starter.bi_chain_agent import BusinessIntelligenceChain as StarterBIChain
    STARTERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Starter files not found: {e}")
    STARTERS_AVAILABLE = False


class TestTODO17SequentialChainExecution(unittest.TestCase):
    """Test TODO 1: Sequential Chain Step Execution."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_id = os.getenv("PROJECT_ID", "test-project")
        self.sample_steps = [
            ChainStep(
                name="Market Analysis",
                step_type=ChainStepType.ANALYSIS,
                prompt_template="Analyze the market landscape",
                quality_threshold=0.7
            ),
            ChainStep(
                name="Strategic Recommendations",
                step_type=ChainStepType.RECOMMENDATION,
                prompt_template="Provide strategic recommendations",
                quality_threshold=0.75
            )
        ]
    
    @unittest.skipUnless(SOLUTIONS_AVAILABLE, "Solution files required")
    def test_solution_chain_execution(self):
        """Test that solution correctly implements chain execution."""
        chain = SolutionSequentialChain(self.project_id)
        
        with patch.object(chain, 'client') as mock_client:
            # Mock Vertex AI responses
            mock_response = Mock()
            mock_response.text = "Comprehensive market analysis with detailed insights and strategic recommendations."
            mock_response.usage_metadata.prompt_token_count = 150
            mock_response.usage_metadata.candidates_token_count = 200
            mock_client.models.generate_content.return_value = mock_response
            
            result = chain.execute_chain("Test business scenario", self.sample_steps)
            
            self.assertIsInstance(result, ChainResult)
            self.assertTrue(result.success)
            self.assertGreaterEqual(result.overall_quality, 0.0)
            self.assertLessEqual(result.overall_quality, 1.0)
            self.assertEqual(len(result.step_results), len(self.sample_steps))
            
            # Verify each step was executed
            for step_result in result.step_results:
                self.assertIsInstance(step_result, StepResult)
                self.assertTrue(step_result.success)
    
    @unittest.skipUnless(STARTERS_AVAILABLE, "Starter files required")
    def test_starter_todo_17_implementation(self):
        """Validate that starter TODO 1 is properly implemented."""
        try:
            chain = StarterSequentialChain(self.project_id)
            
            with patch.object(chain, 'client') as mock_client:
                mock_response = Mock()
                mock_response.text = "Test response"
                mock_response.usage_metadata.prompt_token_count = 100
                mock_response.usage_metadata.candidates_token_count = 50
                mock_client.models.generate_content.return_value = mock_response
                
                result = chain.execute_chain("Test scenario", self.sample_steps[:1])
                
                if result is not None:
                    self.assertIsInstance(result, ChainResult)
                    print("✅ TODO 1: Chain execution implemented")
                else:
                    print("❌ TODO 1: Not implemented (returns None)")
                    
        except Exception as e:
            print(f"❌ TODO 1: Implementation error - {e}")


class TestTODO18ContextFlowManagement(unittest.TestCase):
    """Test TODO 2: Context Flow Management."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_id = os.getenv("PROJECT_ID", "test-project")
        self.test_context = ChainContext(initial_input="Test business scenario")
        self.test_step = ChainStep(
            name="Test Step",
            step_type=ChainStepType.ANALYSIS,
            prompt_template="Test prompt"
        )
    
    @unittest.skipUnless(SOLUTIONS_AVAILABLE, "Solution files required")
    def test_solution_context_flow(self):
        """Test that solution correctly manages context flow."""
        chain = SolutionSequentialChain(self.project_id)
        
        # Test context prompt creation
        prompt = chain._create_step_prompt(self.test_step, self.test_context, 0)
        self.assertIsInstance(prompt, str)
        self.assertIn(self.test_step.prompt_template, prompt)
        self.assertIn("Test business scenario", prompt)
        
        # Test context updates
        step_result = StepResult(
            step_name="Test Step",
            content="Test analysis content",
            quality_score=0.8,
            execution_time=1.0,
            token_usage={"input_tokens": 100, "output_tokens": 50},
            success=True,
            key_insights=["Key insight 1", "Key insight 2"]
        )
        
        updated_context = chain._update_context_flow(self.test_context, step_result)
        
        self.assertIsInstance(updated_context, ChainContext)
        self.assertEqual(len(updated_context.step_history), 1)
        self.assertEqual(len(updated_context.accumulated_insights), 2)
        self.assertGreater(updated_context.token_usage["total_input"], 0)
    
    @unittest.skipUnless(STARTERS_AVAILABLE, "Starter files required")
    def test_starter_todo_18_implementation(self):
        """Validate that starter TODO 2 is properly implemented."""
        try:
            chain = StarterSequentialChain(self.project_id)
            
            # Test context prompt creation
            prompt = chain._create_step_prompt(self.test_step, self.test_context, 0)
            
            if prompt is not None:
                self.assertIsInstance(prompt, str)
                print("✅ TODO 2: Context flow management implemented")
            else:
                print("❌ TODO 2: Not implemented (returns None)")
                
        except Exception as e:
            print(f"❌ TODO 2: Implementation error - {e}")


class TestTODO19ChainQualityValidation(unittest.TestCase):
    """Test TODO 3: Chain Quality Validation."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_id = os.getenv("PROJECT_ID", "test-project")
        self.test_step = ChainStep(
            name="Test Step",
            step_type=ChainStepType.ANALYSIS,
            prompt_template="Test prompt",
            validation_level=ValidationLevel.STANDARD
        )
        self.test_context = ChainContext(initial_input="Test scenario")
    
    @unittest.skipUnless(SOLUTIONS_AVAILABLE, "Solution files required")
    def test_solution_quality_validation(self):
        """Test that solution correctly validates quality."""
        chain = SolutionSequentialChain(self.project_id)
        
        # Test high-quality content
        high_quality_content = """
        This comprehensive analysis provides detailed insights into market dynamics, 
        competitive positioning, and strategic opportunities. The analysis shows clear
        trends toward digital transformation and increased market consolidation.
        
        Key findings include significant growth potential in emerging markets,
        competitive advantages through technology integration, and strategic
        recommendations for market expansion.
        """
        
        quality_score = chain._validate_step_result(high_quality_content, self.test_step, self.test_context)
        
        self.assertIsInstance(quality_score, float)
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)
        self.assertGreater(quality_score, 0.5)  # Should be decent quality
        
        # Test low-quality content
        low_quality_content = "Brief response."
        
        low_quality_score = chain._validate_step_result(low_quality_content, self.test_step, self.test_context)
        
        self.assertLess(low_quality_score, quality_score)  # Should be lower quality
    
    @unittest.skipUnless(STARTERS_AVAILABLE, "Starter files required")
    def test_starter_todo_19_implementation(self):
        """Validate that starter TODO 3 is properly implemented."""
        try:
            chain = StarterSequentialChain(self.project_id)
            
            quality_score = chain._validate_step_result(
                "Test content for quality validation",
                self.test_step,
                self.test_context
            )
            
            if quality_score is not None:
                self.assertIsInstance(quality_score, (int, float))
                print("✅ TODO 3: Quality validation implemented")
            else:
                print("❌ TODO 3: Not implemented (returns None)")
                
        except Exception as e:
            print(f"❌ TODO 3: Implementation error - {e}")


class TestTODO20BranchingLogic(unittest.TestCase):
    """Test TODO 4: Conditional Branching Logic."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_id = os.getenv("PROJECT_ID", "test-project")
        self.test_context = ChainContext(initial_input="Test scenario")
        self.test_context.step_history = [{"quality_score": 0.5, "content_summary": "Brief analysis"}]
        self.test_context.quality_scores = [0.5]
        self.test_step = ChainStep("Test Step", ChainStepType.ANALYSIS, "Test prompt")
    
    @unittest.skipUnless(SOLUTIONS_AVAILABLE, "Solution files required")
    def test_solution_branching_logic(self):
        """Test that solution correctly implements branching logic."""
        chain = SolutionConditionalChain(self.project_id)
        
        decision = chain.evaluate_branching_condition(self.test_context, self.test_step)
        
        self.assertIsInstance(decision, BranchingDecision)
        self.assertIsInstance(decision.confidence, float)
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)
        self.assertIsInstance(decision.decision, str)
        self.assertIsInstance(decision.reasoning, str)
    
    @unittest.skipUnless(STARTERS_AVAILABLE, "Starter files required")
    def test_starter_todo_20_implementation(self):
        """Validate that starter TODO 4 is properly implemented."""
        try:
            chain = StarterConditionalChain(self.project_id)
            
            decision = chain.evaluate_branching_condition(self.test_context, self.test_step)
            
            if decision is not None:
                self.assertIsInstance(decision, BranchingDecision)
                print("✅ TODO 4: Branching logic implemented")
            else:
                print("❌ TODO 4: Not implemented (returns None)")
                
        except Exception as e:
            print(f"❌ TODO 4: Implementation error - {e}")


class TestTODO21AdaptivePromptSelection(unittest.TestCase):
    """Test TODO 5: Adaptive Prompt Selection."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_id = os.getenv("PROJECT_ID", "test-project")
        self.test_step = ChainStep("Test Step", ChainStepType.ANALYSIS, "Test prompt")
        self.test_context = ChainContext(initial_input="Test scenario")
    
    @unittest.skipUnless(SOLUTIONS_AVAILABLE, "Solution files required") 
    def test_solution_adaptive_selection(self):
        """Test that solution correctly implements adaptive prompt selection."""
        chain = SolutionConditionalChain(self.project_id)
        
        with patch.object(chain, 'client') as mock_client:
            mock_response = Mock()
            mock_response.text = "Adaptive response content"
            mock_response.usage_metadata.prompt_token_count = 100
            mock_response.usage_metadata.candidates_token_count = 150
            mock_client.models.generate_content.return_value = mock_response
            
            result = chain._execute_adaptive_step(self.test_step, self.test_context)
            
            self.assertIsInstance(result, StepResult)
            self.assertTrue(result.success)
    
    @unittest.skipUnless(STARTERS_AVAILABLE, "Starter files required")
    def test_starter_todo_21_implementation(self):
        """Validate that starter TODO 5 is properly implemented."""
        try:
            chain = StarterConditionalChain(self.project_id)
            
            with patch.object(chain, 'client') as mock_client:
                mock_response = Mock()
                mock_response.text = "Test response"
                mock_response.usage_metadata.prompt_token_count = 100
                mock_response.usage_metadata.candidates_token_count = 50
                mock_client.models.generate_content.return_value = mock_response
                
                result = chain._execute_adaptive_step(self.test_step, self.test_context)
                
                if result is not None:
                    self.assertIsInstance(result, StepResult)
                    print("✅ TODO 5: Adaptive prompt selection implemented")
                else:
                    print("❌ TODO 5: Not implemented (returns None)")
                    
        except Exception as e:
            print(f"❌ TODO 5: Implementation error - {e}")


class TestTODO22MultiPathReasoning(unittest.TestCase):
    """Test TODO 6: Multi-Path Reasoning."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_id = os.getenv("PROJECT_ID", "test-project")
        self.test_paths = [ReasoningPath.ANALYTICAL, ReasoningPath.CREATIVE]
        self.test_context = ChainContext(initial_input="Test scenario")
    
    @unittest.skipUnless(SOLUTIONS_AVAILABLE, "Solution files required")
    def test_solution_multipath_reasoning(self):
        """Test that solution correctly implements multi-path reasoning."""
        chain = SolutionConditionalChain(self.project_id)
        
        with patch.object(chain, 'client') as mock_client:
            mock_response = Mock()
            mock_response.text = "Multi-path analysis result"
            mock_response.usage_metadata.prompt_token_count = 120
            mock_response.usage_metadata.candidates_token_count = 180
            mock_client.models.generate_content.return_value = mock_response
            
            result = chain.execute_multi_path_reasoning(
                "Test prompt for multi-path analysis",
                self.test_paths,
                self.test_context
            )
            
            self.assertIsInstance(result, StepResult)
            self.assertTrue(result.success)
            self.assertIn("multi_path", result.step_name.lower())
    
    @unittest.skipUnless(STARTERS_AVAILABLE, "Starter files required")
    def test_starter_todo_22_implementation(self):
        """Validate that starter TODO 6 is properly implemented."""
        try:
            chain = StarterConditionalChain(self.project_id)
            
            with patch.object(chain, 'client') as mock_client:
                mock_response = Mock()
                mock_response.text = "Test response"
                mock_response.usage_metadata.prompt_token_count = 100
                mock_response.usage_metadata.candidates_token_count = 50
                mock_client.models.generate_content.return_value = mock_response
                
                result = chain.execute_multi_path_reasoning(
                    "Test prompt",
                    self.test_paths[:1],
                    self.test_context
                )
                
                if result is not None:
                    self.assertIsInstance(result, StepResult)
                    print("✅ TODO 6: Multi-path reasoning implemented")
                else:
                    print("❌ TODO 6: Not implemented (returns None)")
                    
        except Exception as e:
            print(f"❌ TODO 6: Implementation error - {e}")


class TestTODO23BIReportChain(unittest.TestCase):
    """Test TODO 7: Complete BI Report Chain."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_id = os.getenv("PROJECT_ID", "test-project")
        self.test_scenario = BusinessScenario(
            company_name="TestCorp",
            industry="Technology",
            market_focus="enterprise software",
            strategic_question="Should we expand internationally?",
            additional_context="Strong domestic presence"
        )
    
    @unittest.skipUnless(SOLUTIONS_AVAILABLE, "Solution files required")
    def test_solution_bi_report_generation(self):
        """Test that solution correctly generates BI reports."""
        bi_agent = SolutionBIChain(self.project_id)
        
        with patch.object(bi_agent, 'client') as mock_client:
            mock_response = Mock()
            mock_response.text = "Comprehensive business intelligence analysis section"
            mock_response.usage_metadata.prompt_token_count = 200
            mock_response.usage_metadata.candidates_token_count = 300
            mock_client.models.generate_content.return_value = mock_response
            
            report = bi_agent.generate_complete_report(self.test_scenario)
            
            self.assertIsInstance(report, BIReport)
            self.assertEqual(report.scenario, self.test_scenario)
            self.assertIsInstance(report.metrics, BIReportMetrics)
            
            if report.success:
                self.assertGreater(len(report.sections), 0)
                self.assertIsInstance(report.executive_summary, str)
    
    @unittest.skipUnless(STARTERS_AVAILABLE, "Starter files required")
    def test_starter_todo_23_implementation(self):
        """Validate that starter TODO 7 is properly implemented."""
        try:
            bi_agent = StarterBIChain(self.project_id)
            
            with patch.object(bi_agent, 'client') as mock_client:
                mock_response = Mock()
                mock_response.text = "Test BI section"
                mock_response.usage_metadata.prompt_token_count = 100
                mock_response.usage_metadata.candidates_token_count = 150
                mock_client.models.generate_content.return_value = mock_response
                
                report = bi_agent.generate_complete_report(self.test_scenario)
                
                if report is not None:
                    self.assertIsInstance(report, BIReport)
                    print("✅ TODO 7: BI report chain implemented")
                else:
                    print("❌ TODO 7: Not implemented (returns None)")
                    
        except Exception as e:
            print(f"❌ TODO 7: Implementation error - {e}")


class TestTODO24ErrorRecovery(unittest.TestCase):
    """Test TODO 8: Advanced Error Recovery."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_id = os.getenv("PROJECT_ID", "test-project")
        self.test_step = ChainStep("Test Step", ChainStepType.ANALYSIS, "Test prompt")
        self.test_context = ChainContext(initial_input="Test scenario")
        self.test_error = Exception("Test error for recovery")
    
    @unittest.skipUnless(SOLUTIONS_AVAILABLE, "Solution files required")
    def test_solution_error_recovery(self):
        """Test that solution correctly implements error recovery."""
        bi_agent = SolutionBIChain(self.project_id)
        
        with patch.object(bi_agent, 'client') as mock_client:
            mock_response = Mock()
            mock_response.text = "Recovery response"
            mock_response.usage_metadata.prompt_token_count = 100
            mock_response.usage_metadata.candidates_token_count = 100
            mock_client.models.generate_content.return_value = mock_response
            
            recovery_success, recovery_result = bi_agent.handle_chain_failure(
                self.test_step, 
                self.test_error, 
                self.test_context
            )
            
            self.assertIsInstance(recovery_success, bool)
            
            if recovery_success:
                self.assertIsInstance(recovery_result, StepResult)
            else:
                # Recovery failed, but method should handle gracefully
                pass
    
    @unittest.skipUnless(STARTERS_AVAILABLE, "Starter files required")
    def test_starter_todo_24_implementation(self):
        """Validate that starter TODO 8 is properly implemented."""
        try:
            bi_agent = StarterBIChain(self.project_id)
            
            recovery_result = bi_agent.handle_chain_failure(
                self.test_step,
                self.test_error,
                self.test_context
            )
            
            if recovery_result is not None:
                self.assertIsInstance(recovery_result, tuple)
                print("✅ TODO 8: Error recovery implemented")
            else:
                print("❌ TODO 8: Not implemented (returns None)")
                
        except Exception as e:
            print(f"❌ TODO 8: Implementation error - {e}")


class TestIntegrationWorkflow(unittest.TestCase):
    """Test complete workflow integration across all TODOs."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.project_id = os.getenv("PROJECT_ID", "test-project")
        self.test_scenario = BusinessScenario(
            company_name="IntegrationTest Corp",
            industry="Technology",
            market_focus="AI solutions",
            strategic_question="How should we approach market expansion?",
            additional_context="Startup with strong technology foundation"
        )
    
    @unittest.skipUnless(SOLUTIONS_AVAILABLE, "Solution files required")
    def test_complete_chaining_workflow(self):
        """Test the complete chaining workflow from sequential to BI agent."""
        
        # Test 1: Sequential Chain
        seq_chain = SolutionSequentialChain(self.project_id)
        steps = [
            ChainStep("Analysis", ChainStepType.ANALYSIS, "Analyze market", quality_threshold=0.6),
            ChainStep("Recommendations", ChainStepType.RECOMMENDATION, "Provide recommendations", quality_threshold=0.6)
        ]
        
        with patch.object(seq_chain, 'client') as mock_client:
            mock_response = Mock()
            mock_response.text = "Integration test analysis"
            mock_response.usage_metadata.prompt_token_count = 100
            mock_response.usage_metadata.candidates_token_count = 150
            mock_client.models.generate_content.return_value = mock_response
            
            seq_result = seq_chain.execute_chain("Test scenario", steps)
            self.assertTrue(seq_result.success)
        
        # Test 2: Conditional Chain 
        cond_chain = SolutionConditionalChain(self.project_id)
        
        with patch.object(cond_chain, 'client') as mock_client:
            mock_client.models.generate_content.return_value = mock_response
            
            cond_result = cond_chain.execute_conditional_chain("Test scenario", steps[:1])
            self.assertTrue(cond_result.success)
        
        # Test 3: BI Agent
        bi_agent = SolutionBIChain(self.project_id)
        
        with patch.object(bi_agent, 'client') as mock_client:
            mock_client.models.generate_content.return_value = mock_response
            
            bi_report = bi_agent.generate_complete_report(self.test_scenario)
            self.assertIsInstance(bi_report, BIReport)
        
        print("✅ Complete integration workflow validated")


def run_todo_specific_test(todo_number: int, verbose: bool = False):
    """Run tests for a specific TODO."""
    test_classes = {
        17: TestTODO17SequentialChainExecution,
        18: TestTODO18ContextFlowManagement,
        19: TestTODO19ChainQualityValidation,
        20: TestTODO20BranchingLogic,
        21: TestTODO21AdaptivePromptSelection,
        22: TestTODO22MultiPathReasoning,
        23: TestTODO23BIReportChain,
        24: TestTODO24ErrorRecovery
    }
    
    if todo_number not in test_classes:
        print(f"❌ Invalid TODO number: {todo_number}")
        return
    
    print(f"\n🧪 Testing TODO {todo_number}")
    print("=" * 50)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(test_classes[todo_number])
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print(f"✅ TODO {todo_number}: All tests passed")
    else:
        print(f"❌ TODO {todo_number}: {len(result.failures)} failures, {len(result.errors)} errors")


def run_integration_test(verbose: bool = False):
    """Run full integration test."""
    print("\n🔄 Running Integration Tests")
    print("=" * 50)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegrationWorkflow)
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("✅ Integration test passed")
    else:
        print("❌ Integration test failed")


def validate_environment():
    """Validate test environment setup."""
    print("🔍 Validating Environment")
    print("=" * 30)
    
    # Check PROJECT_ID
    project_id = os.getenv("PROJECT_ID")
    if not project_id or project_id == "your-project-id":
        print("❌ PROJECT_ID environment variable not set")
        return False
    else:
        print(f"✅ PROJECT_ID: {project_id}")
    
    # Check file availability
    solution_available = SOLUTIONS_AVAILABLE
    starter_available = STARTERS_AVAILABLE
    
    print(f"✅ Solution files: {'Available' if solution_available else 'Missing'}")
    print(f"✅ Starter files: {'Available' if starter_available else 'Missing'}")
    
    if not (solution_available or starter_available):
        print("❌ No test files available")
        return False
    
    return True


def main():
    """Main test runner with command line interface."""
    parser = argparse.ArgumentParser(description="Lesson 4: Prompt Chaining Test Suite")
    parser.add_argument("--todo", type=int, choices=[17, 18, 19, 20, 21, 22, 23, 24], 
                       help="Test specific TODO number")
    parser.add_argument("--integration", action="store_true", 
                       help="Run integration tests")
    parser.add_argument("--verbose", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--validate-env", action="store_true",
                       help="Only validate environment")
    
    args = parser.parse_args()
    
    print("🧪 Lesson 4: Prompt Chaining Test Suite")
    print("=" * 60)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    if args.validate_env:
        print("✅ Environment validation complete")
        return
    
    # Run specific tests
    if args.todo:
        run_todo_specific_test(args.todo, args.verbose)
    elif args.integration:
        run_integration_test(args.verbose)
    else:
        # Run all tests
        print("\n🧪 Running All TODO Tests")
        print("=" * 50)
        
        for todo_num in [17, 18, 19, 20, 21, 22, 23, 24]:
            run_todo_specific_test(todo_num, args.verbose)
            print()
        
        # Run integration test
        run_integration_test(args.verbose)
        
        print("\n📊 Test Summary")
        print("=" * 30)
        print("All TODO tests completed. Check output above for results.")
        print("\n🎯 Cross-Lesson Integration:")
        print("- Lesson 1: Personas integrated in BI agent")
        print("- Lesson 2: CoT/ReACT techniques in conditional chains")
        print("- Lesson 3: Prompt optimization in adaptive selection")
        print("- Lesson 4: Complete chaining workflow")


if __name__ == "__main__":
    main()