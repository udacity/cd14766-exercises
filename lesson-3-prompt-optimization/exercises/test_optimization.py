#!/usr/bin/env python3
"""
Lesson 3: Prompt Optimization - Comprehensive Test Suite

This test suite validates all TODOs 1-16 for Lesson 3, ensuring students
implement proper prompt analysis and Vertex AI Optimizer integration.

Test Coverage:
- TODO 1: Prompt quality assessment implementation
- TODO 2: Baseline performance measurement
- TODO 3: Optimization opportunity detection
- TODO 4: Vertex AI Optimizer setup and integration
- TODO 5: Systematic prompt optimization workflow
- TODO 6: Optimization results analysis and comparison

Usage:
    python test_optimization.py              # Test all TODOs
    python test_optimization.py --todo 11    # Test specific TODO
    python test_optimization.py --verbose    # Detailed output
    python test_optimization.py --integration # Full integration test

Requirements:
- PROJECT_ID environment variable set
- Vertex AI API access configured
- All solution files present in lesson-3-prompt-optimization/exercises/solution/

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
    from solution.prompt_analyzer import PromptAnalyzer as SolutionAnalyzer, PromptAnalysis, PerformanceMetrics
    from solution.vertex_optimizer import VertexPromptOptimizer as SolutionOptimizer, OptimizationResult
    SOLUTIONS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Solution files not found: {e}")
    SOLUTIONS_AVAILABLE = False

try:
    from starter.prompt_analyzer import PromptAnalyzer as StarterAnalyzer
    from starter.vertex_optimizer import VertexPromptOptimizer as StarterOptimizer
    STARTERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Starter files not found: {e}")
    STARTERS_AVAILABLE = False


class TestTODO11PromptAnalysis(unittest.TestCase):
    """Test TODO 1: Prompt Quality Assessment Implementation."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_id = os.getenv("PROJECT_ID", "test-project")
        self.sample_prompts = {
            "simple": "You are helpful.",
            "business_analyst": """You are a senior business analyst with 15+ years of experience in strategic consulting and market analysis. Your expertise spans multiple industries including technology, healthcare, finance, and retail.

Your role is to provide comprehensive business analysis that combines quantitative data analysis with strategic thinking. You should:

1. Analyze market opportunities and competitive landscapes
2. Identify potential risks and mitigation strategies  
3. Provide data-driven recommendations for business growth
4. Consider financial implications and ROI for all suggestions

When analyzing business scenarios, always structure your response with clear sections: Market Overview, Competitive Analysis, Risk Assessment, and Strategic Recommendations. Support your analysis with specific reasoning and actionable insights.""",
            "unclear": "Maybe you could help with business stuff if needed, perhaps analyze some things.",
            "technical": """You are a senior software architect specializing in cloud-native applications and microservices architecture. Your expertise includes:

- Distributed systems design and implementation
- Cloud platforms (AWS, GCP, Azure) and container orchestration
- API design, security, and performance optimization
- DevOps practices and CI/CD pipeline automation

When analyzing technical problems:
1. Assess system requirements and constraints
2. Evaluate architectural trade-offs and alternatives
3. Recommend specific technologies and design patterns
4. Consider scalability, security, and maintainability

Provide detailed technical specifications with concrete implementation guidance."""
        }
    
    @unittest.skipUnless(SOLUTIONS_AVAILABLE, "Solution files required")
    def test_solution_prompt_analysis_implementation(self):
        """Test that solution correctly implements prompt analysis."""
        analyzer = SolutionAnalyzer(self.project_id)
        
        # Test with business analyst prompt
        analysis = analyzer.analyze_prompt_quality(
            self.sample_prompts["business_analyst"], 
            "persona"
        )
        
        self.assertIsInstance(analysis, PromptAnalysis)
        self.assertGreaterEqual(analysis.clarity_score, 0.0)
        self.assertLessEqual(analysis.clarity_score, 1.0)
        self.assertGreaterEqual(analysis.overall_score, 0.0)
        self.assertLessEqual(analysis.overall_score, 1.0)
        self.assertIsInstance(analysis.optimization_targets, list)
        self.assertGreater(analysis.word_count, 0)
        self.assertGreater(analysis.character_count, 0)
    
    @unittest.skipUnless(STARTERS_AVAILABLE, "Starter files required")
    def test_starter_todo_11_implementation(self):
        """Validate that starter TODO 1 is properly implemented."""
        try:
            analyzer = StarterAnalyzer(self.project_id)
            analysis = analyzer.analyze_prompt_quality(
                self.sample_prompts["business_analyst"]
            )
            
            # If TODO 1 is implemented, should return PromptAnalysis
            if analysis is not None:
                self.assertIsInstance(analysis, PromptAnalysis)
                print("✅ TODO 1: Prompt analysis implemented")
            else:
                print("❌ TODO 1: Not implemented (returns None)")
                
        except Exception as e:
            print(f"❌ TODO 1: Implementation error - {e}")
    
    def test_prompt_quality_scoring_accuracy(self):
        """Test accuracy of prompt quality scoring."""
        if not SOLUTIONS_AVAILABLE:
            self.skipTest("Solution files required")
            
        analyzer = SolutionAnalyzer(self.project_id)
        
        # High-quality prompt should score well
        good_analysis = analyzer.analyze_prompt_quality(
            self.sample_prompts["business_analyst"], "persona"
        )
        
        # Low-quality prompt should score poorly
        poor_analysis = analyzer.analyze_prompt_quality(
            self.sample_prompts["unclear"], "persona"
        )
        
        self.assertGreater(good_analysis.overall_score, poor_analysis.overall_score)
        self.assertGreater(good_analysis.clarity_score, poor_analysis.clarity_score)
        self.assertGreater(good_analysis.specificity_score, poor_analysis.specificity_score)


class TestTODO12PerformanceMeasurement(unittest.TestCase):
    """Test TODO 2: Baseline Performance Measurement."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_id = os.getenv("PROJECT_ID", "test-project")
        self.test_scenarios = [
            {
                "company_name": "TechFlow Solutions",
                "industry": "Software Technology",
                "market_focus": "enterprise workflow automation",
                "strategic_question": "Should we expand into small business markets?",
                "additional_context": "Strong enterprise presence, considering SMB expansion."
            }
        ]
    
    @unittest.skipUnless(SOLUTIONS_AVAILABLE, "Solution files required")
    def test_solution_performance_measurement(self):
        """Test that solution correctly measures baseline performance."""
        analyzer = SolutionAnalyzer(self.project_id)
        
        with patch.object(analyzer, 'client') as mock_client:
            # Mock Vertex AI response
            mock_response = Mock()
            mock_response.text = "Comprehensive business analysis with market overview and strategic recommendations."
            mock_response.usage_metadata.prompt_token_count = 500
            mock_response.usage_metadata.candidates_token_count = 150
            mock_client.models.generate_content.return_value = mock_response
            
            performance = analyzer.measure_baseline_performance(
                "You are a business analyst.",
                self.test_scenarios,
                num_runs=2
            )
            
            self.assertIsInstance(performance, PerformanceMetrics)
            self.assertIsInstance(performance.quality_metrics, dict)
            self.assertIsInstance(performance.token_usage, dict)
            self.assertGreaterEqual(performance.generation_time, 0)
            self.assertGreaterEqual(performance.consistency_score, 0)
            self.assertLessEqual(performance.consistency_score, 1)
    
    @unittest.skipUnless(STARTERS_AVAILABLE, "Starter files required")
    def test_starter_todo_12_implementation(self):
        """Validate that starter TODO 2 is properly implemented."""
        try:
            analyzer = StarterAnalyzer(self.project_id)
            
            with patch.object(analyzer, 'client') as mock_client:
                mock_response = Mock()
                mock_response.text = "Test response"
                mock_response.usage_metadata.prompt_token_count = 100
                mock_response.usage_metadata.candidates_token_count = 50
                mock_client.models.generate_content.return_value = mock_response
                
                performance = analyzer.measure_baseline_performance(
                    "Test prompt",
                    self.test_scenarios[:1],
                    num_runs=1
                )
                
                if performance is not None:
                    self.assertIsInstance(performance, PerformanceMetrics)
                    print("✅ TODO 2: Performance measurement implemented")
                else:
                    print("❌ TODO 2: Not implemented (returns None)")
                    
        except Exception as e:
            print(f"❌ TODO 2: Implementation error - {e}")


class TestTODO13OptimizationOpportunities(unittest.TestCase):
    """Test TODO 3: Optimization Opportunity Detection."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_id = os.getenv("PROJECT_ID", "test-project")
        
        # Mock analysis and performance data
        self.mock_analysis = PromptAnalysis(
            clarity_score=0.6,
            specificity_score=0.5,
            completeness_score=0.7,
            structure_score=0.8,
            overall_score=0.65,
            optimization_targets=["improve_clarity", "enhance_specificity"],
            word_count=120,
            character_count=850,
            readability_issues=["long_sentences", "ambiguous_terms"]
        )
        
        self.mock_performance = PerformanceMetrics(
            quality_metrics={"overall_quality": 0.6, "coherence": 0.7, "relevance": 0.5},
            token_usage={"avg_input_tokens": 1200, "avg_output_tokens": 180},
            generation_time=2.5,
            consistency_score=0.4,
            response_length=145,
            test_runs=3
        )
    
    @unittest.skipUnless(SOLUTIONS_AVAILABLE, "Solution files required")
    def test_solution_opportunity_detection(self):
        """Test that solution correctly detects optimization opportunities."""
        analyzer = SolutionAnalyzer(self.project_id)
        
        opportunities = analyzer.detect_optimization_opportunities(
            self.mock_analysis,
            self.mock_performance
        )
        
        self.assertIsInstance(opportunities, dict)
        self.assertIn("priority_targets", opportunities)
        self.assertIn("optimization_strategies", opportunities)
        self.assertIn("expected_improvements", opportunities)
        self.assertIn("optimization_urgency", opportunities)
        
        # With low scores, should identify high urgency
        self.assertIn(opportunities["optimization_urgency"], ["medium", "high"])
    
    @unittest.skipUnless(STARTERS_AVAILABLE, "Starter files required")
    def test_starter_todo_13_implementation(self):
        """Validate that starter TODO 3 is properly implemented."""
        try:
            analyzer = StarterAnalyzer(self.project_id)
            
            opportunities = analyzer.detect_optimization_opportunities(
                self.mock_analysis,
                self.mock_performance
            )
            
            if opportunities is not None:
                self.assertIsInstance(opportunities, dict)
                print("✅ TODO 3: Optimization opportunity detection implemented")
            else:
                print("❌ TODO 3: Not implemented (returns None)")
                
        except Exception as e:
            print(f"❌ TODO 3: Implementation error - {e}")


class TestTODO14VertexOptimizerSetup(unittest.TestCase):
    """Test TODO 4: Vertex AI Optimizer Setup and Integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_id = os.getenv("PROJECT_ID", "test-project")
        self.sample_prompt = "You are a business analyst. Analyze this market opportunity."
    
    @unittest.skipUnless(SOLUTIONS_AVAILABLE, "Solution files required")
    def test_solution_optimizer_setup(self):
        """Test that solution correctly sets up Vertex AI Optimizer."""
        optimizer = SolutionOptimizer(self.project_id)
        
        with patch.object(optimizer, 'client') as mock_client:
            # Mock optimizer response
            mock_optimization_response = Mock()
            mock_optimization_response.optimized_prompt = "You are an expert business analyst with proven expertise in market analysis."
            mock_optimization_response.guidelines_applied = ["improve_specificity", "enhance_clarity"]
            mock_optimization_response.optimization_time = 1.2
            mock_client.prompt_optimizer.optimize_prompt.return_value = mock_optimization_response
            
            result = optimizer.optimize_prompt(self.sample_prompt, "instructions")
            
            self.assertIsInstance(result, OptimizationResult)
            self.assertIsNotNone(result.optimized_prompt)
            self.assertIsInstance(result.guidelines_applied, list)
            self.assertGreaterEqual(result.optimization_time, 0)
    
    @unittest.skipUnless(STARTERS_AVAILABLE, "Starter files required") 
    def test_starter_todo_14_implementation(self):
        """Validate that starter TODO 4 is properly implemented."""
        try:
            optimizer = StarterOptimizer(self.project_id)
            
            with patch.object(optimizer, 'client') as mock_client:
                mock_response = Mock()
                mock_response.optimized_prompt = "Optimized prompt"
                mock_response.guidelines_applied = ["test_guideline"]
                mock_response.optimization_time = 1.0
                mock_client.prompt_optimizer.optimize_prompt.return_value = mock_response
                
                result = optimizer.optimize_prompt(self.sample_prompt)
                
                if result is not None:
                    print("✅ TODO 4: Vertex AI Optimizer integration implemented")
                else:
                    print("❌ TODO 4: Not implemented (returns None)")
                    
        except Exception as e:
            print(f"❌ TODO 4: Implementation error - {e}")


class TestTODO15SystematicOptimization(unittest.TestCase):
    """Test TODO 5: Systematic Prompt Optimization Workflow."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_id = os.getenv("PROJECT_ID", "test-project") 
        self.test_prompts = [
            "You are helpful.",
            "You are a business analyst with expertise in market analysis.",
            "Analyze this business scenario and provide recommendations."
        ]
    
    @unittest.skipUnless(SOLUTIONS_AVAILABLE, "Solution files required")
    def test_solution_systematic_optimization(self):
        """Test that solution implements systematic optimization workflow."""
        optimizer = SolutionOptimizer(self.project_id)
        
        with patch.object(optimizer, 'client') as mock_client:
            mock_response = Mock()
            mock_response.optimized_prompt = "Optimized business analyst prompt"
            mock_response.guidelines_applied = ["improve_specificity", "enhance_structure"]
            mock_response.optimization_time = 1.5
            mock_client.prompt_optimizer.optimize_prompt.return_value = mock_response
            
            # Test batch optimization
            results = optimizer.optimize_prompt_batch(
                self.test_prompts,
                optimization_type="instructions"
            )
            
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), len(self.test_prompts))
            
            for result in results:
                self.assertIsInstance(result, OptimizationResult)
    
    @unittest.skipUnless(STARTERS_AVAILABLE, "Starter files required")
    def test_starter_todo_15_implementation(self):
        """Validate that starter TODO 5 is properly implemented."""
        try:
            optimizer = StarterOptimizer(self.project_id)
            
            with patch.object(optimizer, 'client') as mock_client:
                mock_response = Mock()
                mock_response.optimized_prompt = "Optimized prompt"
                mock_response.guidelines_applied = ["test"]
                mock_response.optimization_time = 1.0
                mock_client.prompt_optimizer.optimize_prompt.return_value = mock_response
                
                # Test if batch optimization is implemented
                if hasattr(optimizer, 'optimize_prompt_batch'):
                    results = optimizer.optimize_prompt_batch(self.test_prompts[:2])
                    if results is not None:
                        print("✅ TODO 5: Systematic optimization workflow implemented")
                    else:
                        print("❌ TODO 5: Batch optimization returns None")
                else:
                    print("❌ TODO 5: optimize_prompt_batch method not found")
                    
        except Exception as e:
            print(f"❌ TODO 5: Implementation error - {e}")


class TestTODO16OptimizationResultsAnalysis(unittest.TestCase):
    """Test TODO 6: Optimization Results Analysis and Comparison."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_id = os.getenv("PROJECT_ID", "test-project")
        self.original_prompt = "You are a business analyst."
        self.optimized_prompt = "You are an expert business analyst with 15+ years of experience in strategic consulting and market analysis."
    
    @unittest.skipUnless(SOLUTIONS_AVAILABLE, "Solution files required")
    def test_solution_results_analysis(self):
        """Test that solution correctly analyzes optimization results."""
        optimizer = SolutionOptimizer(self.project_id)
        
        with patch.object(optimizer, 'analyzer') as mock_analyzer:
            # Mock analysis results
            mock_original_analysis = Mock()
            mock_original_analysis.overall_score = 0.6
            mock_original_analysis.clarity_score = 0.5
            mock_original_analysis.word_count = 25
            
            mock_optimized_analysis = Mock()
            mock_optimized_analysis.overall_score = 0.85
            mock_optimized_analysis.clarity_score = 0.9
            mock_optimized_analysis.word_count = 45
            
            mock_analyzer.analyze_prompt_quality.side_effect = [
                mock_original_analysis, mock_optimized_analysis
            ]
            
            comparison = optimizer.compare_optimization_results(
                self.original_prompt,
                self.optimized_prompt,
                ["improve_specificity", "enhance_clarity"]
            )
            
            self.assertIsInstance(comparison, dict)
            self.assertIn("improvement_metrics", comparison)
            self.assertIn("applied_guidelines", comparison)
            self.assertIn("recommendation", comparison)
    
    @unittest.skipUnless(STARTERS_AVAILABLE, "Starter files required")
    def test_starter_todo_16_implementation(self):
        """Validate that starter TODO 6 is properly implemented."""
        try:
            optimizer = StarterOptimizer(self.project_id)
            
            # Test if comparison method exists and works
            if hasattr(optimizer, 'compare_optimization_results'):
                comparison = optimizer.compare_optimization_results(
                    self.original_prompt,
                    self.optimized_prompt,
                    ["test_guideline"]
                )
                
                if comparison is not None:
                    self.assertIsInstance(comparison, dict)
                    print("✅ TODO 6: Optimization results analysis implemented")
                else:
                    print("❌ TODO 6: compare_optimization_results returns None")
            else:
                print("❌ TODO 6: compare_optimization_results method not found")
                
        except Exception as e:
            print(f"❌ TODO 6: Implementation error - {e}")


class TestIntegrationWorkflow(unittest.TestCase):
    """Test complete workflow integration across all TODOs."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.project_id = os.getenv("PROJECT_ID", "test-project")
        self.test_prompt = """You are a business analyst. Analyze market opportunities and provide recommendations."""
        self.test_scenarios = [
            {
                "company_name": "TestCorp",
                "industry": "Technology",
                "market_focus": "enterprise software",
                "strategic_question": "Should we expand internationally?",
                "additional_context": "Strong domestic presence, considering global expansion."
            }
        ]
    
    @unittest.skipUnless(SOLUTIONS_AVAILABLE, "Solution files required")
    def test_complete_optimization_workflow(self):
        """Test the complete optimization workflow from analysis to comparison."""
        # Step 1: Analyze original prompt
        analyzer = SolutionAnalyzer(self.project_id)
        
        with patch.object(analyzer, 'client') as mock_analyzer_client:
            mock_response = Mock()
            mock_response.text = "Comprehensive business analysis"
            mock_response.usage_metadata.prompt_token_count = 100
            mock_response.usage_metadata.candidates_token_count = 50
            mock_analyzer_client.models.generate_content.return_value = mock_response
            
            analysis = analyzer.analyze_prompt_quality(self.test_prompt, "task")
            performance = analyzer.measure_baseline_performance(
                self.test_prompt, 
                self.test_scenarios,
                num_runs=1
            )
            opportunities = analyzer.detect_optimization_opportunities(analysis, performance)
        
        # Step 2: Optimize prompt
        optimizer = SolutionOptimizer(self.project_id)
        
        with patch.object(optimizer, 'client') as mock_optimizer_client:
            mock_opt_response = Mock()
            mock_opt_response.optimized_prompt = "You are an expert business analyst with proven expertise in market analysis and strategic planning."
            mock_opt_response.guidelines_applied = ["improve_specificity", "enhance_clarity"]
            mock_opt_response.optimization_time = 1.2
            mock_optimizer_client.prompt_optimizer.optimize_prompt.return_value = mock_opt_response
            
            optimization_result = optimizer.optimize_prompt(self.test_prompt, "instructions")
        
        # Step 3: Compare results
        with patch.object(optimizer, 'analyzer') as mock_comparison_analyzer:
            mock_original = Mock()
            mock_original.overall_score = 0.6
            mock_optimized = Mock() 
            mock_optimized.overall_score = 0.85
            mock_comparison_analyzer.analyze_prompt_quality.side_effect = [mock_original, mock_optimized]
            
            comparison = optimizer.compare_optimization_results(
                self.test_prompt,
                optimization_result.optimized_prompt,
                optimization_result.guidelines_applied
            )
        
        # Verify complete workflow
        self.assertIsNotNone(analysis)
        self.assertIsNotNone(performance)
        self.assertIsNotNone(opportunities)
        self.assertIsNotNone(optimization_result)
        self.assertIsNotNone(comparison)
        
        print("✅ Complete optimization workflow validated")


def run_todo_specific_test(todo_number: int, verbose: bool = False):
    """Run tests for a specific TODO."""
    test_classes = {
        11: TestTODO11PromptAnalysis,
        12: TestTODO12PerformanceMeasurement,
        13: TestTODO13OptimizationOpportunities,
        14: TestTODO14VertexOptimizerSetup,
        15: TestTODO15SystematicOptimization,
        16: TestTODO16OptimizationResultsAnalysis
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
    parser = argparse.ArgumentParser(description="Lesson 3: Prompt Optimization Test Suite")
    parser.add_argument("--todo", type=int, choices=[11, 12, 13, 14, 15, 16], 
                       help="Test specific TODO number")
    parser.add_argument("--integration", action="store_true", 
                       help="Run integration tests")
    parser.add_argument("--verbose", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--validate-env", action="store_true",
                       help="Only validate environment")
    
    args = parser.parse_args()
    
    print("🧪 Lesson 3: Prompt Optimization Test Suite")
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
        
        for todo_num in [11, 12, 13, 14, 15, 16]:
            run_todo_specific_test(todo_num, args.verbose)
            print()
        
        # Run integration test
        run_integration_test(args.verbose)
        
        print("\n📊 Test Summary")
        print("=" * 30)
        print("All TODO tests completed. Check output above for results.")


if __name__ == "__main__":
    main()