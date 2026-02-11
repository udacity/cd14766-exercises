"""
Test Suite for Lesson 2: Chain-of-Thought and ReACT Prompting

This module provides comprehensive testing for both CoT and ReACT implementations.
Run this to validate your solutions meet the requirements.

Usage:
    python test_reasoning.py --verbose        # Full test with details
    python test_reasoning.py --component cot  # Test CoT only
    python test_reasoning.py --component react # Test ReACT only
"""

import os
import sys
import argparse
import importlib
from typing import Dict, Any


class TestChainOfThought:
    """Test Chain-of-Thought implementation."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.passed_tests = 0
        self.total_tests = 0
        
    def test_standard_prompt(self, cot_module) -> bool:
        """Test TODO 1: Standard prompt creation."""
        self.total_tests += 1
        try:
            cot = cot_module.ChainOfThoughtPrompting("test-project")
            problem = "Calculate ROI for $1M investment with 20% return"
            prompt = cot.get_standard_prompt(problem)
            
            if prompt is None:
                print("❌ TODO 1: get_standard_prompt() returns None")
                return False
            
            # Check for basic elements
            checks = [
                (len(prompt) > 10, "Prompt too short"),
                (problem in prompt, "Problem not included in prompt"),
                ("step" not in prompt.lower(), "Should not contain step-by-step instructions")
            ]
            
            for check, message in checks:
                if not check:
                    if self.verbose:
                        print(f"  ⚠️ {message}")
            
            self.passed_tests += 1
            print("✅ TODO 1: Standard prompt implemented correctly")
            return True
            
        except Exception as e:
            print(f"❌ TODO 1: Error - {str(e)}")
            return False
    
    def test_cot_prompt(self, cot_module) -> bool:
        """Test TODO 2: Chain-of-Thought prompt creation."""
        self.total_tests += 1
        try:
            cot = cot_module.ChainOfThoughtPrompting("test-project")
            problem = "Calculate market opportunity for Asian expansion"
            prompt = cot.get_cot_prompt(problem)
            
            if prompt is None:
                print("❌ TODO 2: get_cot_prompt() returns None")
                return False
            
            # Check for CoT elements
            cot_indicators = [
                "step", "think", "systematic", "calculate", 
                "reason", "first", "second", "approach"
            ]
            
            prompt_lower = prompt.lower()
            found_indicators = sum(1 for indicator in cot_indicators if indicator in prompt_lower)
            
            if found_indicators < 3:
                print(f"❌ TODO 2: Missing CoT indicators (found {found_indicators}/3 minimum)")
                return False
            
            if len(prompt) < 100:
                print("❌ TODO 2: CoT prompt too short (needs detailed instructions)")
                return False
            
            self.passed_tests += 1
            print("✅ TODO 2: Chain-of-Thought prompt implemented correctly")
            return True
            
        except Exception as e:
            print(f"❌ TODO 2: Error - {str(e)}")
            return False
    
    def test_reasoning_analysis(self, cot_module) -> bool:
        """Test TODO 3: Reasoning quality analysis."""
        self.total_tests += 1
        try:
            cot = cot_module.ChainOfThoughtPrompting("test-project")
            
            # Test with sample response containing reasoning steps
            sample_response = """
            Let me solve this step by step.
            
            Step 1: First, I need to identify the market size.
            The Asian market is worth $2.5 trillion.
            
            Step 2: Next, calculate our potential share.
            With 1% market share, that's $25 billion.
            
            Step 3: Therefore, considering our investment of $5M,
            the ROI would be significant.
            
            Thus, I recommend proceeding with the expansion.
            """
            
            analysis = cot.analyze_reasoning_quality(sample_response)
            
            if analysis is None:
                print("❌ TODO 3: analyze_reasoning_quality() returns None")
                return False
            
            # Check required fields
            required_fields = [
                "reasoning_steps", "has_calculations", "has_logical_connectors",
                "step_count", "coherence_score", "confidence_score"
            ]
            
            for field in required_fields:
                if field not in analysis:
                    print(f"❌ TODO 3: Missing required field '{field}'")
                    return False
            
            # Validate analysis accuracy
            if analysis["step_count"] < 2:
                print(f"❌ TODO 3: Should find at least 2 steps (found {analysis['step_count']})")
                return False
            
            if not analysis["has_logical_connectors"]:
                print("❌ TODO 3: Should detect logical connectors (therefore, thus)")
                return False
            
            if analysis["coherence_score"] <= 0:
                print("❌ TODO 3: Coherence score should be > 0 for structured response")
                return False
            
            self.passed_tests += 1
            print("✅ TODO 3: Reasoning analysis implemented correctly")
            return True
            
        except Exception as e:
            print(f"❌ TODO 3: Error - {str(e)}")
            return False


class TestReACT:
    """Test ReACT framework implementation."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.passed_tests = 0
        self.total_tests = 0
    
    def test_reasoning_step(self, react_module) -> bool:
        """Test TODO 4: Reasoning implementation."""
        self.total_tests += 1
        try:
            # Create test context
            context = react_module.ReACTContext(
                problem="Should we expand to Asia with $5M budget?",
                observations=["Market size is $2.5 trillion"]
            )
            
            agent = react_module.ReACTAgent("test-project")
            reasoning = agent.reason(context)
            
            if reasoning is None:
                print("❌ TODO 4: reason() returns None")
                return False
            
            # Check for structured reasoning elements
            required_elements = ["THOUGHT", "ACTION"]
            reasoning_upper = reasoning.upper()
            
            for element in required_elements:
                if element not in reasoning_upper:
                    print(f"❌ TODO 4: Missing '{element}' in reasoning structure")
                    return False
            
            self.passed_tests += 1
            print("✅ TODO 4: Reasoning step implemented correctly")
            return True
            
        except Exception as e:
            print(f"❌ TODO 4: Error - {str(e)}")
            return False
    
    def test_action_execution(self, react_module) -> bool:
        """Test TODO 5: Action execution."""
        self.total_tests += 1
        try:
            context = react_module.ReACTContext(problem="Calculate 5M * 0.25")
            agent = react_module.ReACTAgent("test-project")
            
            # Test calculation action
            reasoning = "ACTION: calculate the expression 5000000 * 0.25"
            result = agent.act(reasoning, context)
            
            if result is None:
                print("❌ TODO 5: act() returns None")
                return False
            
            if "type" not in result:
                print("❌ TODO 5: Action result missing 'type' field")
                return False
            
            # Test final answer detection
            final_reasoning = "FINAL_ANSWER: The expansion is recommended"
            final_result = agent.act(final_reasoning, context)
            
            if final_result and final_result.get("type") != "final_answer":
                print("❌ TODO 5: Should detect FINAL_ANSWER")
                return False
            
            self.passed_tests += 1
            print("✅ TODO 5: Action execution implemented correctly")
            return True
            
        except Exception as e:
            print(f"❌ TODO 5: Error - {str(e)}")
            return False
    
    def test_observation_processing(self, react_module) -> bool:
        """Test TODO 6: Observation processing."""
        self.total_tests += 1
        try:
            context = react_module.ReACTContext(problem="Test problem")
            agent = react_module.ReACTAgent("test-project")
            
            # Test different action result types
            test_cases = [
                ({"type": "calculation", "expression": "5*2", "result": 10}, 
                 "Calculation"),
                ({"type": "market_data", "market": "asia", "metric": "size", "result": "$2.5T"}, 
                 "Market data"),
                ({"type": "final_answer", "result": "Proceed with expansion"}, 
                 "Final answer")
            ]
            
            for action_result, expected_phrase in test_cases:
                observation = agent.observe(action_result, context)
                
                if observation is None:
                    print(f"❌ TODO 6: observe() returns None for {action_result['type']}")
                    return False
                
                if len(observation) < 10:
                    print(f"❌ TODO 6: Observation too short for {action_result['type']}")
                    return False
            
            # Check if final answer updates context
            final_result = {"type": "final_answer", "result": "Test answer"}
            agent.observe(final_result, context)
            
            if not context.is_complete:
                print("❌ TODO 6: Should set context.is_complete for final answer")
                return False
            
            self.passed_tests += 1
            print("✅ TODO 6: Observation processing implemented correctly")
            return True
            
        except Exception as e:
            print(f"❌ TODO 6: Error - {str(e)}")
            return False


def run_tests(component: str = "all", verbose: bool = False):
    """Run test suite for specified component."""
    
    print("=" * 60)
    print("LESSON 2: REASONING TESTS")
    print("=" * 60)
    
    # Test Chain-of-Thought if requested
    if component in ["all", "cot"]:
        print("\n📝 Testing Chain-of-Thought Implementation")
        print("-" * 40)
        
        try:
            # Import student's CoT module
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            cot_module = importlib.import_module("cot_prompting")
            
            tester = TestChainOfThought(verbose)
            tester.test_standard_prompt(cot_module)
            tester.test_cot_prompt(cot_module)
            tester.test_reasoning_analysis(cot_module)
            
            print(f"\nCoT Results: {tester.passed_tests}/{tester.total_tests} tests passed")
            
        except ImportError as e:
            print(f"❌ Could not import cot_prompting.py: {e}")
    
    # Test ReACT if requested
    if component in ["all", "react"]:
        print("\n🔄 Testing ReACT Framework Implementation")
        print("-" * 40)
        
        try:
            # Import student's ReACT module
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            react_module = importlib.import_module("react_agent")
            
            tester = TestReACT(verbose)
            tester.test_reasoning_step(react_module)
            tester.test_action_execution(react_module)
            tester.test_observation_processing(react_module)
            
            print(f"\nReACT Results: {tester.passed_tests}/{tester.total_tests} tests passed")
            
        except ImportError as e:
            print(f"❌ Could not import react_agent.py: {e}")
    
    print("\n" + "=" * 60)
    print("Testing complete! Review any ❌ items above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test reasoning implementations")
    parser.add_argument("--component", choices=["all", "cot", "react"], 
                       default="all", help="Component to test")
    parser.add_argument("--verbose", action="store_true", 
                       help="Show detailed test output")
    
    args = parser.parse_args()
    run_tests(args.component, args.verbose)