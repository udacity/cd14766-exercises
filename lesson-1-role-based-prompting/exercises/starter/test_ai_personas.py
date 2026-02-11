"""
Enhanced Test Suite for Lesson 1: Role-Based Prompting with AI Integration

This module provides comprehensive testing for both classic personas and 
AI integration functionality (TODOs 1-5).

Usage:
    python test_ai_personas.py --verbose           # Full test with details
    python test_ai_personas.py --component classic # Test classic personas only
    python test_ai_personas.py --component ai      # Test AI integration only
    python test_ai_personas.py --component comparison # Test persona comparison
"""

import os
import sys
import argparse
import importlib
from typing import Dict, Any


class TestClassicPersonas:
    """Test classic persona implementations (TODOs 1-3)."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.passed_tests = 0
        self.total_tests = 0
    
    def test_persona_content(self, persona_text: str, persona_name: str, todo_num: int) -> bool:
        """Test individual persona content quality."""
        self.total_tests += 1
        
        if "[YOUR TODO" in persona_text:
            print(f"❌ TODO {todo_num}: {persona_name} not implemented")
            return False
        
        # Content quality checks
        word_count = len(persona_text.split())
        if word_count < 100:
            print(f"❌ TODO {todo_num}: {persona_name} too short ({word_count} words, need 100+)")
            return False
        
        # Required sections
        required_sections = ["Role:", "Expertise:", "Communication Style:", "Analytical Approach:"]
        missing_sections = [section for section in required_sections if section not in persona_text]
        
        if missing_sections:
            print(f"❌ TODO {todo_num}: {persona_name} missing sections: {missing_sections}")
            return False
        
        # Framework checks by persona type
        framework_checks = {
            "Business Analyst": ["TAM", "SAM", "SOM", "market", "analysis"],
            "Market Researcher": ["Porter", "competitive", "market share", "positioning"],
            "Strategic Consultant": ["risk", "strategy", "ROI", "implementation"]
        }
        
        expected_terms = framework_checks.get(persona_name, [])
        found_terms = sum(1 for term in expected_terms if term.lower() in persona_text.lower())
        
        if found_terms < len(expected_terms) // 2:
            print(f"❌ TODO {todo_num}: {persona_name} missing key terminology")
            if self.verbose:
                print(f"  Expected terms: {expected_terms}")
                print(f"  Found: {found_terms}/{len(expected_terms)}")
            return False
        
        self.passed_tests += 1
        print(f"✅ TODO {todo_num}: {persona_name} implemented correctly")
        return True


class TestAIIntegration:
    """Test AI integration functionality (TODOs 4-5)."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.passed_tests = 0
        self.total_tests = 0
    
    def test_vertex_ai_setup(self, ai_module) -> bool:
        """Test Vertex AI client initialization."""
        self.total_tests += 1
        
        try:
            # Test if PersonaAITester can be instantiated
            tester = ai_module.PersonaAITester("test-project-id")
            
            # Check if required attributes exist
            required_attrs = ["client", "model_name", "personas"]
            for attr in required_attrs:
                if not hasattr(tester, attr):
                    print(f"❌ AI Setup: Missing attribute '{attr}'")
                    return False
            
            # Check if personas are loaded
            if len(tester.personas) != 3:
                print(f"❌ AI Setup: Expected 3 personas, found {len(tester.personas)}")
                return False
            
            self.passed_tests += 1
            print("✅ AI Setup: Vertex AI client initialized correctly")
            return True
            
        except Exception as e:
            print(f"❌ AI Setup: Initialization failed - {str(e)}")
            return False
    
    def test_persona_testing_function(self, ai_module) -> bool:
        """Test TODO 4: test_persona_with_scenario implementation."""
        self.total_tests += 1
        
        try:
            tester = ai_module.PersonaAITester("test-project-id")
            
            # Mock scenario
            test_scenario = {
                "name": "Test Scenario",
                "company_name": "Test Corp",
                "industry": "Technology",
                "market_focus": "software development", 
                "strategic_question": "Should we expand internationally?",
                "additional_context": "Growing startup with limited resources"
            }
            
            # Test method exists and is callable
            if not hasattr(tester, 'test_persona_with_scenario'):
                print("❌ TODO 4: test_persona_with_scenario method not found")
                return False
            
            # Test method signature (should not crash with mock call)
            result = tester.test_persona_with_scenario("business_analyst", test_scenario)
            
            if result is None:
                print("❌ TODO 4: test_persona_with_scenario returns None (not implemented)")
                return False
            
            # Check for expected result structure
            if isinstance(result, dict) and "error" not in result:
                expected_fields = ["persona", "scenario", "response", "quality_analysis"]
                missing_fields = [field for field in expected_fields if field not in result]
                
                if missing_fields:
                    print(f"❌ TODO 4: Missing result fields: {missing_fields}")
                    return False
            
            self.passed_tests += 1
            print("✅ TODO 4: Persona testing function implemented")
            return True
            
        except Exception as e:
            print(f"❌ TODO 4: Function test failed - {str(e)}")
            return False
    
    def test_persona_comparison_function(self, ai_module) -> bool:
        """Test TODO 5: compare_personas implementation."""
        self.total_tests += 1
        
        try:
            tester = ai_module.PersonaAITester("test-project-id")
            
            # Mock scenario
            test_scenario = {
                "name": "Test Comparison",
                "company_name": "CompareTest Corp",
                "industry": "Test Industry",
                "market_focus": "test market",
                "strategic_question": "Test question?",
                "additional_context": "Test context"
            }
            
            # Test method exists
            if not hasattr(tester, 'compare_personas'):
                print("❌ TODO 5: compare_personas method not found")
                return False
            
            # Test method execution
            result = tester.compare_personas(test_scenario)
            
            if result is None:
                print("❌ TODO 5: compare_personas returns None (not implemented)")
                return False
            
            # Check for expected result structure
            if isinstance(result, dict):
                # Should have results for each persona
                expected_personas = ["business_analyst", "market_researcher", "strategic_consultant"]
                missing_personas = [p for p in expected_personas if p not in result]
                
                if len(missing_personas) == len(expected_personas):
                    print("❌ TODO 5: No persona results found")
                    return False
            
            self.passed_tests += 1
            print("✅ TODO 5: Persona comparison function implemented")
            return True
            
        except Exception as e:
            print(f"❌ TODO 5: Comparison test failed - {str(e)}")
            return False


def test_environment_setup():
    """Test if environment is properly configured."""
    print("🔧 Environment Setup Check")
    print("-" * 30)
    
    # Check PROJECT_ID
    project_id = os.getenv("PROJECT_ID")
    if project_id and project_id != "your-project-id":
        print("✅ PROJECT_ID environment variable set")
    else:
        print("❌ PROJECT_ID not set or using placeholder")
        print("   Run: export PROJECT_ID=your-gcp-project-id")
        return False
    
    # Check if Vertex AI imports work
    try:
        import vertexai
        from google import genai
        print("✅ Vertex AI dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Run: pip install google-genai vertexai")
        return False
    
    return True


def run_tests(component: str = "all", verbose: bool = False):
    """Run test suite for specified component."""
    
    print("=" * 60)
    print("LESSON 1: ENHANCED PERSONA TESTS")
    print("=" * 60)
    
    # Environment check
    if not test_environment_setup():
        print("\n❌ Environment setup failed. Please fix issues above.")
        return
    
    # Test classic personas if requested
    if component in ["all", "classic"]:
        print("\n📝 Testing Classic Personas (TODOs 1-3)")
        print("-" * 40)
        
        try:
            # Import the AI-enhanced personas module
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            ai_module = importlib.import_module("personas_with_ai")
            
            classic_tester = TestClassicPersonas(verbose)
            
            # Test each persona
            personas_to_test = [
                (ai_module.BUSINESS_ANALYST_PERSONA, "Business Analyst", 6),
                (ai_module.MARKET_RESEARCHER_PERSONA, "Market Researcher", 7),
                (ai_module.STRATEGIC_CONSULTANT_PERSONA, "Strategic Consultant", 8)
            ]
            
            for persona_text, persona_name, todo_num in personas_to_test:
                classic_tester.test_persona_content(persona_text, persona_name, todo_num)
            
            print(f"\nClassic Personas: {classic_tester.passed_tests}/{classic_tester.total_tests} tests passed")
            
        except ImportError as e:
            print(f"❌ Could not import personas_with_ai.py: {e}")
    
    # Test AI integration if requested
    if component in ["all", "ai", "comparison"]:
        print("\n🤖 Testing AI Integration (TODOs 4-5)")
        print("-" * 40)
        
        try:
            # Import AI module
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            ai_module = importlib.import_module("personas_with_ai")
            
            ai_tester = TestAIIntegration(verbose)
            
            # Test AI components
            ai_tester.test_vertex_ai_setup(ai_module)
            ai_tester.test_persona_testing_function(ai_module)
            ai_tester.test_persona_comparison_function(ai_module)
            
            print(f"\nAI Integration: {ai_tester.passed_tests}/{ai_tester.total_tests} tests passed")
            
        except ImportError as e:
            print(f"❌ Could not import personas_with_ai.py: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS:")
    
    if component in ["all", "classic"]:
        print("1. Ensure all classic personas (TODOs 1-3) are complete")
    
    if component in ["all", "ai"]:
        print("2. Implement AI testing (TODO 4) with Vertex AI")
        print("3. Implement persona comparison (TODO 5)")
        print("4. Test with: python personas_with_ai.py")
    
    print("5. Ready for Lesson 2: Chain-of-Thought and ReACT Prompting!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test enhanced persona implementations")
    parser.add_argument("--component", choices=["all", "classic", "ai", "comparison"], 
                       default="all", help="Component to test")
    parser.add_argument("--verbose", action="store_true", 
                       help="Show detailed test output")
    
    args = parser.parse_args()
    run_tests(args.component, args.verbose)