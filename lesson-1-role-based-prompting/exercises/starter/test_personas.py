#!/usr/bin/env python3
"""
Lesson 1 Test Suite: Role-Based Prompting

This test suite validates student implementations of TODOs 6, 7, and 8.
Run this file to check if your persona designs meet the requirements.

Usage:
    python test_personas.py                    # Test all personas
    python test_personas.py --verbose          # Detailed feedback
    python test_personas.py --persona analyst  # Test specific persona

Author: Noble Ackerson (Udacity)
Date: 2025
"""

import argparse
import sys
from personas import (
    BUSINESS_ANALYST_PERSONA, 
    MARKET_RESEARCHER_PERSONA, 
    STRATEGIC_CONSULTANT_PERSONA,
    validate_persona
)


class PersonaTester:
    """Test suite for persona implementations"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.personas = {
            "business_analyst": {
                "name": "Business Analyst", 
                "persona": BUSINESS_ANALYST_PERSONA,
                "todo": "TODO 1"
            },
            "market_researcher": {
                "name": "Market Researcher",
                "persona": MARKET_RESEARCHER_PERSONA, 
                "todo": "TODO 2"
            },
            "strategic_consultant": {
                "name": "Strategic Consultant",
                "persona": STRATEGIC_CONSULTANT_PERSONA,
                "todo": "TODO 3"
            }
        }
    
    def test_persona_implementation(self, persona_key: str) -> dict:
        """Test a specific persona implementation"""
        persona_info = self.personas[persona_key]
        result = validate_persona(persona_info["persona"], persona_info["name"])
        result["todo"] = persona_info["todo"]
        return result
    
    def test_all_personas(self) -> dict:
        """Test all persona implementations"""
        results = {}
        total_score = 0
        
        print("🧪 TESTING PERSONA IMPLEMENTATIONS")
        print("=" * 60)
        
        for key, info in self.personas.items():
            result = self.test_persona_implementation(key)
            results[key] = result
            total_score += result["score"]
            
            # Display result
            status = "✅ PASS" if result["valid"] else "❌ FAIL"
            print(f"{info['name']} ({info['todo']}): {status}")
            print(f"   Score: {result['score']:.2f}/1.00")
            print(f"   Feedback: {result['feedback']}")
            
            if self.verbose and "checks" in result:
                print("   Detailed Checks:")
                for check, passed in result["checks"].items():
                    check_status = "✅" if passed else "❌"
                    print(f"     {check_status} {check.replace('_', ' ').title()}")
            print()
        
        avg_score = total_score / len(self.personas)
        overall_pass = avg_score >= 0.8
        
        print("=" * 60)
        print(f"OVERALL RESULT: {'✅ ALL PASS' if overall_pass else '❌ NEEDS WORK'}")
        print(f"Average Score: {avg_score:.2f}/1.00")
        print(f"Grade: {self._calculate_grade(avg_score)}")
        
        # Provide next steps
        if overall_pass:
            print("\n🎉 Excellent work! Your personas are ready.")
            print("📝 Next Step: Move to Lesson 2 - Quality Validation")
        else:
            print("\n💡 Keep working on your persona designs.")
            print("📖 Tips:")
            print("   - Each persona should be 50+ words minimum")
            print("   - Include specific business frameworks")
            print("   - Define clear communication styles")
            print("   - Add professional experience details")
        
        print("=" * 60)
        
        return {
            "overall_pass": overall_pass,
            "average_score": avg_score,
            "individual_results": results
        }
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B" 
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def test_single_persona(self, persona_key: str):
        """Test and display results for a single persona"""
        if persona_key not in self.personas:
            print(f"❌ Unknown persona: {persona_key}")
            print(f"Available personas: {', '.join(self.personas.keys())}")
            return
        
        info = self.personas[persona_key]
        result = self.test_persona_implementation(persona_key)
        
        print(f"🧪 TESTING {info['name'].upper()} ({info['todo']})")
        print("=" * 50)
        
        status = "✅ PASS" if result["valid"] else "❌ FAIL"
        print(f"Status: {status}")
        print(f"Score: {result['score']:.2f}/1.00")
        print(f"Grade: {self._calculate_grade(result['score'])}")
        print(f"Feedback: {result['feedback']}")
        
        if "checks" in result:
            print("\nDetailed Checks:")
            for check, passed in result["checks"].items():
                check_status = "✅" if passed else "❌"
                print(f"  {check_status} {check.replace('_', ' ').title()}")
        
        # Show persona preview
        print(f"\nPersona Preview (first 200 characters):")
        print(f'"{info["persona"][:200]}..."')
        
        print("=" * 50)


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test persona implementations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--persona", "-p", choices=["business_analyst", "market_researcher", "strategic_consultant", "analyst", "researcher", "consultant"], help="Test specific persona")
    
    args = parser.parse_args()
    
    # Map short names to full names
    persona_mapping = {
        "analyst": "business_analyst",
        "researcher": "market_researcher", 
        "consultant": "strategic_consultant"
    }
    
    if args.persona:
        persona_key = persona_mapping.get(args.persona, args.persona)
        tester = PersonaTester(verbose=True)  # Always verbose for single tests
        tester.test_single_persona(persona_key)
    else:
        tester = PersonaTester(verbose=args.verbose)
        results = tester.test_all_personas()
        
        # Exit with appropriate code
        if results["overall_pass"]:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure


if __name__ == "__main__":
    main()