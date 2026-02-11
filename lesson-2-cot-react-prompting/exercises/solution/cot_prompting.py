"""
Lesson 2: Chain-of-Thought Prompting - Complete Solution

This module demonstrates the implementation of Chain-of-Thought (CoT) prompting
techniques for improved reasoning and problem-solving with Vertex AI Gemini.

Learning Objectives:
- Implement step-by-step reasoning prompts
- Compare standard vs CoT approaches
- Analyze reasoning quality and accuracy

Author: Noble Ackerson (Udacity)
Date: 2025
"""

import os
import re
from typing import Dict, List, Tuple, Any
from google import genai
from google.genai.types import GenerateContentConfig


class ChainOfThoughtPrompting:
    """
    Implements Chain-of-Thought prompting techniques for complex problem solving.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize with Vertex AI Gemini client."""
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        self.model_name = "gemini-2.5-flash"
        
    def get_standard_prompt(self, problem: str) -> str:
        """
        TODO 1 SOLUTION: Create a standard problem-solving prompt.
        
        This baseline prompt asks for a direct answer without requiring
        step-by-step reasoning.
        """
        return f"""
You are a business analyst. Answer the following question:

{problem}

Provide your answer concisely.
"""
    
    def get_cot_prompt(self, problem: str) -> str:
        """
        TODO 2 SOLUTION: Create a Chain-of-Thought prompt.
        
        This enhanced prompt guides the model through step-by-step reasoning
        with explicit markers and structure.
        """
        return f"""
You are a senior business analyst solving a complex problem. Think through this step-by-step.

Problem: {problem}

Let's approach this systematically:

Step 1: First, identify the key components and what we need to calculate.
Step 2: Break down the problem into manageable parts.
Step 3: Perform necessary calculations or analysis for each part.
Step 4: Validate the logic and check for consistency.
Step 5: Synthesize the findings into a final conclusion.

Important: 
- Show all your work and calculations
- Explain your reasoning at each step
- Use "Therefore", "This means", "Based on" to connect ideas
- Number each step clearly

Begin your step-by-step analysis:
"""
    
    def analyze_reasoning_quality(self, response: str) -> Dict[str, Any]:
        """
        TODO 3 SOLUTION: Analyze the quality of reasoning in the response.
        
        Extracts reasoning steps, validates logic, and measures quality metrics.
        """
        analysis = {
            "reasoning_steps": [],
            "has_calculations": False,
            "has_logical_connectors": False,
            "step_count": 0,
            "coherence_score": 0.0,
            "confidence_score": 0.0
        }
        
        # Extract reasoning steps (look for Step patterns)
        step_patterns = [
            r'Step \d+:.*?(?=Step \d+:|$)',
            r'\d+\.\s+.*?(?=\d+\.|$)',
            r'First,.*?(?=Second,|Next,|Then,|Finally,|$)',
            r'Second,.*?(?=Third,|Next,|Then,|Finally,|$)',
            r'Third,.*?(?=Fourth,|Next,|Then,|Finally,|$)',
            r'Finally,.*?$'
        ]
        
        for pattern in step_patterns:
            steps = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if steps:
                analysis["reasoning_steps"].extend([s.strip() for s in steps])
        
        analysis["step_count"] = len(analysis["reasoning_steps"])
        
        # Check for calculations (numbers and mathematical operations)
        calc_patterns = [r'\d+\s*[+\-*/]\s*\d+', r'\$[\d,]+', r'\d+%', r'=\s*\d+']
        for pattern in calc_patterns:
            if re.search(pattern, response):
                analysis["has_calculations"] = True
                break
        
        # Check for logical connectors
        logical_connectors = [
            'therefore', 'thus', 'hence', 'consequently',
            'this means', 'this indicates', 'based on',
            'as a result', 'it follows that', 'we can conclude'
        ]
        response_lower = response.lower()
        connector_count = sum(1 for conn in logical_connectors if conn in response_lower)
        analysis["has_logical_connectors"] = connector_count > 0
        
        # Calculate coherence score based on structure
        coherence_factors = [
            (analysis["step_count"] >= 3, 0.3),  # Has multiple steps
            (analysis["has_calculations"], 0.2),   # Shows work
            (analysis["has_logical_connectors"], 0.2),  # Uses connectors
            (len(response.split()) > 100, 0.15),  # Adequate detail
            ('therefore' in response_lower or 'conclusion' in response_lower, 0.15)  # Has conclusion
        ]
        
        analysis["coherence_score"] = sum(score for condition, score in coherence_factors if condition)
        
        # Estimate confidence based on reasoning quality
        if analysis["step_count"] >= 3 and analysis["has_logical_connectors"]:
            analysis["confidence_score"] = min(0.9, 0.6 + (analysis["step_count"] * 0.1))
        else:
            analysis["confidence_score"] = 0.5
        
        return analysis
    
    def compare_approaches(self, problem: str) -> Dict[str, Any]:
        """
        Compare standard vs Chain-of-Thought approaches for the same problem.
        """
        results = {}
        
        # Get standard response
        standard_prompt = self.get_standard_prompt(problem)
        standard_response = self.client.models.generate_content(
            model=self.model_name,
            contents=standard_prompt,
            config=GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=500
            )
        )
        results["standard"] = {
            "response": standard_response.text or "",
            "analysis": self.analyze_reasoning_quality(standard_response.text or "")
        }
        
        # Get CoT response
        cot_prompt = self.get_cot_prompt(problem)
        cot_response = self.client.models.generate_content(
            model=self.model_name,
            contents=cot_prompt,
            config=GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1000
            )
        )
        results["cot"] = {
            "response": cot_response.text or "",
            "analysis": self.analyze_reasoning_quality(cot_response.text or "")
        }
        
        # Calculate improvement
        standard_score = results["standard"]["analysis"]["coherence_score"]
        cot_score = results["cot"]["analysis"]["coherence_score"]
        results["improvement"] = {
            "coherence_increase": cot_score - standard_score,
            "step_increase": (results["cot"]["analysis"]["step_count"] - 
                            results["standard"]["analysis"]["step_count"]),
            "percentage_improvement": ((cot_score - standard_score) / max(standard_score, 0.1)) * 100
        }
        
        return results


def run_demonstration():
    """Demonstrate Chain-of-Thought prompting with a business problem."""
    
    # Initialize with project ID
    PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id")
    if PROJECT_ID == "your-project-id":
        print("Please set PROJECT_ID environment variable")
        return
    
    cot = ChainOfThoughtPrompting(PROJECT_ID)
    
    # Test problem
    problem = """
    A software company is considering expanding into the Asian market. 
    They have $5M budget, expect 20% market growth, and competitors have 60% market share.
    Current US revenue is $50M with 30% profit margin.
    Should they expand, and if so, what's the expected ROI in Year 1?
    """
    
    print("=" * 60)
    print("CHAIN-OF-THOUGHT PROMPTING DEMONSTRATION")
    print("=" * 60)
    
    # Compare approaches
    results = cot.compare_approaches(problem)
    
    print("\n📊 STANDARD APPROACH:")
    print("-" * 40)
    print(f"Response length: {len(results['standard']['response'].split())} words")
    print(f"Reasoning steps: {results['standard']['analysis']['step_count']}")
    print(f"Coherence score: {results['standard']['analysis']['coherence_score']:.2f}")
    print(f"Has calculations: {results['standard']['analysis']['has_calculations']}")
    print("\nResponse preview:")
    print(results['standard']['response'][:300] + "...")
    
    print("\n🔍 CHAIN-OF-THOUGHT APPROACH:")
    print("-" * 40)
    print(f"Response length: {len(results['cot']['response'].split())} words")
    print(f"Reasoning steps: {results['cot']['analysis']['step_count']}")
    print(f"Coherence score: {results['cot']['analysis']['coherence_score']:.2f}")
    print(f"Has calculations: {results['cot']['analysis']['has_calculations']}")
    print(f"Logical connectors: {results['cot']['analysis']['has_logical_connectors']}")
    print("\nResponse preview:")
    print(results['cot']['response'][:500] + "...")
    
    print("\n📈 IMPROVEMENT METRICS:")
    print("-" * 40)
    print(f"Coherence improvement: +{results['improvement']['coherence_increase']:.2f}")
    print(f"Additional reasoning steps: +{results['improvement']['step_increase']}")
    print(f"Overall improvement: {results['improvement']['percentage_improvement']:.1f}%")
    
    # Show extracted reasoning steps
    if results['cot']['analysis']['reasoning_steps']:
        print("\n🎯 EXTRACTED REASONING STEPS:")
        print("-" * 40)
        for i, step in enumerate(results['cot']['analysis']['reasoning_steps'][:3], 1):
            print(f"{i}. {step[:100]}...")
    
    print("\n✅ Chain-of-Thought prompting demonstration complete!")


if __name__ == "__main__":
    run_demonstration()