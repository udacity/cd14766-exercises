"""
Lesson 2: Chain-of-Thought Prompting - Student Template

This module demonstrates the implementation of Chain-of-Thought (CoT) prompting
techniques for improved reasoning and problem-solving with Vertex AI Gemini.

Learning Objectives:
- Implement step-by-step reasoning prompts
- Compare standard vs CoT approaches  
- Analyze reasoning quality and accuracy

Complete TODOs 1-3 to implement Chain-of-Thought prompting.

Author: [Your Name]
Date: [Current Date]
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
        TODO 1: Create a standard problem-solving prompt.
        
        Requirements:
        - Simple, direct question format
        - No reasoning steps required
        - This will be your baseline for comparison
        
        Example structure:
        "You are a [role]. Answer the following question: [problem]
         Provide your answer concisely."
        
        Args:
            problem: The business problem to solve
            
        Returns:
            A standard prompt string
        """
        # TODO 1: Implement standard prompt
        # Hint: Keep it simple - just ask for the answer directly
        pass
    
    def get_cot_prompt(self, problem: str) -> str:
        """
        TODO 2: Create a Chain-of-Thought prompt.
        
        Requirements:
        - Guide step-by-step reasoning with explicit markers
        - Include phrases like "Let's think step by step"
        - Structure the analysis process (Step 1, Step 2, etc.)
        - Request showing work and calculations
        - Ask for logical connections between steps
        
        Key elements to include:
        1. Role and expertise context
        2. Clear instruction to think step-by-step
        3. Structured approach (numbered steps)
        4. Request for showing calculations
        5. Emphasis on explaining reasoning
        
        Args:
            problem: The business problem to solve
            
        Returns:
            A Chain-of-Thought prompt string
        """
        # TODO 2: Implement Chain-of-Thought prompt
        # Your prompt should include:
        # - "Let's think step by step" or similar phrase
        # - Numbered steps structure
        # - Instructions to show work
        # - Request for clear reasoning at each step
        pass
    
    def analyze_reasoning_quality(self, response: str) -> Dict[str, Any]:
        """
        TODO 3: Analyze the quality of reasoning in the response.
        
        Requirements:
        - Extract reasoning steps from the response
        - Check for presence of calculations
        - Detect logical connectors
        - Calculate quality scores
        
        Return dictionary should include:
        - reasoning_steps: List of extracted steps
        - has_calculations: Boolean for presence of math
        - has_logical_connectors: Boolean for connectors
        - step_count: Number of reasoning steps found
        - coherence_score: Float between 0 and 1
        - confidence_score: Float between 0 and 1
        
        Scoring guidelines:
        - 3+ steps: High quality
        - Calculations present: +0.2 to score
        - Logical connectors: +0.2 to score
        - 100+ words: Indicates detail
        
        Args:
            response: The model's response to analyze
            
        Returns:
            Dictionary with analysis metrics
        """
        analysis = {
            "reasoning_steps": [],
            "has_calculations": False,
            "has_logical_connectors": False,
            "step_count": 0,
            "coherence_score": 0.0,
            "confidence_score": 0.0
        }
        
        # TODO 3: Implement reasoning analysis
        # 
        # Step 1: Extract reasoning steps
        # Hint: Look for patterns like "Step 1:", "First,", "Second,", etc.
        # You can use regex: r'Step \d+:.*?(?=Step \d+:|$)'
        
        # Step 2: Check for calculations
        # Hint: Look for numbers with operators: \d+\s*[+\-*/]\s*\d+
        
        # Step 3: Check for logical connectors
        # Hint: Search for words like 'therefore', 'thus', 'hence', 'consequently'
        
        # Step 4: Calculate coherence score
        # Hint: Base it on:
        # - Number of steps (3+ is good)
        # - Presence of calculations
        # - Use of logical connectors
        # - Response length
        
        # Step 5: Calculate confidence score
        # Hint: Higher confidence if more steps and better structure
        
        pass
    
    def compare_approaches(self, problem: str) -> Dict[str, Any]:
        """
        Compare standard vs Chain-of-Thought approaches for the same problem.
        
        This method is provided complete to help you test your implementations.
        """
        results = {}
        
        # Get standard response
        standard_prompt = self.get_standard_prompt(problem)
        if standard_prompt:  # Check if TODO 1 is implemented
            standard_response = self.client.models.generate_content(
                model=self.model_name,
                contents=standard_prompt,
                config=GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=500
                )
            )
            results["standard"] = {
                "response": standard_response.text,
                "analysis": self.analyze_reasoning_quality(standard_response.text)
            }
        
        # Get CoT response
        cot_prompt = self.get_cot_prompt(problem)
        if cot_prompt:  # Check if TODO 2 is implemented
            cot_response = self.client.models.generate_content(
                model=self.model_name,
                contents=cot_prompt,
                config=GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1000
                )
            )
            results["cot"] = {
                "response": cot_response.text,
                "analysis": self.analyze_reasoning_quality(cot_response.text)
            }
        
        # Calculate improvement if both approaches are implemented
        if "standard" in results and "cot" in results:
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
    
    # Test if TODOs are implemented
    if not cot.get_standard_prompt(problem):
        print("\n❌ TODO 1 not implemented: Create standard prompt")
        return
        
    if not cot.get_cot_prompt(problem):
        print("\n❌ TODO 2 not implemented: Create CoT prompt")
        return
    
    # Compare approaches
    results = cot.compare_approaches(problem)
    
    if "standard" in results:
        print("\n📊 STANDARD APPROACH:")
        print("-" * 40)
        analysis = results['standard']['analysis']
        if analysis['step_count'] > 0:  # Check if TODO 3 is implemented
            print(f"Response length: {len(results['standard']['response'].split())} words")
            print(f"Reasoning steps: {analysis['step_count']}")
            print(f"Coherence score: {analysis['coherence_score']:.2f}")
            print(f"Has calculations: {analysis['has_calculations']}")
        else:
            print("❌ TODO 3 not implemented: Analyze reasoning quality")
            
    if "cot" in results:
        print("\n🔍 CHAIN-OF-THOUGHT APPROACH:")
        print("-" * 40)
        analysis = results['cot']['analysis']
        if analysis['step_count'] > 0:
            print(f"Response length: {len(results['cot']['response'].split())} words")
            print(f"Reasoning steps: {analysis['step_count']}")
            print(f"Coherence score: {analysis['coherence_score']:.2f}")
            print(f"Has calculations: {analysis['has_calculations']}")
            print(f"Logical connectors: {analysis['has_logical_connectors']}")
    
    if "improvement" in results:
        print("\n📈 IMPROVEMENT METRICS:")
        print("-" * 40)
        print(f"Coherence improvement: +{results['improvement']['coherence_increase']:.2f}")
        print(f"Additional reasoning steps: +{results['improvement']['step_increase']}")
        print(f"Overall improvement: {results['improvement']['percentage_improvement']:.1f}%")
    
    print("\n✅ Chain-of-Thought prompting demonstration complete!")


if __name__ == "__main__":
    run_demonstration()