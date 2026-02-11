"""
Lesson 2: ReACT Framework Implementation - Student Template

This module implements the ReACT (Reason, Act, Observe) pattern for iterative
problem-solving with tool integration using Vertex AI Gemini.

Learning Objectives:
- Build Reason→Act→Observe loops
- Integrate function calling for tool use
- Implement iterative problem-solving with context management

Complete TODOs 4-6 to implement the ReACT framework.

Author: [Your Name]
Date: [Current Date]
"""

import os
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from google import genai
from google.genai.types import GenerateContentConfig, Tool, FunctionDeclaration, Schema, Type


@dataclass
class ReACTContext:
    """Maintains context across ReACT iterations."""
    problem: str
    observations: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    current_iteration: int = 0
    is_complete: bool = False
    final_answer: Optional[str] = None


class ReACTAgent:
    """
    Implements the ReACT framework for iterative problem-solving with tools.
    """
    
    MAX_ITERATIONS = 5
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize with Vertex AI Gemini client."""
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        self.model_name = "gemini-2.5-flash"
        
        # Define available tools
        self.tools = self._define_tools()
        
    def _define_tools(self) -> List[Tool]:
        """Define function tools for Gemini to use."""
        return [
            Tool(
                function_declarations=[
                    FunctionDeclaration(
                        name="calculate",
                        description="Perform mathematical calculations",
                        parameters=Schema(
                            type=Type.OBJECT,
                            properties={
                                "expression": Schema(
                                    type=Type.STRING,
                                    description="Mathematical expression to evaluate"
                                )
                            },
                            required=["expression"]
                        )
                    ),
                    FunctionDeclaration(
                        name="get_market_data",
                        description="Retrieve market analysis data",
                        parameters=Schema(
                            type=Type.OBJECT,
                            properties={
                                "market": Schema(
                                    type=Type.STRING,
                                    description="Market region (e.g., Asia, Europe, US)"
                                ),
                                "metric": Schema(
                                    type=Type.STRING,
                                    enum=["size", "growth", "share", "segments"],
                                    description="Type of market data"
                                )
                            },
                            required=["market", "metric"]
                        )
                    ),
                    FunctionDeclaration(
                        name="analyze_competitors",
                        description="Get competitive analysis data",
                        parameters=Schema(
                            type=Type.OBJECT,
                            properties={
                                "market": Schema(
                                    type=Type.STRING,
                                    description="Market to analyze"
                                ),
                                "aspect": Schema(
                                    type=Type.STRING,
                                    enum=["market_share", "strengths", "weaknesses", "strategy"],
                                    description="Aspect of competition to analyze"
                                )
                            },
                            required=["market", "aspect"]
                        )
                    )
                ]
            )
        ]
    
    def reason(self, context: ReACTContext) -> str:
        """
        TODO 4: Implement the reasoning step.
        
        Requirements:
        - Analyze the current context (problem + observations)
        - Determine what information is still needed
        - Plan the next action to take
        - Return reasoning in a structured format
        
        Format to use in your prompt:
        THOUGHT: [Analysis of current situation]
        NEED: [What information is needed next]
        ACTION: [Specific action to take]
        REASON: [Why this action helps]
        
        If enough information exists, use:
        THOUGHT: I have sufficient information
        FINAL_ANSWER: [Complete solution]
        
        Args:
            context: Current ReACTContext with problem and observations
            
        Returns:
            String containing structured reasoning
        """
        # TODO 4: Build reasoning prompt
        # 
        # Step 1: Create a context summary
        # Include:
        # - The original problem
        # - List of previous observations (if any)
        # - Current iteration number
        
        # Step 2: Create a reasoning prompt that asks Gemini to:
        # - Analyze what information we have
        # - Identify what's still missing
        # - Decide on next action
        # - Use the structured format (THOUGHT, NEED, ACTION, REASON)
        
        # Step 3: Call Gemini with the reasoning prompt
        # Use temperature=0.3 for consistent reasoning
        
        # Step 4: Return the reasoning response

        return ""  # TODO: Replace with actual implementation
    
    def act(self, reasoning: str, context: ReACTContext) -> Dict[str, Any]:
        """
        TODO 5: Execute actions based on reasoning.
        
        Requirements:
        - Parse the reasoning to extract the action
        - Execute the appropriate tool/function
        - Return results in a structured format
        
        Action types to handle:
        1. "calculate" - Perform mathematical calculations
        2. "get_market_data" - Retrieve market information
        3. "analyze_competitors" - Get competitive analysis
        4. "final_answer" - Problem is solved
        
        Return format:
        {"type": action_type, "result": action_result, ...}
        
        Args:
            reasoning: The reasoning output from the previous step
            context: Current ReACTContext
            
        Returns:
            Dictionary with action type and results
        """
        # TODO 5: Parse reasoning and execute action
        # 
        # Step 1: Check for FINAL_ANSWER
        # If found, return {"type": "final_answer", "result": answer}
        
        # Step 2: Extract ACTION from reasoning
        # Use regex: r'ACTION:\s*(\w+)'
        # If no action found, return error
        
        # Step 3: Execute action based on type
        # For "calculate":
        #   - Extract mathematical expression
        #   - Use ast.literal_eval() or safe math parser (avoid eval())
        #   - Return calculation result
        
        # For "get_market_data":
        #   - Extract market and metric from reasoning
        #   - Use provided market_data dictionary (simulate data)
        #   - Return market information
        
        # For "analyze_competitors":
        #   - Extract aspect from reasoning
        #   - Use provided competitor_data (simulate data)
        #   - Return analysis
        
        # Sample market data to use:
        market_data = {
            ("asia", "size"): "$2.5 trillion total addressable market",
            ("asia", "growth"): "15% annual growth rate",
            ("asia", "share"): "Top 3 competitors hold 60% market share",
            # Add more as needed
        }

        return {"type": "error", "result": "Not implemented"}  # TODO: Replace with actual implementation
    
    def observe(self, action_result: Dict[str, Any], context: ReACTContext) -> str:
        """
        TODO 6: Process action results and update context.
        
        Requirements:
        - Convert action results into observations
        - Format observations clearly
        - Check if the problem is solved
        - Update context flags if complete
        
        Observation format examples:
        - "Calculation result: 5M * 0.2 = 1M"
        - "Market data (Asia - size): $2.5 trillion"
        - "Competitor analysis: 60% market share held by top 3"
        - "Final answer determined: [answer]"
        
        Args:
            action_result: Results from the act() method
            context: Current ReACTContext to update
            
        Returns:
            Formatted observation string
        """
        # TODO 6: Process results into observations
        # 
        # Step 1: Check action type and format observation
        # For "final_answer":
        #   - Set context.is_complete = True
        #   - Set context.final_answer
        #   - Return "Final answer determined: ..."
        
        # For "calculation":
        #   - Format: "Calculation result: [expression] = [result]"
        
        # For "market_data":
        #   - Format: "Market data ([market] - [metric]): [result]"
        
        # For "competitor_analysis":
        #   - Format: "Competitor analysis ([aspect]): [result]"
        
        # For "error":
        #   - Format: "Error occurred: [message]"
        
        # Step 2: Check iteration limit
        # If context.current_iteration >= MAX_ITERATIONS - 1:
        #   - Set context.is_complete = True
        #   - Add "(Max iterations reached)" to observation
        
        # Step 3: Return formatted observation

        return ""  # TODO: Replace with actual implementation
    
    def solve(self, problem: str) -> str:
        """
        Main ReACT loop to solve the problem iteratively.
        
        This method is provided complete to orchestrate your ReACT implementation.
        """
        context = ReACTContext(problem=problem)
        
        print("\n" + "="*60)
        print("REACT PROBLEM SOLVING")
        print("="*60)
        print(f"Problem: {problem}\n")
        
        while not context.is_complete and context.current_iteration < self.MAX_ITERATIONS:
            print(f"\n--- Iteration {context.current_iteration + 1} ---")
            
            # REASON: Analyze current state
            print("🤔 REASONING...")
            reasoning = self.reason(context)
            if reasoning:  # Check if TODO 4 is implemented
                print(f"Reasoning: {reasoning[:200]}...")
            else:
                print("❌ TODO 4 not implemented")
                break
            
            # ACT: Execute action based on reasoning
            print("\n⚡ ACTING...")
            action_result = self.act(reasoning, context)
            if action_result:  # Check if TODO 5 is implemented
                context.actions_taken.append(str(action_result.get("type", "unknown")))
                print(f"Action: {action_result.get('type', 'unknown')}")
            else:
                print("❌ TODO 5 not implemented")
                break
            
            # OBSERVE: Process results and update context
            print("\n👁️ OBSERVING...")
            observation = self.observe(action_result, context)
            if observation:  # Check if TODO 6 is implemented
                context.observations.append(observation)
                print(f"Observation: {observation}")
            else:
                print("❌ TODO 6 not implemented")
                break
            
            context.current_iteration += 1
        
        # Return final solution
        if context.final_answer:
            return context.final_answer
        elif context.observations:
            # Synthesize answer from observations
            synthesis_prompt = f"""
Based on the following problem and observations, provide a final answer:

Problem: {problem}

Observations:
{chr(10).join(f"- {obs}" for obs in context.observations)}

Provide a clear, comprehensive answer with specific recommendations.
"""
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=synthesis_prompt,
                config=GenerateContentConfig(temperature=0.3, max_output_tokens=500)
            )
            return response.text or "Unable to generate synthesis"
        else:
            return "Unable to solve problem - please implement all TODOs"


def run_demonstration():
    """Demonstrate the ReACT framework with a business problem."""
    
    # Initialize with project ID
    PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id")
    if PROJECT_ID == "your-project-id":
        print("Please set PROJECT_ID environment variable")
        return
    
    agent = ReACTAgent(PROJECT_ID)
    
    # Test problem requiring multiple steps
    problem = """
    Should TechCorp expand into the Asian market? 
    They have a $5M budget and need at least 25% ROI in Year 1.
    Current revenue is $50M with 30% profit margin.
    Analyze market opportunity and competitive landscape to make a recommendation.
    """
    
    # Solve using ReACT
    solution = agent.solve(problem)
    
    print("\n" + "="*60)
    print("💡 FINAL SOLUTION")
    print("="*60)
    print(solution)
    
    print("\n✅ ReACT framework demonstration complete!")


if __name__ == "__main__":
    run_demonstration()