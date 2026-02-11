"""
Lesson 2: ReACT Framework Implementation - Complete Solution

This module implements the ReACT (Reason, Act, Observe) pattern for iterative
problem-solving with tool integration using Vertex AI Gemini.

Learning Objectives:
- Build Reason→Act→Observe loops
- Integrate function calling for tool use
- Implement iterative problem-solving with context management

Author: Noble Ackerson (Udacity)
Date: 2025
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
        TODO 4 SOLUTION: Implement the reasoning step.
        
        Analyzes current context and determines next action needed.
        """
        # Build context summary
        context_summary = f"Problem: {context.problem}\n"
        
        if context.observations:
            context_summary += "\nPrevious observations:\n"
            for i, obs in enumerate(context.observations, 1):
                context_summary += f"{i}. {obs}\n"
        
        reasoning_prompt = f"""
You are solving a business problem using the ReACT framework.
Current iteration: {context.current_iteration + 1} of {self.MAX_ITERATIONS}

{context_summary}

INSTRUCTIONS: You MUST respond using EXACTLY this format. Do not deviate from the format.

Available actions and how to request specific data:
1. get_market_data - Include keywords: asia/europe/us AND size/growth/share/segments
   Example: "NEED: Asia market growth rate data"
2. analyze_competitors - Include keywords: market_share/strengths/weaknesses/strategy
   Example: "NEED: Competitor strengths analysis"
3. calculate - For math calculations
   Example: "NEED: Calculate ROI: (1.25M / 5M) * 100"

Response format (use EXACT keywords in NEED):
THOUGHT: I need to understand market growth potential
NEED: Asia market growth data
ACTION: get_market_data
REASON: Growth rate will show if market is expanding

Another example:
THOUGHT: I need to assess competitive landscape
NEED: Competitor market_share analysis
ACTION: analyze_competitors
REASON: Understanding competition is critical for entry strategy

If you have enough information to answer, use this EXACT format:
THOUGHT: I have sufficient information from the analysis
FINAL_ANSWER: [Your complete recommendation with reasoning]
"""
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=reasoning_prompt,
            config=GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=500
            )
        )
        
        return response.text or ""
    
    def act(self, reasoning: str, context: ReACTContext) -> Dict[str, Any]:
        """
        TODO 5 SOLUTION: Execute actions based on reasoning.
        
        Parses reasoning to determine action and executes appropriate tool.
        """
        # Check if we have a final answer
        if "FINAL_ANSWER:" in reasoning:
            final_answer = reasoning.split("FINAL_ANSWER:")[1].strip()
            return {"type": "final_answer", "result": final_answer}
        
        # Extract action from reasoning
        action_match = re.search(r'ACTION:\s*(.+?)(?:\n|$)', reasoning, re.IGNORECASE)
        if not action_match:
            # Fallback: if no ACTION but mentions calculation/competition/market, infer it
            if re.search(r'\bcalculat|\bcomput|\bestimate|\broi\b', reasoning, re.IGNORECASE):
                action = 'calculate'
            elif re.search(r'\bcompetitor|\bcompetition|\bmarket share\b', reasoning, re.IGNORECASE):
                action = 'analyze_competitors'
            elif re.search(r'\bmarket.*data|\bgrowth|\bsegment', reasoning, re.IGNORECASE):
                action = 'get_market_data'
            else:
                return {"type": "error", "result": "No action specified"}
        else:
            action_text = action_match.group(1).strip().lower()

            # Map action variations to canonical actions
            if 'market' in action_text or 'data' in action_text:
                action = 'get_market_data'
            elif 'compet' in action_text or 'analyz' in action_text:
                action = 'analyze_competitors'
            elif 'calc' in action_text or 'comput' in action_text or 'estimate' in action_text:
                action = 'calculate'
            else:
                action = action_text.split()[0] if action_text else ''
        
        # Execute action based on type
        if action == "calculate":
            # Extract expression from reasoning
            expr_match = re.search(r'calculate[^\n]*?(\d+[\d\s+\-*/%.]+\d+)', reasoning, re.IGNORECASE)
            if expr_match:
                expression = expr_match.group(1)
                try:
                    # Safe evaluation of mathematical expression using ast
                    import ast
                    import operator
                    
                    # Define safe operations
                    ops = {
                        ast.Add: operator.add,
                        ast.Sub: operator.sub,
                        ast.Mult: operator.mul,
                        ast.Div: operator.truediv,
                        ast.Mod: operator.mod,
                        ast.Pow: operator.pow,
                        ast.USub: operator.neg,
                        ast.UAdd: operator.pos,
                    }
                    
                    def safe_eval(node):
                        if isinstance(node, ast.Constant):
                            return node.value
                        elif isinstance(node, ast.BinOp):
                            left = safe_eval(node.left)
                            right = safe_eval(node.right)
                            return ops[type(node.op)](left, right)
                        elif isinstance(node, ast.UnaryOp):
                            operand = safe_eval(node.operand)
                            return ops[type(node.op)](operand)
                        else:
                            raise ValueError(f"Unsupported operation: {type(node)}")
                    
                    tree = ast.parse(expression, mode='eval')
                    result = safe_eval(tree.body)
                    return {"type": "calculation", "expression": expression, "result": result}
                except:
                    return {"type": "error", "result": f"Failed to calculate: {expression}"}
            else:
                # Use Gemini for complex calculations
                calc_prompt = f"Calculate the following based on this context: {reasoning}"
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=calc_prompt,
                    config=GenerateContentConfig(temperature=0.1, max_output_tokens=200)
                )
                return {"type": "calculation", "result": response.text}
        
        elif action == "get_market_data":
            # Simulate market data retrieval
            market_data = {
                ("asia", "size"): "$2.5 trillion total addressable market",
                ("asia", "growth"): "15% annual growth rate",
                ("asia", "share"): "Top 3 competitors hold 60% market share",
                ("asia", "segments"): "Enterprise (40%), SMB (35%), Consumer (25%)",
                ("europe", "size"): "$1.8 trillion total addressable market",
                ("europe", "growth"): "8% annual growth rate",
                ("us", "size"): "$3.2 trillion total addressable market",
                ("us", "growth"): "10% annual growth rate",
            }
            
            # Extract market and metric from reasoning
            reasoning_lower = reasoning.lower()

            # Extract the NEED line specifically for better metric detection
            need_match = re.search(r'need:\s*(.+?)(?:\n|action:|$)', reasoning_lower, re.DOTALL)
            search_text = need_match.group(1) if need_match else reasoning_lower

            market_match = re.search(r'\b(asia|europe|us)\b', search_text)
            # Look for metric keywords in NEED line
            metric_match = re.search(r'\b(growth|growing|grow|segment|segments|share|size)\b', search_text)

            if market_match and metric_match:
                market = market_match.group(1)
                metric = metric_match.group(1)
                # Normalize metric variations
                if metric in ['growing', 'grow']:
                    metric = 'growth'
                elif metric in ['segment', 'segments']:
                    metric = 'segments'

                key = (market, metric)
                if key in market_data:
                    return {"type": "market_data", "market": key[0], "metric": key[1],
                           "result": market_data[key]}
            
            return {"type": "market_data", "result": "Market data: Asia market shows strong potential"}
        
        elif action == "analyze_competitors":
            # Simulate competitor analysis
            competitor_data = {
                "market_share": "Competitor A: 35%, Competitor B: 25%, Others: 40%",
                "strengths": "Strong brand recognition, established distribution channels",
                "weaknesses": "High prices, limited product innovation, slow adaptation",
                "strategy": "Focus on premium segment, acquisition-based growth"
            }
            
            aspect_match = re.search(r'(market_share|strengths|weaknesses|strategy)', reasoning.lower())
            aspect = aspect_match.group(1) if aspect_match else "market_share"
            
            return {"type": "competitor_analysis", "aspect": aspect, 
                   "result": competitor_data.get(aspect, "Competitors show vulnerability in innovation")}
        
        else:
            return {"type": "error", "result": f"Unknown action: {action}"}
    
    def observe(self, action_result: Dict[str, Any], context: ReACTContext) -> str:
        """
        TODO 6 SOLUTION: Process action results and update context.
        
        Converts action results into observations and determines if goal is achieved.
        """
        # Format observation based on action type
        if action_result["type"] == "final_answer":
            context.is_complete = True
            context.final_answer = action_result["result"]
            return f"Final answer determined: {action_result['result']}"
        
        elif action_result["type"] == "calculation":
            observation = f"Calculation result: {action_result.get('expression', 'complex calculation')} = {action_result['result']}"
        
        elif action_result["type"] == "market_data":
            observation = f"Market data ({action_result.get('market', 'target')} - {action_result.get('metric', 'info')}): {action_result['result']}"
        
        elif action_result["type"] == "competitor_analysis":
            observation = f"Competitor analysis ({action_result.get('aspect', 'general')}): {action_result['result']}"
        
        elif action_result["type"] == "error":
            observation = f"Error occurred: {action_result['result']}"
        
        else:
            observation = f"Action completed: {action_result.get('result', 'unknown result')}"
        
        # Check if we should continue or stop
        if context.current_iteration >= self.MAX_ITERATIONS - 1:
            context.is_complete = True
            observation += " (Max iterations reached)"
        
        return observation
    
    def solve(self, problem: str) -> str:
        """
        Main ReACT loop to solve the problem iteratively.
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
            print(f"Reasoning: {reasoning[:200]}...")
            
            # ACT: Execute action based on reasoning
            print("\n⚡ ACTING...")
            action_result = self.act(reasoning, context)
            context.actions_taken.append(str(action_result.get("type", "unknown")))
            print(f"Action: {action_result['type']}")
            
            # OBSERVE: Process results and update context
            print("\n👁️ OBSERVING...")
            observation = self.observe(action_result, context)
            context.observations.append(observation)
            print(f"Observation: {observation}")
            
            context.current_iteration += 1
        
        # Return final solution
        if context.final_answer:
            return context.final_answer
        else:
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