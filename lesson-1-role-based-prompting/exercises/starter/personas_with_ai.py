"""
Lesson 1: Role-Based Prompting with Vertex AI - Student Template

Learning Objectives:
- Design effective business personas for AI agents
- Test personas with Vertex AI Gemini API
- Compare persona effectiveness using real AI responses
- Understand prompt engineering best practices

Complete TODOs 1-10:
- TODOs 1-8: Create three business personas
- TODO 4: Test personas with Vertex AI
- TODO 5: Compare persona effectiveness

Author: [Your Name]
Date: [Current Date]
"""

import os
import json
import time
from typing import Dict, List, Tuple
from google import genai
from google.genai.types import GenerateContentConfig


# TODO 1: Design Business Analyst Persona
# Create a comprehensive business analyst persona that includes:
# - Professional background (15+ years experience)
# - Expertise in quantitative market analysis
# - Data-driven communication style
# - Specific analytical frameworks (TAM/SAM/SOM, market sizing, trend analysis)
# 
# Your persona should help the agent:
# 1. Analyze market opportunities with specific metrics
# 2. Provide data-driven insights with clear reasoning
# 3. Use professional business terminology
# 4. Structure analysis in logical, methodical way
# 
# Example structure:
# Role: You are a Senior Business Analyst with X years of experience...
# Expertise: Your specialization includes [specific areas]...
# Communication Style: [how they communicate]...
# Analytical Approach: [frameworks and methods they use]...

BUSINESS_ANALYST_PERSONA = """
[YOUR TODO 1 IMPLEMENTATION HERE]

Replace this placeholder with your business analyst persona design.
Make sure to include:
- Role and experience level (15+ years)
- Expertise in quantitative analysis
- Data-driven communication style
- Analytical frameworks (TAM/SAM/SOM, CAGR, etc.)
- Professional business terminology
"""


# TODO 2: Design Market Researcher Persona  
# Create a market research specialist persona focusing on:
# - Competitive intelligence and industry analysis
# - Strategic positioning assessment
# - Market dynamics understanding
# - Competitive landscape mapping
#
# This persona should excel at:
# 1. Competitive analysis and positioning
# 2. Industry research and trend identification
# 3. Market share dynamics assessment
# 4. Barrier analysis and competitive threats
#
# Include frameworks like Porter's Five Forces when relevant

MARKET_RESEARCHER_PERSONA = """
[YOUR TODO 2 IMPLEMENTATION HERE]

Replace this placeholder with your market researcher persona design.
Focus on:
- Competitive intelligence expertise
- Industry analysis capabilities
- Strategic positioning frameworks (Porter's Five Forces)
- Market dynamics understanding
- Competitive landscape mapping
"""


# TODO 3: Design Strategic Consultant Persona
# Create a strategic consultant persona specializing in:
# - Risk assessment and strategic planning
# - Implementable business recommendations
# - Business rationale and ROI considerations
# - Strategic options analysis
#
# This persona should provide:
# 1. Comprehensive risk evaluation
# 2. Strategic alternatives assessment
# 3. Implementation roadmaps
# 4. Success metrics and KPIs
# 5. Clear business rationale for recommendations

STRATEGIC_CONSULTANT_PERSONA = """
[YOUR TODO 3 IMPLEMENTATION HERE]

Replace this placeholder with your strategic consultant persona design.
Emphasize:
- Strategic planning expertise
- Risk assessment capabilities (risk matrices)
- Implementation focus with timelines
- ROI and success metrics
- Action-oriented recommendations
"""


class PersonaAITester:
    """
    Tests persona effectiveness using Vertex AI Gemini.
    This class demonstrates how to integrate personas with real AI.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize Vertex AI client."""
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        self.model_name = "gemini-2.5-flash"
        
        self.personas = {
            "business_analyst": BUSINESS_ANALYST_PERSONA,
            "market_researcher": MARKET_RESEARCHER_PERSONA,
            "strategic_consultant": STRATEGIC_CONSULTANT_PERSONA
        }
    
    def test_persona_with_scenario(self, persona_key: str, scenario: Dict) -> Dict:
        """
        TODO 4: Test persona with Vertex AI Gemini.
        
        Requirements:
        - Combine persona prompt with business scenario
        - Call Vertex AI Gemini to generate response
        - Analyze response quality and persona alignment
        - Return structured results with metrics
        
        Steps to implement:
        1. Check if persona_key exists in self.personas
        2. Get the persona text
        3. Create a comprehensive prompt combining:
           - The persona description
           - The business scenario details
           - Clear instructions for analysis
        4. Call Gemini API using self.client.models.generate_content()
        5. Measure generation time
        6. Analyze response quality using _analyze_response_quality()
        7. Return structured result dictionary
        
        Expected return format:
        {
            "persona": persona_key,
            "scenario": scenario name,
            "response": AI response text,
            "generation_time": time in seconds,
            "token_count": approximate tokens,
            "quality_analysis": quality metrics dict
        }
        
        Args:
            persona_key: Which persona to test ("business_analyst", etc.)
            scenario: Business scenario dictionary with company details
            
        Returns:
            Dictionary with test results and metrics
        """
        # TODO 4: Implement persona testing with Vertex AI
        # 
        # Step 1: Validate persona exists
        # if persona_key not in self.personas:
        #     return {"error": f"Unknown persona: {persona_key}"}
        
        # Step 2: Get persona and create comprehensive prompt
        # persona = self.personas[persona_key]
        # full_prompt = f"""
        # {persona}
        # 
        # Business Scenario:
        # Company: {scenario['company_name']}
        # Industry: {scenario['industry']}
        # Market Focus: {scenario['market_focus']}
        # Strategic Question: {scenario['strategic_question']}
        # Context: {scenario['additional_context']}
        # 
        # Based on your expertise as described above, provide your analysis...
        # """
        
        # Step 3: Call Gemini API
        # Use: self.client.models.generate_content()
        # Model: self.model_name
        # Config: GenerateContentConfig(temperature=0.7, max_output_tokens=800)
        
        # Step 4: Analyze response and return results
        
        pass  # Replace with your implementation
    
    def _analyze_response_quality(self, response: str, persona_key: str) -> Dict:
        """
        Analyze the quality of AI response for persona alignment.
        
        This method is provided to help you focus on the main TODOs.
        """
        # Persona-specific keywords to check for
        expected_elements = {
            "business_analyst": [
                "market size", "tam", "sam", "som", "growth rate", "metrics", 
                "analysis", "data", "quantitative", "market share"
            ],
            "market_researcher": [
                "competitive", "porter", "competitors", "market dynamics", 
                "positioning", "barriers", "competitive advantage", "market structure"
            ],
            "strategic_consultant": [
                "risk", "strategy", "implementation", "roi", "recommendations", 
                "strategic", "business rationale", "timeline", "metrics"
            ]
        }
        
        response_lower = response.lower()
        expected_keywords = expected_elements.get(persona_key, [])
        
        # Calculate keyword coverage
        keyword_matches = sum(1 for keyword in expected_keywords if keyword in response_lower)
        keyword_coverage = keyword_matches / len(expected_keywords) if expected_keywords else 0
        
        # Check for business frameworks
        frameworks = [
            "tam", "sam", "som", "porter", "five forces", "swot", "pestle",
            "roi", "npv", "irr", "cagr", "market share", "competitive advantage"
        ]
        framework_mentions = sum(1 for framework in frameworks if framework in response_lower)
        
        # Check for structured thinking
        structure_indicators = [
            "first", "second", "third", "additionally", "furthermore", 
            "therefore", "however", "consequently", "in conclusion"
        ]
        structure_score = sum(1 for indicator in structure_indicators if indicator in response_lower)
        
        # Overall quality score
        quality_score = min(1.0, (
            keyword_coverage * 0.4 +
            min(framework_mentions / 3, 1.0) * 0.3 +
            min(structure_score / 3, 1.0) * 0.2 +
            (1.0 if len(response.split()) > 100 else 0.5) * 0.1
        ))
        
        return {
            "keyword_coverage": keyword_coverage,
            "framework_mentions": framework_mentions,
            "structure_score": structure_score,
            "response_length": len(response.split()),
            "quality_score": quality_score,
            "persona_alignment": "High" if quality_score > 0.7 else "Medium" if quality_score > 0.5 else "Low"
        }
    
    def compare_personas(self, scenario: Dict) -> Dict:
        """
        TODO 5: Compare all personas on same scenario.
        
        Requirements:
        - Test all three personas with the same business scenario
        - Display comparison results with quality metrics
        - Identify the best performing persona
        - Show detailed analysis
        
        Steps to implement:
        1. Create results dictionary to store all persona results
        2. Loop through all personas in self.personas.keys()
        3. For each persona, call test_persona_with_scenario()
        4. Display progress and results as you go
        5. Find the best performing persona based on quality_score
        6. Print summary with best persona highlighted
        7. Return complete results dictionary
        
        Args:
            scenario: Business scenario to test all personas against
            
        Returns:
            Dictionary with results for all personas
        """
        # TODO 5: Implement persona comparison
        # 
        # Step 1: Initialize results and print header
        # results = {}
        # print(f"\n{'='*60}")
        # print(f"PERSONA COMPARISON: {scenario['name']}")
        # print(f"{'='*60}")
        
        # Step 2: Test each persona
        # for persona_key in self.personas.keys():
        #     print(f"\n🧠 Testing {persona_key.replace('_', ' ').title()}...")
        #     result = self.test_persona_with_scenario(persona_key, scenario)
        #     results[persona_key] = result
        #     
        #     # Show immediate feedback
        #     if "error" in result:
        #         print(f"   ❌ Error: {result['error']}")
        #     else:
        #         quality = result['quality_analysis']
        #         print(f"   ✅ Quality Score: {quality['quality_score']:.2f}")
        
        # Step 3: Find and announce best persona
        # valid_results = {k: v for k, v in results.items() if "error" not in v}
        # if valid_results:
        #     best_persona = max(valid_results.keys(), 
        #                      key=lambda k: valid_results[k]['quality_analysis']['quality_score'])
        #     print(f"\n🏆 BEST PERFORMING PERSONA: {best_persona.replace('_', ' ').title()}")
        
        # Step 4: Return results
        # return results
        
        pass  # Replace with your implementation


def load_test_scenarios() -> List[Dict]:
    """Load test scenarios from JSON file."""
    try:
        with open('scenarios.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback scenario if file not found
        return [
            {
                "name": "EcoTech Solar Expansion",
                "company_name": "EcoTech Solutions",
                "industry": "Clean Technology", 
                "market_focus": "residential solar energy systems",
                "strategic_question": "Should we expand into commercial solar markets, or focus on residential customer acquisition?",
                "additional_context": "Currently serving residential customers in California and Texas with 15% market share. Strong technical capabilities but limited sales resources."
            }
        ]


def run_persona_ai_demonstration():
    """Demonstrate persona testing with Vertex AI."""
    
    # Check for project ID
    PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id")
    if PROJECT_ID == "your-project-id":
        print("❌ Please set PROJECT_ID environment variable")
        print("   export PROJECT_ID=your-gcp-project-id")
        return
    
    # Check if TODOs are implemented
    if "[YOUR TODO 1" in BUSINESS_ANALYST_PERSONA:
        print("❌ TODO 1 not implemented: Create Business Analyst persona")
        return
    
    if "[YOUR TODO 2" in MARKET_RESEARCHER_PERSONA:
        print("❌ TODO 2 not implemented: Create Market Researcher persona")
        return
    
    if "[YOUR TODO 3" in STRATEGIC_CONSULTANT_PERSONA:
        print("❌ TODO 3 not implemented: Create Strategic Consultant persona")
        return
    
    print("🚀 PERSONA AI TESTING DEMONSTRATION")
    print("="*60)
    
    # Initialize AI tester
    tester = PersonaAITester(PROJECT_ID)
    
    # Load test scenarios
    scenarios = load_test_scenarios()
    
    # Test personas if TODOs 4-10 are implemented
    test_scenario = scenarios[0]
    
    # Check if TODO 4 is implemented
    test_result = tester.test_persona_with_scenario("business_analyst", test_scenario)
    if test_result is None:
        print("❌ TODO 4 not implemented: Test persona with Vertex AI")
        return
    
    # Check if TODO 5 is implemented
    comparison_results = tester.compare_personas(test_scenario)
    if comparison_results is None:
        print("❌ TODO 5 not implemented: Compare personas")
        return
    
    print(f"\n✅ All TODOs implemented successfully!")
    print("🎯 Personas are working with Vertex AI Gemini!")


# Backward compatibility - keep original function for existing tests
def validate_persona(persona_text: str) -> dict:
    """Original validation function for compatibility."""
    if "[YOUR TODO" in persona_text or len(persona_text.strip()) < 50:
        return {"valid": False, "score": 0.0, "feedback": "Not implemented"}
    
    checks = {
        "sufficient_length": len(persona_text.split()) >= 50,
        "role_definition": "Role:" in persona_text,
        "business_terminology": any(term in persona_text.lower() for term in 
                                  ["market", "strategic", "analysis", "competitive"])
    }
    
    score = sum(checks.values()) / len(checks)
    return {
        "valid": score >= 0.8,
        "score": score,
        "feedback": "Good persona" if score >= 0.8 else "Needs improvement"
    }


if __name__ == "__main__":
    run_persona_ai_demonstration()