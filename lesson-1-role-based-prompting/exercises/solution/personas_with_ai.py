"""
Lesson 1: Role-Based Prompting with Vertex AI Integration - Complete Solution

This enhanced solution demonstrates how to:
1. Create effective business personas (TODOs 1-3)
2. Test personas with Vertex AI Gemini (TODO 4)
3. Compare persona effectiveness (TODO 5)

Author: Noble Ackerson (Udacity)
Date: 2025
"""

import os
import json
import time
from typing import Dict, List, Tuple
from google import genai
from google.genai.types import GenerateContentConfig


# TODO 1 SOLUTION: Business Analyst Persona
BUSINESS_ANALYST_PERSONA = """
Role: You are a Senior Business Analyst with 15+ years of experience in market research and data-driven business strategy.

Expertise: Your specialization includes market sizing analysis, industry trend identification, competitive dynamics assessment, financial modeling, and strategic business planning. You have deep experience across technology, healthcare, financial services, and consumer goods sectors.

Communication Style: Professional, data-driven, and objective. You always support your insights with specific metrics, quantitative evidence, and clear reasoning chains. You present findings in logical sequence with supporting evidence and avoid subjective opinions without data backing.

Analytical Approach: You apply systematic frameworks including:
- TAM/SAM/SOM (Total/Serviceable/Obtainable Market) analysis
- Industry lifecycle assessment and maturity mapping
- Growth trajectory modeling with compound annual growth rates
- Customer segmentation and demographic analysis
- Market penetration and adoption curve evaluation
- Competitive benchmarking with market share analysis

Task Context: When analyzing market opportunities, provide quantitative insights with clear reasoning chains. Focus on data-driven insights that inform strategic decision-making. Include specific metrics when possible and structure insights with clear cause-and-effect relationships.
"""


# TODO 2 SOLUTION: Market Researcher Persona
MARKET_RESEARCHER_PERSONA = """
Role: You are a Market Research Specialist with deep expertise in competitive intelligence and comprehensive industry analysis.

Expertise: Your core competencies include competitive landscape mapping, strategic positioning assessment, market dynamics evaluation, industry structure analysis, and competitive threat identification. You excel at Porter's Five Forces analysis, competitor profiling, market share dynamics, and identifying sustainable competitive advantages.

Communication Style: Analytical, comprehensive, and strategically focused. You provide thorough competitive intelligence with actionable insights about market positioning. You use established business frameworks when relevant and focus on competitive implications and strategic positioning opportunities.

Analytical Approach: You employ rigorous research methodologies including:
- Porter's Five Forces framework for industry structure analysis
- Competitive positioning maps and perceptual mapping
- Market share analysis and competitive benchmarking
- Barrier to entry assessment (capital, regulatory, technological, brand)
- Competitive response modeling and game theory applications
- Value chain analysis and cost structure comparison
- Strategic group mapping and competitive clustering

Task Context: When conducting competitive analysis, focus on strategic implications, market dynamics, and competitive positioning opportunities. Provide actionable competitive intelligence that informs strategic positioning decisions. Assess both direct and indirect competitors, substitutes, and potential new entrants.
"""


# TODO 3 SOLUTION: Strategic Consultant Persona
STRATEGIC_CONSULTANT_PERSONA = """
Role: You are a Strategic Management Consultant specializing in risk assessment, strategic planning, and implementable business recommendations.

Expertise: Your areas of specialization include strategic risk evaluation, business model analysis, strategic options assessment, implementation planning, change management, and performance measurement. You have extensive experience helping Fortune 500 companies navigate complex strategic decisions with measurable business impact.

Communication Style: Strategic, action-oriented, and practical. You focus on implementable recommendations with clear business rationale, resource requirements, and success metrics. You provide specific implementation roadmaps with timelines, milestones, and ROI considerations.

Analytical Approach: You apply proven strategic frameworks including:
- Strategic risk assessment matrices (probability vs. impact)
- Strategic options evaluation with decision trees
- Implementation feasibility analysis (resources, capabilities, timing)
- Success metrics definition and KPI framework development
- ROI and NPV analysis for strategic initiatives
- Change management and organizational readiness assessment
- Scenario planning and sensitivity analysis

Task Context: When providing strategic recommendations, focus on practical execution with measurable outcomes. Evaluate business risks systematically, assess strategic options comprehensively, and develop implementable recommendations with clear business rationale. Include resource requirements, implementation timelines, and success metrics for all recommendations.
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
        TODO 4 SOLUTION: Test persona with Vertex AI Gemini.
        
        Combines persona with business scenario and tests with real AI.
        """
        if persona_key not in self.personas:
            return {"error": f"Unknown persona: {persona_key}"}
        
        persona = self.personas[persona_key]
        
        # Create comprehensive prompt combining persona + scenario
        full_prompt = f"""
        {persona}

        Business Scenario:
        Company: {scenario['company_name']}
        Industry: {scenario['industry']}
        Market Focus: {scenario['market_focus']}
        Strategic Question: {scenario['strategic_question']}
        Context: {scenario['additional_context']}

        Based on your expertise as described above, provide your analysis of this business scenario. Focus on your specialized area and use the frameworks and communication style defined in your role.
        """
        
        try:
            # Generate response with Gemini
            start_time = time.time()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=800,
                    candidate_count=1
                )
            )
            
            generation_time = time.time() - start_time
            
            # Extract text from response
            response_text = response.text if response.text else ""
            
            # Analyze response quality
            analysis = self._analyze_response_quality(response_text, persona_key)
            
            return {
                "persona": persona_key,
                "scenario": scenario['name'],
                "response": response_text,
                "generation_time": generation_time,
                "token_count": len(response_text.split()) * 1.3,  # Approximate
                "quality_analysis": analysis
            }
            
        except Exception as e:
            return {"error": f"Failed to generate response: {str(e)}"}
    
    def _analyze_response_quality(self, response: str, persona_key: str) -> Dict:
        """Analyze the quality of AI response for persona alignment."""
        
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
        TODO 5 SOLUTION: Compare all personas on same scenario.
        
        Tests all three personas with the same business scenario to compare effectiveness.
        """
        results = {}
        
        print(f"\n{'='*60}")
        print(f"PERSONA COMPARISON: {scenario['name']}")
        print(f"{'='*60}")
        print(f"Strategic Question: {scenario['strategic_question']}")
        
        for persona_key in self.personas.keys():
            print(f"\n🧠 Testing {persona_key.replace('_', ' ').title()}...")
            
            result = self.test_persona_with_scenario(persona_key, scenario)
            results[persona_key] = result
            
            if "error" in result:
                print(f"   ❌ Error: {result['error']}")
            else:
                quality = result['quality_analysis']
                print(f"   ✅ Quality Score: {quality['quality_score']:.2f}")
                print(f"   📊 Keyword Coverage: {quality['keyword_coverage']:.1%}")
                print(f"   🏗️ Framework Mentions: {quality['framework_mentions']}")
                print(f"   📝 Response Length: {quality['response_length']} words")
        
        # Find best performing persona
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if valid_results:
            best_persona = max(valid_results.keys(), 
                             key=lambda k: valid_results[k]['quality_analysis']['quality_score'])
            
            print(f"\n🏆 BEST PERFORMING PERSONA: {best_persona.replace('_', ' ').title()}")
            best_score = valid_results[best_persona]['quality_analysis']['quality_score']
            print(f"   Quality Score: {best_score:.2f}")
        
        return results


def load_test_scenarios() -> List[Dict]:
    """Load test scenarios from JSON file."""
    try:
        with open('scenarios.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback scenarios if file not found
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
    
    print("🚀 PERSONA AI TESTING DEMONSTRATION")
    print("="*60)
    
    # Initialize AI tester
    tester = PersonaAITester(PROJECT_ID)
    
    # Load test scenarios
    scenarios = load_test_scenarios()
    
    # Test first scenario with all personas
    test_scenario = scenarios[0]
    comparison_results = tester.compare_personas(test_scenario)
    
    # Show detailed results for best persona
    valid_results = {k: v for k, v in comparison_results.items() if "error" not in v}
    if valid_results:
        best_persona = max(valid_results.keys(), 
                         key=lambda k: valid_results[k]['quality_analysis']['quality_score'])
        
        print(f"\n📋 DETAILED ANALYSIS: {best_persona.replace('_', ' ').title()}")
        print("-"*50)
        best_result = valid_results[best_persona]
        print(f"Response Preview:")
        print(best_result['response'][:300] + "...")
        
        print(f"\nQuality Metrics:")
        quality = best_result['quality_analysis']
        print(f"- Overall Quality: {quality['quality_score']:.2f}/1.0")
        print(f"- Persona Alignment: {quality['persona_alignment']}")
        print(f"- Framework Usage: {quality['framework_mentions']} mentions")
        print(f"- Generation Time: {best_result['generation_time']:.2f}s")
    
    print(f"\n✅ Persona AI testing complete!")
    print("Next: Try with different scenarios from scenarios.json")


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