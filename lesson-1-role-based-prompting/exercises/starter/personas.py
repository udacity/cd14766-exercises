"""
Lesson 1: Role-Based Prompting - Student Template

Learning Objectives:
- Design effective business personas for AI agents
- Understand prompt engineering best practices
- Create role-specific communication styles and expertise areas

Complete TODOs 6, 7, and 8 to create three distinct business personas:
- Business Analyst (quantitative market analysis)
- Market Researcher (competitive intelligence)  
- Strategic Consultant (risk assessment and recommendations)

Instructions:
1. Each persona should be 15-20 lines with specific expertise
2. Include communication style guidelines
3. Define analytical frameworks they would use
4. Use professional business terminology
5. Test your personas using test_personas.py

Author: [Your Name]
Date: [Current Date]
"""


# TODO 1: Design Business Analyst Persona
# Create a comprehensive business analyst persona that includes:
# - Professional background (15+ years experience)
# - Expertise in quantitative market analysis
# - Data-driven communication style
# - Specific analytical frameworks (market sizing, trend analysis, growth projections)
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
- Role and experience level
- Specific expertise areas
- Communication style
- Analytical frameworks
- Professional terminology usage
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
- Strategic positioning frameworks
- Market dynamics understanding
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
- Risk assessment capabilities
- Implementation focus
- ROI and success metrics
"""


# Validation function for testing
def validate_persona(persona_text: str, persona_name: str) -> dict:
    """
    Validate a persona design for completeness and quality.
    
    Returns:
        dict: Validation results with scores and feedback
    """
    if "[YOUR TODO" in persona_text:
        return {
            "valid": False,
            "score": 0.0,
            "feedback": f"{persona_name} not implemented - contains placeholder text"
        }
    
    # Basic validation checks
    checks = {
        "length": len(persona_text.split()) >= 50,  # Minimum 50 words
        "role_defined": any(word in persona_text.lower() for word in ["role:", "you are"]),
        "experience": any(word in persona_text.lower() for word in ["experience", "years", "expertise"]),
        "communication": any(word in persona_text.lower() for word in ["style", "communication", "approach"]),
        "frameworks": any(word in persona_text.lower() for word in ["framework", "analysis", "method", "approach"])
    }
    
    score = sum(checks.values()) / len(checks)
    
    feedback = []
    if not checks["length"]:
        feedback.append("Persona too short - aim for 50+ words")
    if not checks["role_defined"]:
        feedback.append("Clearly define the role")
    if not checks["experience"]:
        feedback.append("Include experience level and expertise")
    if not checks["communication"]:
        feedback.append("Specify communication style")
    if not checks["frameworks"]:
        feedback.append("Include analytical frameworks or methods")
    
    return {
        "valid": score >= 0.8,
        "score": score,
        "feedback": "; ".join(feedback) if feedback else "Good persona design!",
        "checks": checks
    }


# Test function
def test_all_personas():
    """Test all three personas for completeness"""
    personas = {
        "Business Analyst": BUSINESS_ANALYST_PERSONA,
        "Market Researcher": MARKET_RESEARCHER_PERSONA, 
        "Strategic Consultant": STRATEGIC_CONSULTANT_PERSONA
    }
    
    results = {}
    total_score = 0
    
    print("=" * 60)
    print("PERSONA VALIDATION RESULTS")
    print("=" * 60)
    
    for name, persona in personas.items():
        result = validate_persona(persona, name)
        results[name] = result
        total_score += result["score"]
        
        status = "✅ PASS" if result["valid"] else "❌ FAIL"
        print(f"{name}: {status} (Score: {result['score']:.2f})")
        print(f"   Feedback: {result['feedback']}")
        print()
    
    avg_score = total_score / len(personas)
    overall_status = "✅ ALL COMPLETE" if avg_score >= 0.8 else "❌ NEEDS WORK"
    
    print(f"OVERALL: {overall_status} (Average Score: {avg_score:.2f})")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Run validation when script is executed directly
    test_all_personas()