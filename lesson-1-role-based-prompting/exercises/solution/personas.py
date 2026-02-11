"""
Lesson 1: Role-Based Prompting - Complete Solution

This file contains the complete implementation of three business personas
for the Business Intelligence Agent. This serves as the reference solution
for students completing TODOs 6, 7, and 8.

Author: Noble Ackerson (Udacity)
Date: 2025
"""


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


# Enhanced validation function for solution reference
def validate_persona_advanced(persona_text: str, persona_name: str) -> dict:
    """
    Advanced validation function for persona quality assessment.
    
    This version includes more sophisticated checks for persona quality
    including business terminology, framework usage, and professional depth.
    """
    if "[YOUR TODO" in persona_text or len(persona_text.strip()) < 100:
        return {
            "valid": False,
            "score": 0.0,
            "feedback": f"{persona_name} not properly implemented"
        }
    
    # Advanced validation checks
    checks = {
        "sufficient_length": len(persona_text.split()) >= 150,  # More comprehensive
        "role_definition": "Role:" in persona_text,
        "expertise_section": "Expertise:" in persona_text or "specialization" in persona_text.lower(),
        "communication_style": "Communication Style:" in persona_text or "style" in persona_text.lower(),
        "analytical_approach": "Analytical Approach:" in persona_text or "framework" in persona_text.lower(),
        "task_context": "Task Context:" in persona_text or "when" in persona_text.lower(),
        "business_terminology": any(term in persona_text.lower() for term in 
                                  ["market", "strategic", "analysis", "competitive", "roi", "metrics"]),
        "frameworks_mentioned": any(framework in persona_text for framework in 
                                  ["Porter", "TAM", "SAM", "SOM", "analysis", "assessment"]),
        "professional_depth": any(indicator in persona_text.lower() for indicator in 
                                ["years", "experience", "expertise", "specialization", "competencies"])
    }
    
    score = sum(checks.values()) / len(checks)
    
    # Detailed feedback
    feedback = []
    if not checks["sufficient_length"]:
        feedback.append("Persona needs more detail (aim for 150+ words)")
    if not checks["role_definition"]:
        feedback.append("Clearly define the role with 'Role:' section")
    if not checks["expertise_section"]:
        feedback.append("Include expertise/specialization section")
    if not checks["communication_style"]:
        feedback.append("Define communication style")
    if not checks["analytical_approach"]:
        feedback.append("Specify analytical approaches and frameworks")
    if not checks["business_terminology"]:
        feedback.append("Include more business terminology")
    if not checks["frameworks_mentioned"]:
        feedback.append("Reference specific business frameworks")
    if not checks["professional_depth"]:
        feedback.append("Add more professional depth and experience details")
    
    return {
        "valid": score >= 0.9,  # Higher standard for solution
        "score": score,
        "feedback": "; ".join(feedback) if feedback else "Excellent persona design!",
        "checks": checks,
        "grade": _calculate_grade(score)
    }


def _calculate_grade(score: float) -> str:
    """Calculate letter grade based on score"""
    if score >= 0.95:
        return "A+"
    elif score >= 0.9:
        return "A"
    elif score >= 0.85:
        return "B+"
    elif score >= 0.8:
        return "B"
    elif score >= 0.75:
        return "C+"
    elif score >= 0.7:
        return "C"
    else:
        return "F"


def analyze_persona_quality():
    """Comprehensive analysis of all personas"""
    personas = {
        "Business Analyst": BUSINESS_ANALYST_PERSONA,
        "Market Researcher": MARKET_RESEARCHER_PERSONA, 
        "Strategic Consultant": STRATEGIC_CONSULTANT_PERSONA
    }
    
    print("=" * 70)
    print("COMPREHENSIVE PERSONA QUALITY ANALYSIS")
    print("=" * 70)
    
    total_score = 0
    all_valid = True
    
    for name, persona in personas.items():
        result = validate_persona_advanced(persona, name)
        total_score += result["score"]
        
        if not result["valid"]:
            all_valid = False
        
        print(f"\n{name.upper()}:")
        print(f"  Score: {result['score']:.2f} | Grade: {result['grade']} | {'✅ PASS' if result['valid'] else '❌ FAIL'}")
        print(f"  Word Count: {len(persona.split())} words")
        print(f"  Feedback: {result['feedback']}")
        
        # Show detailed check results
        passed_checks = [k for k, v in result['checks'].items() if v]
        failed_checks = [k for k, v in result['checks'].items() if not v]
        
        if passed_checks:
            print(f"  ✅ Passed: {', '.join(passed_checks)}")
        if failed_checks:
            print(f"  ❌ Failed: {', '.join(failed_checks)}")
    
    avg_score = total_score / len(personas)
    overall_grade = _calculate_grade(avg_score)
    
    print("\n" + "=" * 70)
    print(f"OVERALL ASSESSMENT:")
    print(f"Average Score: {avg_score:.2f} | Grade: {overall_grade}")
    print(f"Status: {'✅ ALL PERSONAS MEET STANDARDS' if all_valid else '❌ SOME PERSONAS NEED IMPROVEMENT'}")
    print("=" * 70)
    
    return {
        "overall_score": avg_score,
        "overall_grade": overall_grade,
        "all_valid": all_valid,
        "individual_results": {name: validate_persona_advanced(persona, name) 
                             for name, persona in personas.items()}
    }


if __name__ == "__main__":
    # Run comprehensive analysis when executed directly
    analyze_persona_quality()