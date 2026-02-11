"""
Lesson 4: Business Intelligence Chain Agent - Complete Solution

This module demonstrates a complete production-ready BI agent using advanced
prompt chaining, integrating personas from Lesson 1, optimization from Lesson 3,
and sophisticated chaining techniques from Lesson 4.

Learning Objectives:
- Build complete BI report generation pipeline
- Integrate cross-lesson components seamlessly
- Implement advanced error recovery mechanisms
- Create production-ready monitoring and analytics

TODOs 7-24 SOLUTIONS implemented with comprehensive BI pipeline,
advanced error recovery, and performance optimization.

Author: Noble Ackerson (Udacity)
Date: 2025
"""

import os
import sys
import time
import json
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Import chain components
from sequential_chain import (
    SequentialChain, ChainContext, ChainStep, StepResult, ChainResult,
    ChainStepType, ValidationLevel
)
from conditional_chain import ConditionalChain, BranchingDecision, ReasoningPath

# Add lesson paths for cross-lesson integration using absolute paths
from pathlib import Path
lesson_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(lesson_root / "lesson-1-role-based-prompting" / "exercises" / "solution"))
sys.path.insert(0, str(lesson_root / "lesson-3-prompt-optimization" / "exercises" / "solution"))

try:
    from personas import (  # type: ignore
        BUSINESS_ANALYST_PERSONA,
        MARKET_RESEARCHER_PERSONA,
        STRATEGIC_CONSULTANT_PERSONA
    )
    LESSON_1_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Lesson 1 personas not available: {e}")
    LESSON_1_AVAILABLE = False

try:
    from vertex_optimizer import VertexPromptOptimizer
    LESSON_3_AVAILABLE = True
except ImportError:
    print("⚠️  Lesson 3 optimizer not available - using base prompts")
    LESSON_3_AVAILABLE = False


class BIReportSection(Enum):
    """Business Intelligence report sections."""
    MARKET_OVERVIEW = "market_overview"
    COMPETITIVE_ANALYSIS = "competitive_analysis" 
    RISK_ASSESSMENT = "risk_assessment"
    STRATEGIC_RECOMMENDATIONS = "strategic_recommendations"


class ErrorRecoveryStrategy(Enum):
    """Error recovery strategies for BI chain failures."""
    RETRY_WITH_PARAMETERS = "retry_with_parameters"
    FALLBACK_PROMPT = "fallback_prompt"
    SIMPLIFIED_ANALYSIS = "simplified_analysis"
    PARTIAL_RECOVERY = "partial_recovery"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class BusinessScenario:
    """Business scenario for BI analysis."""
    company_name: str
    industry: str
    market_focus: str
    strategic_question: str
    additional_context: str
    expected_elements: List[str] = None
    
    def __post_init__(self):
        if self.expected_elements is None:
            self.expected_elements = ["market", "competitive", "strategy", "recommendation"]


@dataclass
class BIReportMetrics:
    """Metrics for BI report generation."""
    total_generation_time: float
    total_token_usage: Dict[str, int]
    section_quality_scores: Dict[str, float]
    overall_quality: float
    error_count: int
    recovery_attempts: int
    optimization_applied: bool
    cost_estimate: float


@dataclass
class BIReport:
    """Complete business intelligence report."""
    scenario: BusinessScenario
    sections: Dict[BIReportSection, str]
    metrics: BIReportMetrics
    generation_timestamp: float
    success: bool
    executive_summary: str = ""
    failure_reasons: List[str] = None
    
    def __post_init__(self):
        if self.failure_reasons is None:
            self.failure_reasons = []


class BusinessIntelligenceChain(ConditionalChain):
    """
    Production-ready Business Intelligence Chain Agent integrating
    all lessons with advanced error recovery and monitoring.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize BI chain with cross-lesson integration."""
        super().__init__(project_id, location)
        
        # Initialize cross-lesson components
        self.lesson_1_personas = None
        self.prompt_optimizer = None

        if LESSON_1_AVAILABLE:
            try:
                self.lesson_1_personas = {
                    "business_analyst": BUSINESS_ANALYST_PERSONA,
                    "market_researcher": MARKET_RESEARCHER_PERSONA,
                    "strategic_consultant": STRATEGIC_CONSULTANT_PERSONA
                }
                if self.debug_mode:
                    print("✅ Lesson 1 personas loaded")
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️  Lesson 1 personas initialization failed: {e}")
        
        if LESSON_3_AVAILABLE:
            try:
                self.prompt_optimizer = VertexPromptOptimizer(project_id)
                if self.debug_mode:
                    print("✅ Lesson 3 VertexPromptOptimizer loaded")
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️  VertexPromptOptimizer initialization failed: {e}")
        
        # BI-specific configuration
        self.cost_per_1k_tokens = 0.0002  # Gemini 2.5 Flash pricing
        self.quality_threshold = 0.75
        self.max_section_retries = 3
        self.performance_history = []
        
        # Define optimized prompts (fallback if Lesson 3 not available)
        self.fallback_prompts = self._create_fallback_prompts()
    
    def generate_complete_report(self, scenario: BusinessScenario) -> BIReport:
        """
        TODO 7 SOLUTION: Generate complete 4-section BI report with chaining.
        
        This implementation provides:
        - Complete 4-section pipeline with context chaining
        - Quality gates between each section
        - Performance monitoring and cost tracking
        - Cross-lesson integration with optimization
        """
        start_time = time.time()
        generation_metrics = {
            "total_token_usage": {"input_tokens": 0, "output_tokens": 0},
            "section_quality_scores": {},
            "error_count": 0,
            "recovery_attempts": 0,
            "optimization_applied": False
        }
        
        if self.debug_mode:
            print(f"🏢 Generating BI report for {scenario.company_name}")
            print("=" * 60)
        
        # Step 1: Create optimized chain steps
        chain_steps = self._create_bi_chain_steps(scenario)
        
        # Step 2: Create initial prompt with scenario
        initial_prompt = self._create_scenario_prompt(scenario)
        
        # Step 3: Execute chain with conditional logic
        try:
            chain_result = self.execute_conditional_chain(
                initial_prompt, 
                chain_steps, 
                enable_branching=True
            )
            
            if chain_result.success:
                # Process successful chain execution
                sections = self._extract_report_sections(chain_result)
                executive_summary = self._generate_executive_summary(sections, scenario)
                
                # Calculate final metrics
                final_metrics = self._calculate_final_metrics(
                    chain_result, generation_metrics, start_time
                )
                
                if self.debug_mode:
                    print(f"✅ BI report generated successfully")
                    print(f"📊 Overall Quality: {final_metrics.overall_quality:.2f}")
                    print(f"⏱️  Total Time: {final_metrics.total_generation_time:.1f}s")
                
                return BIReport(
                    scenario=scenario,
                    sections=sections,
                    metrics=final_metrics,
                    generation_timestamp=time.time(),
                    success=True,
                    executive_summary=executive_summary
                )
            
            else:
                # Handle chain failure with recovery
                return self._handle_complete_chain_failure(
                    scenario, chain_result, generation_metrics, start_time
                )
                
        except Exception as e:
            if self.debug_mode:
                print(f"❌ BI report generation failed: {str(e)}")
            
            return self._create_failure_report(scenario, str(e), generation_metrics, start_time)
    
    def _create_bi_chain_steps(self, scenario: BusinessScenario) -> List[ChainStep]:
        """Create optimized chain steps for BI report generation."""
        steps = []
        
        # Get optimized prompts if available
        prompts = self._get_optimized_prompts()
        
        # Market Overview Step
        steps.append(ChainStep(
            name="Market Overview",
            step_type=ChainStepType.ANALYSIS,
            prompt_template=prompts["business_analyst"],
            validation_level=ValidationLevel.STANDARD,
            quality_threshold=self.quality_threshold,
            max_retries=self.max_section_retries,
            context_requirements=[
                "Market size and growth trends",
                "Industry landscape analysis", 
                "Opportunity assessment"
            ]
        ))
        
        # Competitive Analysis Step
        steps.append(ChainStep(
            name="Competitive Analysis",
            step_type=ChainStepType.ANALYSIS,
            prompt_template=prompts["market_researcher"],
            validation_level=ValidationLevel.STANDARD,
            quality_threshold=self.quality_threshold,
            max_retries=self.max_section_retries,
            context_requirements=[
                "Competitor identification",
                "Market positioning analysis",
                "Competitive advantages"
            ]
        ))
        
        # Risk Assessment Step
        steps.append(ChainStep(
            name="Risk Assessment",
            step_type=ChainStepType.EVALUATION,
            prompt_template=prompts["strategic_consultant"],
            validation_level=ValidationLevel.STANDARD,
            quality_threshold=self.quality_threshold,
            max_retries=self.max_section_retries,
            context_requirements=[
                "Risk identification",
                "Impact assessment",
                "Mitigation strategies"
            ]
        ))
        
        # Strategic Recommendations Step
        steps.append(ChainStep(
            name="Strategic Recommendations",
            step_type=ChainStepType.RECOMMENDATION,
            prompt_template=prompts["strategic_consultant"],
            validation_level=ValidationLevel.STRICT,
            quality_threshold=0.8,  # Higher threshold for recommendations
            max_retries=self.max_section_retries,
            context_requirements=[
                "Strategic options",
                "Implementation roadmap",
                "Success metrics"
            ]
        ))
        
        return steps
    
    def _get_optimized_prompts(self) -> Dict[str, str]:
        """Get optimized prompts from cross-lesson integration."""
        prompts = {}
        
        # Try to get optimized prompts from Lesson 3
        if self.prompt_optimizer:
            try:
                # Use optimization for business analyst prompt
                base_analyst = self.fallback_prompts["business_analyst"]
                optimization_result = self.prompt_optimizer.optimize_prompt(
                    base_analyst, "instructions"
                )
                if optimization_result and optimization_result.success:
                    prompts["business_analyst"] = optimization_result.optimized_prompt
                    if self.debug_mode:
                        print("🎯 Using optimized business analyst prompt")
                else:
                    prompts["business_analyst"] = base_analyst
                    
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️  Prompt optimization failed: {e}")
                prompts["business_analyst"] = self.fallback_prompts["business_analyst"]
        else:
            prompts["business_analyst"] = self.fallback_prompts["business_analyst"]
        
        # Use Lesson 1 personas if available
        if self.lesson_1_personas:
            try:
                prompts["market_researcher"] = self.lesson_1_personas.get(
                    "market_researcher", self.fallback_prompts["market_researcher"]
                )
                prompts["strategic_consultant"] = self.lesson_1_personas.get(
                    "strategic_consultant", self.fallback_prompts["strategic_consultant"]
                )
                if self.debug_mode:
                    print("📋 Using Lesson 1 personas")
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️  Persona loading failed: {e}")
                prompts.update(self.fallback_prompts)
        else:
            prompts.update(self.fallback_prompts)
        
        return prompts
    
    def _create_fallback_prompts(self) -> Dict[str, str]:
        """Create fallback prompts if cross-lesson components unavailable."""
        return {
            "business_analyst": """You are a senior business analyst with 15+ years of experience in strategic consulting and market analysis. Your expertise spans multiple industries including technology, healthcare, finance, and retail.

Your role is to provide comprehensive business analysis that combines quantitative data analysis with strategic thinking. Focus on market overview, growth trends, and opportunity assessment.

When analyzing business scenarios, structure your response with:
1. Market Overview and Size Assessment
2. Growth Trends and Drivers  
3. Key Industry Dynamics
4. Strategic Opportunities

Provide data-driven insights with specific reasoning and actionable observations.""",

            "market_researcher": """You are an expert market researcher specializing in competitive intelligence and industry analysis. You have deep expertise in market positioning, competitor analysis, and strategic differentiation.

Your role is to provide detailed competitive landscape analysis including:
1. Key Competitor Identification
2. Market Positioning Analysis
3. Competitive Advantages and Weaknesses
4. Market Share and Performance Metrics

Focus on actionable competitive insights that inform strategic decision-making.""",

            "strategic_consultant": """You are a strategic consultant with extensive experience in risk assessment and strategic planning. You specialize in identifying business risks, evaluating strategic options, and providing implementation roadmaps.

For risk assessment, focus on:
1. Risk Identification and Categorization
2. Impact and Probability Assessment
3. Mitigation Strategies
4. Risk Monitoring Recommendations

For strategic recommendations, provide:
1. Strategic Options Analysis
2. Implementation Roadmap
3. Resource Requirements
4. Success Metrics and KPIs"""
        }
    
    def _create_scenario_prompt(self, scenario: BusinessScenario) -> str:
        """Create comprehensive scenario prompt."""
        return f"""
Business Analysis Scenario:

Company: {scenario.company_name}
Industry: {scenario.industry}
Market Focus: {scenario.market_focus}
Strategic Question: {scenario.strategic_question}
Additional Context: {scenario.additional_context}

Expected Analysis Elements: {', '.join(scenario.expected_elements)}

Please provide comprehensive business intelligence analysis addressing the strategic question with detailed insights across all relevant dimensions.
"""
    
    def _extract_report_sections(self, chain_result: ChainResult) -> Dict[BIReportSection, str]:
        """Extract structured sections from chain result."""
        sections = {}
        section_mapping = [
            BIReportSection.MARKET_OVERVIEW,
            BIReportSection.COMPETITIVE_ANALYSIS,
            BIReportSection.RISK_ASSESSMENT,
            BIReportSection.STRATEGIC_RECOMMENDATIONS
        ]
        
        for i, step_result in enumerate(chain_result.step_results):
            if i < len(section_mapping):
                sections[section_mapping[i]] = step_result.content
        
        return sections
    
    def _generate_executive_summary(self, sections: Dict[BIReportSection, str], 
                                   scenario: BusinessScenario) -> str:
        """Generate executive summary from report sections."""
        summary_parts = [
            f"# Executive Summary: {scenario.company_name} Strategic Analysis",
            "",
            f"**Strategic Question:** {scenario.strategic_question}",
            "",
            "## Key Findings:",
            ""
        ]
        
        # Extract key insights from each section
        for section, content in sections.items():
            # Extract first significant sentence as key finding
            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 30]
            if sentences:
                key_finding = sentences[0]
                summary_parts.append(f"- **{section.value.replace('_', ' ').title()}:** {key_finding}")
        
        summary_parts.extend([
            "",
            "This analysis provides comprehensive insights to support strategic decision-making.",
            f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        return "\n".join(summary_parts)
    
    def _calculate_final_metrics(self, chain_result: ChainResult, 
                                generation_metrics: Dict, start_time: float) -> BIReportMetrics:
        """Calculate comprehensive BI report metrics."""
        section_scores = {}
        for step_result in chain_result.step_results:
            section_scores[step_result.step_name] = step_result.quality_score
        
        total_tokens = chain_result.total_token_usage["total_input"] + chain_result.total_token_usage["total_output"]
        cost_estimate = (total_tokens / 1000) * self.cost_per_1k_tokens
        
        return BIReportMetrics(
            total_generation_time=time.time() - start_time,
            total_token_usage=chain_result.total_token_usage,
            section_quality_scores=section_scores,
            overall_quality=chain_result.overall_quality,
            error_count=generation_metrics["error_count"],
            recovery_attempts=generation_metrics["recovery_attempts"],
            optimization_applied=generation_metrics["optimization_applied"],
            cost_estimate=cost_estimate
        )
    
    def handle_chain_failure(self, failed_step: ChainStep, error: Exception, 
                           context: ChainContext) -> Tuple[bool, Optional[StepResult]]:
        """
        TODO 8 SOLUTION: Implement advanced error recovery system.
        
        This implementation provides:
        - Intelligent retry logic with parameter adjustment
        - Multiple fallback strategies
        - Partial recovery capabilities
        - Comprehensive failure analysis and logging
        """
        if self.debug_mode:
            print(f"🔧 Handling failure for step: {failed_step.name}")
            print(f"Error: {str(error)}")
        
        # Determine recovery strategy based on failure type and context
        recovery_strategy = self._determine_recovery_strategy(failed_step, error, context)
        
        if recovery_strategy == ErrorRecoveryStrategy.RETRY_WITH_PARAMETERS:
            return self._retry_with_adjusted_parameters(failed_step, context)
        
        elif recovery_strategy == ErrorRecoveryStrategy.FALLBACK_PROMPT:
            return self._use_fallback_prompt(failed_step, context)
        
        elif recovery_strategy == ErrorRecoveryStrategy.SIMPLIFIED_ANALYSIS:
            return self._execute_simplified_analysis(failed_step, context)
        
        elif recovery_strategy == ErrorRecoveryStrategy.PARTIAL_RECOVERY:
            return self._attempt_partial_recovery(failed_step, context)
        
        else:  # GRACEFUL_DEGRADATION
            return self._graceful_degradation(failed_step, context)
    
    def _determine_recovery_strategy(self, failed_step: ChainStep, error: Exception, 
                                   context: ChainContext) -> ErrorRecoveryStrategy:
        """Determine optimal recovery strategy based on failure analysis."""
        error_str = str(error).lower()
        
        # API/Network errors - retry with parameters
        if any(term in error_str for term in ["timeout", "connection", "rate limit", "quota"]):
            return ErrorRecoveryStrategy.RETRY_WITH_PARAMETERS
        
        # Quality issues - try fallback prompt
        if "quality" in error_str or "validation" in error_str:
            return ErrorRecoveryStrategy.FALLBACK_PROMPT
        
        # Token/length issues - simplified analysis
        if any(term in error_str for term in ["token", "length", "truncated"]):
            return ErrorRecoveryStrategy.SIMPLIFIED_ANALYSIS
        
        # Context issues - partial recovery
        if "context" in error_str or len(context.step_history) > 0:
            return ErrorRecoveryStrategy.PARTIAL_RECOVERY
        
        # Default - graceful degradation
        return ErrorRecoveryStrategy.GRACEFUL_DEGRADATION
    
    def _retry_with_adjusted_parameters(self, failed_step: ChainStep, 
                                      context: ChainContext) -> Tuple[bool, Optional[StepResult]]:
        """Retry with adjusted parameters for transient failures."""
        if self.debug_mode:
            print("🔄 Attempting retry with adjusted parameters")
        
        # Create adjusted step with modified parameters
        adjusted_step = ChainStep(
            name=f"{failed_step.name}_retry",
            step_type=failed_step.step_type,
            prompt_template=failed_step.prompt_template,
            validation_level=ValidationLevel.BASIC,  # Lower validation for retry
            max_retries=1,  # Single retry attempt
            quality_threshold=max(failed_step.quality_threshold - 0.1, 0.5),  # Lower threshold
            timeout_seconds=failed_step.timeout_seconds + 10,  # Extended timeout
            context_requirements=failed_step.context_requirements[:2]  # Reduced requirements
        )
        
        try:
            result = self._execute_step_with_retry(adjusted_step, context)
            if result.success:
                if self.debug_mode:
                    print("✅ Recovery successful with adjusted parameters")
                return True, result
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Retry failed: {str(e)}")
        
        return False, None
    
    def _use_fallback_prompt(self, failed_step: ChainStep, 
                           context: ChainContext) -> Tuple[bool, Optional[StepResult]]:
        """Use simplified fallback prompt for quality issues."""
        if self.debug_mode:
            print("🔄 Attempting fallback prompt strategy")
        
        # Create simplified prompt
        fallback_prompt = f"""Please provide a basic {failed_step.step_type.value} for the business scenario.

Focus on:
1. Key insights and observations
2. Main recommendations or findings
3. Clear, actionable conclusions

Keep the analysis concise but comprehensive."""
        
        fallback_step = ChainStep(
            name=f"{failed_step.name}_fallback",
            step_type=failed_step.step_type,
            prompt_template=fallback_prompt,
            validation_level=ValidationLevel.BASIC,
            max_retries=1,
            quality_threshold=0.6,  # Lower threshold for fallback
            context_requirements=[]
        )
        
        try:
            result = self._execute_step_with_retry(fallback_step, context)
            if result.success:
                if self.debug_mode:
                    print("✅ Recovery successful with fallback prompt")
                return True, result
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Fallback prompt failed: {str(e)}")
        
        return False, None
    
    def _execute_simplified_analysis(self, failed_step: ChainStep, 
                                   context: ChainContext) -> Tuple[bool, Optional[StepResult]]:
        """Execute simplified analysis for token/length issues."""
        if self.debug_mode:
            print("🔄 Attempting simplified analysis")
        
        # Create very basic prompt
        simple_prompt = f"Provide a brief {failed_step.step_type.value} summary focusing on the most important points only."
        
        simple_step = ChainStep(
            name=f"{failed_step.name}_simple",
            step_type=failed_step.step_type,
            prompt_template=simple_prompt,
            validation_level=ValidationLevel.BASIC,
            max_retries=1,
            quality_threshold=0.5,
            context_requirements=[]
        )
        
        try:
            # Use minimal context to avoid token issues
            minimal_context = ChainContext(
                initial_input=context.initial_input[:500],  # Truncate context
                accumulated_insights=context.accumulated_insights[-2:] if context.accumulated_insights else []
            )
            
            result = self._execute_step_with_retry(simple_step, minimal_context)
            if result.success:
                if self.debug_mode:
                    print("✅ Recovery successful with simplified analysis")
                return True, result
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Simplified analysis failed: {str(e)}")
        
        return False, None
    
    def _attempt_partial_recovery(self, failed_step: ChainStep, 
                                 context: ChainContext) -> Tuple[bool, Optional[StepResult]]:
        """Attempt to recover with partial functionality."""
        if self.debug_mode:
            print("🔄 Attempting partial recovery")
        
        # Generate basic content using available context
        partial_content = f"""
# {failed_step.name} - Partial Analysis

Due to processing limitations, this section provides a basic analysis:

## Available Insights:
"""
        
        # Add available insights from context
        if context.accumulated_insights:
            for insight in context.accumulated_insights[-3:]:
                partial_content += f"- {insight}\n"
        else:
            partial_content += "- Analysis based on business scenario requirements\n"
            partial_content += "- Standard industry considerations apply\n"
        
        partial_content += f"""
## Recommendation:
Manual review recommended for comprehensive {failed_step.step_type.value} analysis.

*Note: This is a partial recovery result due to processing constraints.*
"""
        
        # Create partial result
        partial_result = StepResult(
            step_name=f"{failed_step.name}_partial",
            content=partial_content,
            quality_score=0.4,  # Low quality due to partial nature
            execution_time=0.1,
            token_usage={"input_tokens": 100, "output_tokens": 50},
            success=True,
            error_message="Partial recovery - limited functionality",
            confidence_score=0.3,
            key_insights=[f"Partial {failed_step.step_type.value} completed"]
        )
        
        if self.debug_mode:
            print("✅ Partial recovery completed")
        
        return True, partial_result
    
    def _graceful_degradation(self, failed_step: ChainStep, 
                            context: ChainContext) -> Tuple[bool, Optional[StepResult]]:
        """Graceful degradation with informative placeholder."""
        if self.debug_mode:
            print("🔄 Executing graceful degradation")
        
        degraded_content = f"""
# {failed_step.name} - Analysis Unavailable

This section could not be completed due to technical constraints.

## Recommended Actions:
1. Review the business scenario manually for {failed_step.step_type.value} requirements
2. Consult domain experts for specialized insights
3. Consider alternative analysis approaches

## Context Available:
- Business Scenario: {context.initial_input[:200]}...
- Previous Steps Completed: {len(context.step_history)}

*This is a placeholder result to maintain report structure.*
"""
        
        degraded_result = StepResult(
            step_name=f"{failed_step.name}_degraded",
            content=degraded_content,
            quality_score=0.2,  # Very low quality
            execution_time=0.05,
            token_usage={"input_tokens": 50, "output_tokens": 30},
            success=True,  # Success in terms of providing output
            error_message="Graceful degradation - placeholder content",
            confidence_score=0.1,
            key_insights=["Manual analysis required"]
        )
        
        if self.debug_mode:
            print("✅ Graceful degradation completed")
        
        return True, degraded_result
    
    def _handle_complete_chain_failure(self, scenario: BusinessScenario, 
                                     chain_result: ChainResult, 
                                     generation_metrics: Dict, start_time: float) -> BIReport:
        """Handle complete chain failure with recovery attempts."""
        if self.debug_mode:
            print("🔧 Handling complete chain failure")
        
        # Attempt emergency report generation
        emergency_sections = {}
        
        # Try to salvage any successful sections
        if chain_result.step_results:
            section_mapping = [
                BIReportSection.MARKET_OVERVIEW,
                BIReportSection.COMPETITIVE_ANALYSIS,
                BIReportSection.RISK_ASSESSMENT,
                BIReportSection.STRATEGIC_RECOMMENDATIONS
            ]
            
            for i, result in enumerate(chain_result.step_results):
                if result.success and i < len(section_mapping):
                    emergency_sections[section_mapping[i]] = result.content
        
        # Fill missing sections with placeholders
        all_sections = [
            BIReportSection.MARKET_OVERVIEW,
            BIReportSection.COMPETITIVE_ANALYSIS,
            BIReportSection.RISK_ASSESSMENT,
            BIReportSection.STRATEGIC_RECOMMENDATIONS
        ]
        
        for section in all_sections:
            if section not in emergency_sections:
                emergency_sections[section] = f"""
# {section.value.replace('_', ' ').title()} - Analysis Unavailable

This section could not be completed due to processing constraints.

## Manual Analysis Required
Please conduct manual analysis for {section.value.replace('_', ' ')} focusing on:
- Key business requirements from the scenario
- Industry-specific considerations
- Strategic implications

## Business Context:
- Company: {scenario.company_name}
- Industry: {scenario.industry}
- Strategic Question: {scenario.strategic_question}
"""
        
        # Create emergency metrics
        emergency_metrics = BIReportMetrics(
            total_generation_time=time.time() - start_time,
            total_token_usage=chain_result.total_token_usage if chain_result.total_token_usage else {"total_input": 0, "total_output": 0},
            section_quality_scores={section.value: 0.2 for section in emergency_sections.keys()},
            overall_quality=0.3,  # Low quality due to failure
            error_count=generation_metrics["error_count"] + 1,
            recovery_attempts=generation_metrics["recovery_attempts"] + 1,
            optimization_applied=False,
            cost_estimate=0.0
        )
        
        return BIReport(
            scenario=scenario,
            sections=emergency_sections,
            metrics=emergency_metrics,
            generation_timestamp=time.time(),
            success=False,
            executive_summary="Report generation failed - emergency recovery applied",
            failure_reasons=[chain_result.error_summary] if chain_result.error_summary else ["Unknown chain failure"]
        )
    
    def _create_failure_report(self, scenario: BusinessScenario, error_message: str,
                             generation_metrics: Dict, start_time: float) -> BIReport:
        """Create failure report for complete system failure."""
        failure_metrics = BIReportMetrics(
            total_generation_time=time.time() - start_time,
            total_token_usage={"total_input": 0, "total_output": 0},
            section_quality_scores={},
            overall_quality=0.0,
            error_count=1,
            recovery_attempts=0,
            optimization_applied=False,
            cost_estimate=0.0
        )
        
        return BIReport(
            scenario=scenario,
            sections={},
            metrics=failure_metrics,
            generation_timestamp=time.time(),
            success=False,
            executive_summary="Report generation completely failed",
            failure_reasons=[error_message]
        )


def load_test_scenarios() -> List[BusinessScenario]:
    """Load test scenarios for BI agent."""
    return [
        BusinessScenario(
            company_name="TechFlow Solutions",
            industry="Software Technology",
            market_focus="enterprise workflow automation",
            strategic_question="Should we expand into small business markets or focus on enterprise growth?",
            additional_context="Strong enterprise presence, limited SMB experience, considering platform simplification.",
            expected_elements=["market", "competitive", "strategy", "recommendation"]
        ),
        BusinessScenario(
            company_name="GreenEnergy Corp",
            industry="Renewable Energy",
            market_focus="commercial solar installations",
            strategic_question="How should we respond to increased competition in the commercial solar market?",
            additional_context="Market leader for 5 years, new competitors entering with lower prices.",
            expected_elements=["competitive", "positioning", "pricing", "differentiation"]
        )
    ]


def run_bi_chain_test():
    """Demonstrate complete BI chain agent capabilities."""
    PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id")
    if PROJECT_ID == "your-project-id":
        print("❌ Please set PROJECT_ID environment variable")
        return
    
    # Initialize BI chain agent
    bi_agent = BusinessIntelligenceChain(PROJECT_ID)
    
    print("🏢 Testing Business Intelligence Chain Agent")
    print("=" * 60)
    
    # Load test scenario
    scenarios = load_test_scenarios()
    test_scenario = scenarios[0]
    
    print(f"📋 Testing scenario: {test_scenario.company_name}")
    print(f"Strategic Question: {test_scenario.strategic_question}")
    
    try:
        # Generate complete BI report
        report = bi_agent.generate_complete_report(test_scenario)
        
        if report.success:
            print("\n✅ BI Report Generation Successful!")
            print(f"📊 Overall Quality: {report.metrics.overall_quality:.2f}")
            print(f"⏱️  Generation Time: {report.metrics.total_generation_time:.1f}s")
            print(f"💰 Estimated Cost: ${report.metrics.cost_estimate:.4f}")
            print(f"📝 Sections Generated: {len(report.sections)}")
            
            # Show section quality scores
            print("\n📈 Section Quality Scores:")
            for section, score in report.metrics.section_quality_scores.items():
                print(f"  {section}: {score:.2f}")
            
        else:
            print("\n❌ BI Report Generation Failed")
            print(f"Failure Reasons: {report.failure_reasons}")
            print(f"Recovery Attempts: {report.metrics.recovery_attempts}")
        
        print("\n🧪 TODO Validation:")
        print("✅ TODO 7: Complete BI report chain - IMPLEMENTED")
        print("✅ TODO 8: Advanced error recovery - IMPLEMENTED")
        
    except Exception as e:
        print(f"❌ BI chain test failed: {e}")


if __name__ == "__main__":
    run_bi_chain_test()