"""
Lesson 4: Business Intelligence Chain Agent - Student Template

This module demonstrates a complete production-ready BI agent using advanced
prompt chaining, integrating personas from Lesson 1, optimization from Lesson 3,
and sophisticated chaining techniques from Lesson 4.

Learning Objectives:
- Build complete BI report generation pipeline
- Integrate cross-lesson components seamlessly
- Implement advanced error recovery mechanisms
- Create production-ready monitoring and analytics

Complete TODOs 7-24 to implement comprehensive BI chain agent functionality.

Author: [Your Name]
Date: [Current Date]
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

# Add lesson paths for cross-lesson integration
sys.path.append("../../lesson-1-role-based-prompting/exercises/solution")
sys.path.append("../../lesson-3-prompt-optimization/exercises/solution")

try:
    from personas_with_ai import PersonaManager
    LESSON_1_AVAILABLE = True
except ImportError:
    print("⚠️  Lesson 1 personas not available - using fallback personas")
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
        self.persona_manager = None
        self.prompt_optimizer = None
        
        if LESSON_1_AVAILABLE:
            try:
                self.persona_manager = PersonaManager(project_id)
                if self.debug_mode:
                    print("✅ Lesson 1 PersonaManager loaded")
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️  PersonaManager initialization failed: {e}")
        
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
        TODO 7: Generate complete 4-section BI report with chaining.
        
        Requirements:
        - Implement complete 4-section pipeline with context chaining
        - Add quality gates between each section
        - Include performance monitoring and cost tracking
        - Integrate cross-lesson components (personas, optimization)
        
        Pipeline sections to implement:
        1. Market Overview (Business Analyst persona)
        2. Competitive Analysis (Market Researcher persona)
        3. Risk Assessment (Strategic Consultant persona)  
        4. Strategic Recommendations (Strategic Consultant persona)
        
        Implementation requirements:
        - Use conditional chain execution with branching
        - Apply optimized prompts from Lesson 3 if available
        - Integrate personas from Lesson 1 if available
        - Track comprehensive metrics throughout
        - Generate executive summary from all sections
        
        Quality gates should validate:
        - Section completion and quality thresholds
        - Context preservation across sections
        - Token usage optimization
        - Overall report coherence
        
        Args:
            scenario: BusinessScenario with company and strategic context
            
        Returns:
            BIReport with all sections, metrics, and success status
        """
        # TODO 7: Implement complete BI report generation
        #
        # Step 1: Initialize generation metrics and context
        # start_time = time.time()
        # generation_metrics = {
        #     "total_token_usage": {"input_tokens": 0, "output_tokens": 0},
        #     "section_quality_scores": {},
        #     "error_count": 0,
        #     "recovery_attempts": 0,
        #     "optimization_applied": False
        # }
        
        # Step 2: Create optimized chain steps for BI pipeline
        # chain_steps = self._create_bi_chain_steps(scenario)
        
        # Step 3: Create initial scenario prompt
        # initial_prompt = self._create_scenario_prompt(scenario)
        
        # Step 4: Execute conditional chain with branching
        # try:
        #     chain_result = self.execute_conditional_chain(
        #         initial_prompt, 
        #         chain_steps, 
        #         enable_branching=True
        #     )
        #     
        #     if chain_result.success:
        #         # Process successful execution
        #         sections = self._extract_report_sections(chain_result)
        #         executive_summary = self._generate_executive_summary(sections, scenario)
        #         final_metrics = self._calculate_final_metrics(chain_result, generation_metrics, start_time)
        #         
        #         return BIReport(...)
        #     else:
        #         # Handle chain failure
        #         return self._handle_complete_chain_failure(...)
        # 
        # except Exception as e:
        #     return self._create_failure_report(...)
        
        pass  # Replace with your implementation
    
    def _create_bi_chain_steps(self, scenario: BusinessScenario) -> List[ChainStep]:
        """Create optimized chain steps for BI report generation."""
        # This helper method is provided to guide your implementation
        steps = []
        prompts = self._get_optimized_prompts()
        
        # Market Overview Step
        steps.append(ChainStep(
            name="Market Overview",
            step_type=ChainStepType.ANALYSIS,
            prompt_template=prompts["business_analyst"],
            validation_level=ValidationLevel.STANDARD,
            quality_threshold=self.quality_threshold,
            max_retries=self.max_section_retries,
            context_requirements=["Market size", "Growth trends", "Key players"]
        ))
        
        # Add other steps...
        
        return steps
    
    def _get_optimized_prompts(self) -> Dict[str, str]:
        """Get optimized prompts from cross-lesson integration."""
        # TODO: Implement cross-lesson integration
        # Try to get optimized prompts from Lesson 3 and personas from Lesson 1
        return self.fallback_prompts
    
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
        # TODO: Implement section extraction
        pass
    
    def _generate_executive_summary(self, sections: Dict[BIReportSection, str], 
                                   scenario: BusinessScenario) -> str:
        """Generate executive summary from report sections."""
        # TODO: Implement executive summary generation
        pass
    
    def _calculate_final_metrics(self, chain_result: ChainResult, 
                                generation_metrics: Dict, start_time: float) -> BIReportMetrics:
        """Calculate comprehensive BI report metrics."""
        # TODO: Implement metrics calculation
        pass
    
    def handle_chain_failure(self, failed_step: ChainStep, error: Exception, 
                           context: ChainContext) -> Tuple[bool, Optional[StepResult]]:
        """
        TODO 8: Implement advanced error recovery system.
        
        Requirements:
        - Implement intelligent retry logic with parameter adjustment
        - Create multiple fallback strategies for different error types
        - Support partial recovery for continued execution
        - Provide comprehensive failure analysis and logging
        
        Recovery strategies to implement:
        1. Retry with adjusted parameters (for transient failures)
        2. Fallback prompt usage (for quality/validation issues)
        3. Simplified analysis (for token/length constraints)
        4. Partial recovery (for context/integration issues)
        5. Graceful degradation (for complete failures)
        
        Error analysis should consider:
        - Error type and message analysis
        - Previous retry attempts
        - Context state and history
        - Step importance and dependencies
        - Alternative execution paths
        
        Recovery logic should include:
        - Failure type classification
        - Strategy selection based on error analysis
        - Parameter adjustment for retries
        - Alternative prompt selection
        - Partial content generation
        - Graceful failure handling
        
        Args:
            failed_step: ChainStep that failed
            error: Exception that caused the failure
            context: Current chain context
            
        Returns:
            Tuple of (recovery_success, recovered_result)
        """
        # TODO 8: Implement advanced error recovery
        #
        # Step 1: Analyze failure and determine recovery strategy
        # recovery_strategy = self._determine_recovery_strategy(failed_step, error, context)
        
        # Step 2: Execute appropriate recovery strategy
        # if recovery_strategy == ErrorRecoveryStrategy.RETRY_WITH_PARAMETERS:
        #     return self._retry_with_adjusted_parameters(failed_step, context)
        # 
        # elif recovery_strategy == ErrorRecoveryStrategy.FALLBACK_PROMPT:
        #     return self._use_fallback_prompt(failed_step, context)
        # 
        # elif recovery_strategy == ErrorRecoveryStrategy.SIMPLIFIED_ANALYSIS:
        #     return self._execute_simplified_analysis(failed_step, context)
        # 
        # elif recovery_strategy == ErrorRecoveryStrategy.PARTIAL_RECOVERY:
        #     return self._attempt_partial_recovery(failed_step, context)
        # 
        # else:  # GRACEFUL_DEGRADATION
        #     return self._graceful_degradation(failed_step, context)
        
        pass  # Replace with your implementation
    
    def _determine_recovery_strategy(self, failed_step: ChainStep, error: Exception, 
                                   context: ChainContext) -> ErrorRecoveryStrategy:
        """Determine optimal recovery strategy based on failure analysis."""
        # This helper method is provided to guide your implementation
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
        # TODO: Implement parameter-adjusted retry
        pass
    
    def _use_fallback_prompt(self, failed_step: ChainStep, 
                           context: ChainContext) -> Tuple[bool, Optional[StepResult]]:
        """Use simplified fallback prompt for quality issues."""
        # TODO: Implement fallback prompt strategy
        pass
    
    def _execute_simplified_analysis(self, failed_step: ChainStep, 
                                   context: ChainContext) -> Tuple[bool, Optional[StepResult]]:
        """Execute simplified analysis for token/length issues."""
        # TODO: Implement simplified analysis
        pass
    
    def _attempt_partial_recovery(self, failed_step: ChainStep, 
                                 context: ChainContext) -> Tuple[bool, Optional[StepResult]]:
        """Attempt to recover with partial functionality."""
        # TODO: Implement partial recovery
        pass
    
    def _graceful_degradation(self, failed_step: ChainStep, 
                            context: ChainContext) -> Tuple[bool, Optional[StepResult]]:
        """Graceful degradation with informative placeholder."""
        # TODO: Implement graceful degradation
        pass
    
    def _handle_complete_chain_failure(self, scenario: BusinessScenario, 
                                     chain_result: ChainResult, 
                                     generation_metrics: Dict, start_time: float) -> BIReport:
        """Handle complete chain failure with recovery attempts."""
        # TODO: Implement complete failure handling
        pass
    
    def _create_failure_report(self, scenario: BusinessScenario, error_message: str,
                             generation_metrics: Dict, start_time: float) -> BIReport:
        """Create failure report for complete system failure."""
        # TODO: Implement failure report creation
        pass


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
    
    # Check if TODOs are implemented
    bi_agent = BusinessIntelligenceChain(PROJECT_ID)
    scenarios = load_test_scenarios()
    test_scenario = scenarios[0]
    
    print("🏢 Testing Business Intelligence Chain Agent")
    print("=" * 60)
    
    print(f"📋 Testing scenario: {test_scenario.company_name}")
    print(f"Strategic Question: {test_scenario.strategic_question}")
    
    # Test TODO 7
    try:
        report = bi_agent.generate_complete_report(test_scenario)
        if report is None:
            print("❌ TODO 7 not implemented: generate_complete_report")
            return
        print("✅ TODO 7: Complete BI report chain - IMPLEMENTED")
    except Exception as e:
        print(f"❌ TODO 7: BI report generation error - {e}")
        return
    
    # Test TODO 8
    try:
        test_step = ChainStep("test", ChainStepType.ANALYSIS, "test")
        test_context = ChainContext(initial_input="test")
        test_error = Exception("test error")
        
        recovery_result = bi_agent.handle_chain_failure(test_step, test_error, test_context)
        if recovery_result is None:
            print("❌ TODO 8 not implemented: handle_chain_failure")
            return
        print("✅ TODO 8: Advanced error recovery - IMPLEMENTED")
    except Exception as e:
        print(f"❌ TODO 8: Error recovery test error - {e}")
        return
    
    print("\n🎯 All TODOs implemented successfully!")
    print("🚀 Ready for comprehensive testing!")


if __name__ == "__main__":
    run_bi_chain_test()