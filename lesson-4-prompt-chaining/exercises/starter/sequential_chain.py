"""
Lesson 4: Sequential Chain Implementation - Student Template

This module demonstrates advanced sequential prompt chaining techniques with
sophisticated context management and quality validation.

Learning Objectives:
- Implement robust chain step execution with error handling
- Manage context flow and state preservation across chain steps
- Build comprehensive quality validation framework
- Create production-ready sequential reasoning workflows

Complete TODOs 1-19 to implement comprehensive sequential chain functionality.

Author: [Your Name]
Date: [Current Date]
"""

import os
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from google import genai
from google.genai.types import GenerateContentConfig


class ChainStepType(Enum):
    """Types of chain steps for different reasoning approaches."""
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"
    RECOMMENDATION = "recommendation"


class ValidationLevel(Enum):
    """Quality validation levels for chain steps."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"


@dataclass
class ChainContext:
    """Context object managing state across chain steps."""
    initial_input: str
    step_history: Optional[List[Dict]] = None
    accumulated_insights: Optional[List[str]] = None
    current_focus: str = ""
    quality_scores: Optional[List[float]] = None
    token_usage: Optional[Dict[str, int]] = None
    execution_metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.step_history is None:
            self.step_history = []
        if self.accumulated_insights is None:
            self.accumulated_insights = []
        if self.quality_scores is None:
            self.quality_scores = []
        if self.token_usage is None:
            self.token_usage = {"total_input": 0, "total_output": 0}
        if self.execution_metadata is None:
            self.execution_metadata = {"start_time": time.time()}


@dataclass
class ChainStep:
    """Definition of a single step in the chain."""
    name: str
    step_type: ChainStepType
    prompt_template: str
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    max_retries: int = 3
    quality_threshold: float = 0.7
    timeout_seconds: int = 30
    context_requirements: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.context_requirements is None:
            self.context_requirements = []


@dataclass
class StepResult:
    """Result from executing a single chain step."""
    step_name: str
    content: str
    quality_score: float
    execution_time: float
    token_usage: Dict[str, int]
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0
    confidence_score: float = 0.0
    key_insights: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.key_insights is None:
            self.key_insights = []


@dataclass
class ChainResult:
    """Final result from complete chain execution."""
    success: bool
    final_output: str
    context: ChainContext
    overall_quality: float
    total_execution_time: float
    total_token_usage: Dict[str, int]
    step_results: List[StepResult]
    error_summary: Optional[str] = None


class SequentialChain:
    """
    Advanced sequential prompt chaining system with context management
    and quality validation.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize with Vertex AI client."""
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        self.model_name = "gemini-2.5-flash"
        self.debug_mode = os.getenv("CHAIN_DEBUG", "false").lower() == "true"
        
    def execute_chain(self, initial_prompt: str, steps: List[ChainStep]) -> ChainResult:
        """
        TODO 1: Implement robust chain step execution.
        
        Requirements:
        - Execute each step with proper error handling and retry logic
        - Preserve state and context across all chain steps
        - Validate quality at each step before proceeding
        - Return comprehensive ChainResult with all metrics
        
        Implementation areas:
        1. Chain initialization and context setup
        2. Step-by-step execution with error handling
        3. Quality validation and retry logic
        4. Context updates between steps
        5. Final result synthesis and metrics calculation
        
        Error handling should include:
        - Retry logic with exponential backoff
        - Quality threshold validation
        - Graceful failure recovery
        - Comprehensive error reporting
        
        Args:
            initial_prompt: The starting prompt for the chain
            steps: List of ChainStep objects to execute sequentially
            
        Returns:
            ChainResult with execution results and metrics
        """
        # TODO 1: Implement chain step execution
        #
        # Step 1: Initialize chain execution
        # start_time = time.time()
        # context = ChainContext(initial_input=initial_prompt)
        # step_results = []
        
        # Step 2: Execute each step in sequence
        # for i, step in enumerate(steps):
        #     # Execute step with current context
        #     step_result = self._execute_step_with_retry(step, context)
        #     step_results.append(step_result)
        #     
        #     # Check if step failed
        #     if not step_result.success:
        #         # Return failed chain result
        #         
        #     # Validate step quality
        #     if step_result.quality_score < step.quality_threshold:
        #         # Attempt quality recovery or fail
        #         
        #     # Update context for next step
        #     context = self._update_context_flow(context, step_result)
        
        # Step 3: Calculate final metrics and return result
        # overall_quality = statistics.mean([result.quality_score for result in step_results])
        # total_execution_time = time.time() - start_time
        # final_output = self._synthesize_final_output(step_results, context)
        # 
        # return ChainResult(...)
        
        return ChainResult(
            success=False,
            final_output="",
            context=ChainContext(initial_prompt),
            overall_quality=0.0,
            total_execution_time=0.0,
            total_token_usage={},
            step_results=[]
        )
    
    def _execute_step_with_retry(self, step: ChainStep, context: ChainContext) -> StepResult:
        """Execute single step with retry logic."""
        # This helper method is provided to guide your implementation
        for attempt in range(step.max_retries + 1):
            try:
                result = self._execute_single_step(step, context, attempt)
                if result.success:
                    return result
            except Exception as e:
                if attempt == step.max_retries:
                    return StepResult(
                        step_name=step.name,
                        content="",
                        quality_score=0.0,
                        execution_time=0.0,
                        token_usage={"input_tokens": 0, "output_tokens": 0},
                        success=False,
                        error_message=str(e),
                        retry_count=attempt
                    )
        
        return StepResult(
            step_name=step.name,
            content="",
            quality_score=0.0,
            execution_time=0.0,
            token_usage={"input_tokens": 0, "output_tokens": 0},
            success=False,
            error_message="Max retries exceeded",
            retry_count=step.max_retries
        )
    
    def _execute_single_step(self, step: ChainStep, context: ChainContext, attempt: int) -> StepResult:
        """Execute a single step of the chain."""
        start_time = time.time()
        
        # Create step-specific prompt with context
        full_prompt = self._create_step_prompt(step, context, attempt)
        
        # Configure generation parameters
        config = GenerateContentConfig(
            temperature=0.1 if attempt == 0 else 0.3,  # Increase creativity on retries
            max_output_tokens=2000,
            top_p=0.9,
            top_k=40
        )
        
        # Generate response
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=config
        )
        
        execution_time = time.time() - start_time

        # Extract token usage
        usage_metadata = response.usage_metadata
        token_usage: Dict[str, int] = {
            "input_tokens": int(usage_metadata.prompt_token_count) if (usage_metadata and usage_metadata.prompt_token_count is not None) else 0,
            "output_tokens": int(usage_metadata.candidates_token_count) if (usage_metadata and usage_metadata.candidates_token_count is not None) else 0
        }

        # Validate response quality
        response_text = response.text or ""
        quality_score = self._validate_step_result(response_text, step, context)

        # Extract key insights
        key_insights = self._extract_key_insights(response_text, step.step_type)

        return StepResult(
            step_name=step.name,
            content=response_text,
            quality_score=quality_score,
            execution_time=execution_time,
            token_usage=token_usage,
            success=True,
            retry_count=attempt,
            confidence_score=min(quality_score + 0.1, 1.0),
            key_insights=key_insights
        )
    
    def _create_step_prompt(self, step: ChainStep, context: ChainContext, attempt: int) -> str:
        """
        TODO 2: Create step-specific prompt with managed context flow.
        
        Requirements:
        - Build intelligent context summary for current step
        - Include relevant information from previous steps
        - Manage token usage by selective context inclusion
        - Add retry-specific instructions when needed
        
        Context management should include:
        1. Initial input preservation
        2. Key insights from previous steps
        3. Step-specific requirements
        4. Quality indicators and targets
        5. Retry-specific enhancements
        
        Args:
            step: Current step being executed
            context: Chain context with history
            attempt: Retry attempt number (0 for first attempt)
            
        Returns:
            Complete prompt string for step execution
        """
        # TODO 2: Implement context flow management
        #
        # Step 1: Build context summary
        # context_summary = self._build_context_summary(context, step)
        
        # Step 2: Add retry instructions if needed
        # retry_instructions = ""
        # if attempt > 0:
        #     retry_instructions = f"RETRY ATTEMPT {attempt + 1}: ..."
        
        # Step 3: Create full prompt with context
        # full_prompt = f"""{step.prompt_template}
        # 
        # {retry_instructions}
        # 
        # CONTEXT FROM PREVIOUS STEPS:
        # {context_summary}
        # 
        # CURRENT TASK FOCUS: {step.name}
        # Expected Step Type: {step.step_type.value}
        # 
        # Please provide your {step.step_type.value} response:"""
        #
        # return full_prompt

        return ""  # Replace with your implementation
    
    def _build_context_summary(self, context: ChainContext, current_step: ChainStep) -> str:
        """Build intelligent context summary for current step."""
        # This helper method is provided to guide your implementation
        if not context.step_history:
            return f"Initial Input: {context.initial_input}"
        
        relevant_insights = []
        relevant_insights.append(f"Business Scenario: {context.initial_input}")
        
        if context.accumulated_insights:
            relevant_insights.append("Previous Key Insights:")
            for insight in context.accumulated_insights[-3:]:  # Last 3 insights
                relevant_insights.append(f"- {insight}")
        
        if current_step.context_requirements:
            relevant_insights.append("\nSpecific Context Requirements:")
            for requirement in current_step.context_requirements:
                relevant_insights.append(f"- {requirement}")
        
        return "\n".join(relevant_insights)
    
    def _update_context_flow(self, context: ChainContext, step_result: StepResult) -> ChainContext:
        """
        TODO 2: Update context flow with new step result.
        
        Requirements:
        - Add step to history with relevant metadata
        - Update accumulated insights
        - Track quality and performance metrics
        - Maintain token usage statistics
        
        Args:
            context: Current chain context
            step_result: Result from completed step
            
        Returns:
            Updated ChainContext object
        """
        # TODO 2: Implement context flow updates
        #
        # Step 1: Add step to history
        # context.step_history.append({
        #     "step_name": step_result.step_name,
        #     "quality_score": step_result.quality_score,
        #     "execution_time": step_result.execution_time,
        #     "key_insights": step_result.key_insights,
        #     "content_summary": step_result.content[:200] + "..."
        # })
        
        # Step 2: Update accumulated insights
        # context.accumulated_insights.extend(step_result.key_insights)
        
        # Step 3: Update quality tracking
        # context.quality_scores.append(step_result.quality_score)
        
        # Step 4: Update token usage
        # context.token_usage["total_input"] += step_result.token_usage["input_tokens"]
        # context.token_usage["total_output"] += step_result.token_usage["output_tokens"]
        
        # Step 5: Update current focus
        # context.current_focus = f"Completed {step_result.step_name} with quality {step_result.quality_score:.2f}"

        # return context

        return context  # Replace with your implementation
    
    def _validate_step_result(self, content: str, step: ChainStep, context: ChainContext) -> float:
        """
        TODO 3: Implement comprehensive chain quality validation.
        
        Requirements:
        - Multi-dimensional quality assessment
        - Step-type specific validation criteria
        - Context-aware quality scoring
        - Validation level adaptability
        
        Quality dimensions to assess:
        1. Content length and completeness (25% of score)
        2. Structure and organization (25% of score)
        3. Step-type specific quality (25% of score)
        4. Context relevance and integration (25% of score)
        
        Validation should adapt based on:
        - Step type requirements
        - Validation level (basic/standard/strict)
        - Context completeness
        - Previous step quality
        
        Args:
            content: Generated content to validate
            step: Step configuration with requirements
            context: Chain context for relevance assessment
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        # TODO 3: Implement comprehensive quality validation
        #
        # Step 1: Calculate base metrics
        # word_count = len(content.split())
        # quality_score = 0.0
        
        # Step 2: Content length validation (25% of score)
        # if word_count >= 100:
        #     quality_score += 0.25
        # elif word_count >= 50:
        #     quality_score += 0.15
        
        # Step 3: Structure and organization (25% of score)
        # structure_score = self._assess_content_structure(content, step.step_type)
        # quality_score += structure_score * 0.25
        
        # Step 4: Step-type specific validation (25% of score)
        # type_score = self._assess_step_type_quality(content, step.step_type)
        # quality_score += type_score * 0.25
        
        # Step 5: Context relevance (25% of score)
        # relevance_score = self._assess_context_relevance(content, context)
        # quality_score += relevance_score * 0.25
        
        # Step 6: Apply validation level adjustments
        # if step.validation_level == ValidationLevel.STRICT:
        #     quality_score *= 0.9  # Higher standards
        # elif step.validation_level == ValidationLevel.BASIC:
        #     quality_score = min(quality_score * 1.1, 1.0)  # More lenient
        
        # return min(quality_score, 1.0)

        return 0.0  # Replace with your implementation

    def _assess_content_structure(self, content: str, step_type: ChainStepType) -> float:
        """Assess content structure and organization."""
        # TODO: Implement structure assessment
        # Check for: logical flow indicators, paragraphs, lists, organization
        return 0.0

    def _assess_step_type_quality(self, content: str, step_type: ChainStepType) -> float:
        """Assess quality based on step type requirements."""
        # TODO: Implement step-type specific quality assessment
        # Different criteria for analysis, synthesis, evaluation, recommendation
        return 0.0

    def _assess_context_relevance(self, content: str, context: ChainContext) -> float:
        """Assess how well content relates to the provided context."""
        # TODO: Implement context relevance assessment
        # Check for integration with previous insights and scenario relevance
        return 0.0
    
    def _extract_key_insights(self, content: str, step_type: ChainStepType) -> List[str]:
        """Extract key insights from step content."""
        # This helper method is provided to guide your implementation
        insights = []
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        
        for sentence in sentences[:5]:  # Limit to first 5 sentences
            if step_type == ChainStepType.ANALYSIS and any(term in sentence.lower() for term in ['analysis', 'trend', 'data']):
                insights.append(f"Analysis: {sentence}")
            elif step_type == ChainStepType.SYNTHESIS and any(term in sentence.lower() for term in ['overall', 'comprehensive']):
                insights.append(f"Synthesis: {sentence}")
            elif step_type == ChainStepType.EVALUATION and any(term in sentence.lower() for term in ['evaluate', 'compare']):
                insights.append(f"Evaluation: {sentence}")
            elif step_type == ChainStepType.RECOMMENDATION and any(term in sentence.lower() for term in ['recommend', 'suggest']):
                insights.append(f"Recommendation: {sentence}")
        
        return insights[:3]  # Return top 3 insights
    
    def _synthesize_final_output(self, step_results: List[StepResult], context: ChainContext) -> str:
        """Synthesize final output from all step results."""
        synthesis = []
        
        synthesis.append("# Complete Chain Analysis")
        synthesis.append(f"Generated through {len(step_results)} sequential reasoning steps\n")
        
        for i, result in enumerate(step_results, 1):
            synthesis.append(f"## Step {i}: {result.step_name}")
            synthesis.append(f"Quality Score: {result.quality_score:.2f}")
            synthesis.append(f"{result.content}\n")
        
        # Add summary metrics
        synthesis.append("## Execution Summary")
        synthesis.append(f"- Overall Quality: {statistics.mean([r.quality_score for r in step_results]):.2f}")
        synthesis.append(f"- Total Steps: {len(step_results)}")
        token_usage = context.token_usage or {"total_input": 0, "total_output": 0}
        synthesis.append(f"- Total Tokens: {token_usage['total_input'] + token_usage['total_output']}")
        accumulated_insights = context.accumulated_insights or []
        synthesis.append(f"- Key Insights: {len(accumulated_insights)}")
        
        return "\n".join(synthesis)


def load_business_scenarios() -> List[Dict]:
    """Load business scenarios for testing."""
    return [
        {
            "company_name": "TechFlow Solutions",
            "industry": "Software Technology",
            "market_focus": "enterprise workflow automation",
            "strategic_question": "Should we expand into small business markets or focus on enterprise growth?",
            "additional_context": "Strong enterprise presence, limited SMB experience, considering platform simplification."
        },
        {
            "company_name": "GreenEnergy Corp",
            "industry": "Renewable Energy", 
            "market_focus": "commercial solar installations",
            "strategic_question": "How should we respond to increased competition in the commercial solar market?",
            "additional_context": "Market leader for 5 years, new competitors entering with lower prices."
        }
    ]


def create_sample_chain_steps() -> List[ChainStep]:
    """Create sample chain steps for business analysis."""
    return [
        ChainStep(
            name="Market Overview",
            step_type=ChainStepType.ANALYSIS,
            prompt_template="You are a senior business analyst. Provide a comprehensive market overview analysis.",
            validation_level=ValidationLevel.STANDARD,
            quality_threshold=0.75,
            context_requirements=["Market size", "Growth trends", "Key players"]
        ),
        ChainStep(
            name="Competitive Analysis", 
            step_type=ChainStepType.ANALYSIS,
            prompt_template="You are a market researcher. Analyze the competitive landscape in detail.",
            validation_level=ValidationLevel.STANDARD,
            quality_threshold=0.75,
            context_requirements=["Competitor analysis", "Market positioning", "Differentiation"]
        ),
        ChainStep(
            name="Risk Assessment",
            step_type=ChainStepType.EVALUATION,
            prompt_template="You are a strategic consultant. Evaluate potential risks and challenges.",
            validation_level=ValidationLevel.STANDARD,
            quality_threshold=0.75,
            context_requirements=["Risk identification", "Impact assessment", "Mitigation strategies"]
        ),
        ChainStep(
            name="Strategic Recommendations",
            step_type=ChainStepType.RECOMMENDATION,
            prompt_template="You are a strategic consultant. Provide specific strategic recommendations.",
            validation_level=ValidationLevel.STRICT,
            quality_threshold=0.8,
            context_requirements=["Action plan", "Implementation timeline", "Success metrics"]
        )
    ]


def run_comprehensive_test():
    """Demonstrate complete sequential chain workflow."""
    PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id")
    if PROJECT_ID == "your-project-id":
        print("❌ Please set PROJECT_ID environment variable")
        return
    
    # Check if TODOs are implemented
    chain = SequentialChain(PROJECT_ID)
    scenarios = load_business_scenarios()
    test_scenario = scenarios[0]
    
    scenario_prompt = f"""
Business Scenario:
Company: {test_scenario['company_name']}
Industry: {test_scenario['industry']}
Market Focus: {test_scenario['market_focus']}
Strategic Question: {test_scenario['strategic_question']}
Context: {test_scenario['additional_context']}

Please analyze this business scenario comprehensively.
"""
    
    steps = create_sample_chain_steps()
    
    print("🔗 Testing Sequential Chain Implementation")
    print("=" * 60)
    
    # Test TODO 1
    try:
        result = chain.execute_chain(scenario_prompt, steps)
        if result is None:
            print("❌ TODO 1 not implemented: execute_chain")
            return
        print("✅ TODO 1: Chain step execution - IMPLEMENTED")
    except Exception as e:
        print(f"❌ TODO 1: Chain execution error - {e}")
        return
    
    # Test TODO 2 
    try:
        context = ChainContext(initial_input="Test")
        step = steps[0]
        prompt = chain._create_step_prompt(step, context, 0)
        if prompt is None:
            print("❌ TODO 2 not implemented: context flow management")
            return
        print("✅ TODO 2: Context flow management - IMPLEMENTED")
    except Exception as e:
        print(f"❌ TODO 2: Context flow error - {e}")
        return
    
    # Test TODO 3
    try:
        quality = chain._validate_step_result("Test content", steps[0], ChainContext(initial_input="Test"))
        if quality is None:
            print("❌ TODO 3 not implemented: quality validation")
            return
        print("✅ TODO 3: Chain quality validation - IMPLEMENTED")
    except Exception as e:
        print(f"❌ TODO 3: Quality validation error - {e}")
        return
    
    print("\n🎯 All TODOs implemented successfully!")
    print("🚀 Ready for conditional_chain.py!")


if __name__ == "__main__":
    run_comprehensive_test()