"""
Lesson 4: Sequential Chain Implementation - Complete Solution

This module demonstrates advanced sequential prompt chaining techniques with
sophisticated context management and quality validation.

Learning Objectives:
- Implement robust chain step execution with error handling
- Manage context flow and state preservation across chain steps
- Build comprehensive quality validation framework
- Create production-ready sequential reasoning workflows

TODOs 1-19 SOLUTIONS implemented with comprehensive context management,
quality validation, and error recovery mechanisms.

Author: Noble Ackerson (Udacity)
Date: 2025
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
        TODO 1 SOLUTION: Implement robust chain step execution.
        
        This implementation provides:
        - Comprehensive error handling with retry logic
        - State preservation across all chain steps
        - Quality validation at each step
        - Performance monitoring and optimization
        """
        start_time = time.time()
        context = ChainContext(initial_input=initial_prompt)
        step_results = []
        
        if self.debug_mode:
            print(f"🔗 Starting chain execution with {len(steps)} steps")
        
        for i, step in enumerate(steps):
            if self.debug_mode:
                print(f"⚡ Executing step {i+1}/{len(steps)}: {step.name}")
            
            # Execute step with current context
            step_result = self._execute_step_with_retry(step, context)
            step_results.append(step_result)
            
            # Check if step failed
            if not step_result.success:
                error_msg = f"Chain failed at step '{step.name}': {step_result.error_message}"
                return ChainResult(
                    success=False,
                    final_output="",
                    context=context,
                    overall_quality=0.0,
                    total_execution_time=time.time() - start_time,
                    total_token_usage=context.token_usage or {"total_input": 0, "total_output": 0},
                    step_results=step_results,
                    error_summary=error_msg
                )
            
            # Validate step result quality
            if step_result.quality_score < step.quality_threshold:
                error_msg = f"Quality validation failed for step '{step.name}': {step_result.quality_score} < {step.quality_threshold}"
                if self.debug_mode:
                    print(f"❌ {error_msg}")
                
                # Attempt recovery if retries available
                if step_result.retry_count < step.max_retries:
                    if self.debug_mode:
                        print(f"🔄 Attempting quality recovery for step '{step.name}'")
                    
                    enhanced_step = self._enhance_step_for_quality(step, step_result)
                    recovery_result = self._execute_step_with_retry(enhanced_step, context)
                    
                    if recovery_result.success and recovery_result.quality_score >= step.quality_threshold:
                        step_result = recovery_result
                        step_results[-1] = step_result
                    else:
                        return ChainResult(
                            success=False,
                            final_output="",
                            context=context,
                            overall_quality=0.0,
                            total_execution_time=time.time() - start_time,
                            total_token_usage=context.token_usage or {"total_input": 0, "total_output": 0},
                            step_results=step_results,
                            error_summary=error_msg
                        )
            
            # Update context for next step
            context = self._update_context_flow(context, step_result)
            
            if self.debug_mode:
                print(f"✅ Step '{step.name}' completed - Quality: {step_result.quality_score:.3f}")
        
        # Calculate final metrics
        overall_quality = statistics.mean([result.quality_score for result in step_results])
        total_execution_time = time.time() - start_time
        
        # Generate final output
        final_output = self._synthesize_final_output(step_results, context)
        
        return ChainResult(
            success=True,
            final_output=final_output,
            context=context,
            overall_quality=overall_quality,
            total_execution_time=total_execution_time,
            total_token_usage=context.token_usage or {"total_input": 0, "total_output": 0},
            step_results=step_results
        )
    
    def _execute_step_with_retry(self, step: ChainStep, context: ChainContext) -> StepResult:
        """Execute single step with retry logic."""
        for attempt in range(step.max_retries + 1):
            try:
                result = self._execute_single_step(step, context, attempt)
                if result.success:
                    return result
                
                if self.debug_mode:
                    print(f"🔄 Step '{step.name}' attempt {attempt + 1} failed: {result.error_message}")
                
            except Exception as e:
                if self.debug_mode:
                    print(f"🔄 Step '{step.name}' attempt {attempt + 1} exception: {str(e)}")
                
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
            confidence_score=min(quality_score + 0.1, 1.0),  # Confidence slightly higher than quality
            key_insights=key_insights
        )
    
    def _create_step_prompt(self, step: ChainStep, context: ChainContext, attempt: int) -> str:
        """
        TODO 2 SOLUTION: Create step-specific prompt with managed context flow.
        
        This implementation provides:
        - Intelligent context selection and compression
        - Step-specific prompt formatting
        - Token-aware context management
        - Adaptive prompting based on attempt number
        """
        # Build context summary for this step
        context_summary = self._build_context_summary(context, step)
        
        # Add retry-specific instructions if needed
        retry_instructions = ""
        if attempt > 0:
            retry_instructions = f"""
RETRY ATTEMPT {attempt + 1}: The previous attempt may not have met quality standards. 
Please provide more detailed analysis with specific examples and concrete recommendations.
"""
        
        # Create the full prompt
        full_prompt = f"""{step.prompt_template}

{retry_instructions}

CONTEXT FROM PREVIOUS STEPS:
{context_summary}

CURRENT TASK FOCUS: {step.name}
Expected Step Type: {step.step_type.value}

INSTRUCTIONS:
1. Build upon the insights provided in the context above
2. Focus specifically on the {step.step_type.value} aspects
3. Provide detailed, actionable analysis
4. Structure your response clearly with specific sections
5. Include concrete examples and evidence where applicable

Please provide your {step.step_type.value} response:"""

        return full_prompt
    
    def _build_context_summary(self, context: ChainContext, current_step: ChainStep) -> str:
        """Build intelligent context summary for current step."""
        if not context.step_history:
            return f"Initial Input: {context.initial_input}"
        
        # Determine what context is relevant for this step type
        relevant_insights = []
        
        # Always include initial input
        relevant_insights.append(f"Business Scenario: {context.initial_input}")
        
        # Include key insights from previous steps
        if context.accumulated_insights:
            relevant_insights.append("Previous Key Insights:")
            for insight in context.accumulated_insights[-3:]:  # Last 3 insights to manage tokens
                relevant_insights.append(f"- {insight}")
        
        # Include step-specific context requirements
        if current_step.context_requirements:
            relevant_insights.append("\nSpecific Context Requirements:")
            for requirement in current_step.context_requirements:
                relevant_insights.append(f"- {requirement}")
        
        # Add quality indicators from previous steps
        if context.quality_scores:
            avg_quality = statistics.mean(context.quality_scores)
            relevant_insights.append(f"\nPrevious Analysis Quality: {avg_quality:.2f} (aim for ≥{current_step.quality_threshold})")
        
        return "\n".join(relevant_insights)
    
    def _update_context_flow(self, context: ChainContext, step_result: StepResult) -> ChainContext:
        """
        TODO 2 SOLUTION: Update context flow with new step result.
        
        This implementation provides:
        - Intelligent context accumulation
        - Token usage tracking
        - Quality score monitoring
        - Selective insight preservation
        """
        # Add step to history
        if context.step_history is not None:
            context.step_history.append({
                "step_name": step_result.step_name,
                "quality_score": step_result.quality_score,
                "execution_time": step_result.execution_time,
                "key_insights": step_result.key_insights,
                "content_summary": step_result.content[:200] + "..." if len(step_result.content) > 200 else step_result.content
            })

        # Update accumulated insights
        if context.accumulated_insights is not None and step_result.key_insights is not None:
            context.accumulated_insights.extend(step_result.key_insights)

        # Update quality tracking
        if context.quality_scores is not None:
            context.quality_scores.append(step_result.quality_score)

        # Update token usage
        if context.token_usage is not None:
            context.token_usage["total_input"] += step_result.token_usage["input_tokens"]
            context.token_usage["total_output"] += step_result.token_usage["output_tokens"]
        
        # Update current focus
        context.current_focus = f"Completed {step_result.step_name} with quality {step_result.quality_score:.2f}"
        
        return context
    
    def _validate_step_result(self, content: str, step: ChainStep, context: ChainContext) -> float:
        """
        TODO 3 SOLUTION: Comprehensive chain quality validation.
        
        This implementation provides:
        - Multi-dimensional quality assessment
        - Step-type specific validation
        - Context-aware quality scoring
        - Validation level adaptability
        """
        word_count = len(content.split())
        
        # Base quality score
        quality_score = 0.0
        
        # 1. Content length validation (25% of score)
        if word_count >= 100:
            quality_score += 0.25
        elif word_count >= 50:
            quality_score += 0.15
        
        # 2. Structure and organization (25% of score)
        structure_score = self._assess_content_structure(content, step.step_type)
        quality_score += structure_score * 0.25
        
        # 3. Step-type specific validation (25% of score)
        type_score = self._assess_step_type_quality(content, step.step_type)
        quality_score += type_score * 0.25
        
        # 4. Context relevance (25% of score)
        relevance_score = self._assess_context_relevance(content, context)
        quality_score += relevance_score * 0.25
        
        # Apply validation level adjustments
        if step.validation_level == ValidationLevel.STRICT:
            quality_score *= 0.9  # Higher standards
        elif step.validation_level == ValidationLevel.BASIC:
            quality_score = min(quality_score * 1.1, 1.0)  # More lenient
        
        return min(quality_score, 1.0)
    
    def _assess_content_structure(self, content: str, step_type: ChainStepType) -> float:
        """Assess content structure and organization."""
        structure_indicators = [
            "first", "second", "third", "furthermore", "additionally", 
            "however", "therefore", "in conclusion", "specifically"
        ]
        
        # Check for logical flow indicators
        structure_count = sum(1 for indicator in structure_indicators if indicator.lower() in content.lower())
        structure_score = min(structure_count / 3, 1.0)
        
        # Check for paragraphs/sections
        paragraph_count = len(content.split('\n\n'))
        if paragraph_count >= 3:
            structure_score += 0.3
        elif paragraph_count >= 2:
            structure_score += 0.15
        
        # Check for lists or bullet points
        if any(marker in content for marker in ['•', '-', '1.', '2.', '3.']):
            structure_score += 0.2
        
        return min(structure_score, 1.0)
    
    def _assess_step_type_quality(self, content: str, step_type: ChainStepType) -> float:
        """Assess quality based on step type requirements."""
        content_lower = content.lower()
        
        if step_type == ChainStepType.ANALYSIS:
            # Look for analytical terms
            analysis_terms = ["analyze", "assessment", "evaluation", "trends", "factors", "data", "metrics"]
            score = sum(1 for term in analysis_terms if term in content_lower) / len(analysis_terms)
            
        elif step_type == ChainStepType.SYNTHESIS:
            # Look for synthesis terms
            synthesis_terms = ["integrate", "combine", "synthesize", "overall", "comprehensive", "summary"]
            score = sum(1 for term in synthesis_terms if term in content_lower) / len(synthesis_terms)
            
        elif step_type == ChainStepType.EVALUATION:
            # Look for evaluation terms
            evaluation_terms = ["evaluate", "assess", "compare", "pros", "cons", "advantages", "disadvantages"]
            score = sum(1 for term in evaluation_terms if term in content_lower) / len(evaluation_terms)
            
        elif step_type == ChainStepType.RECOMMENDATION:
            # Look for recommendation terms
            recommendation_terms = ["recommend", "suggest", "propose", "should", "strategy", "action", "implement"]
            score = sum(1 for term in recommendation_terms if term in content_lower) / len(recommendation_terms)
            
        else:
            score = 0.7  # Default score for unknown types
        
        return min(score * 2, 1.0)  # Scale up and cap at 1.0
    
    def _assess_context_relevance(self, content: str, context: ChainContext) -> float:
        """Assess how well content relates to the provided context."""
        if not context.accumulated_insights:
            return 0.8  # Default for first step
        
        # Check if content references previous insights
        content_lower = content.lower()
        relevance_score = 0.0
        
        # Look for context integration
        context_terms = ["previous", "above", "earlier", "based on", "building on", "as mentioned"]
        context_integration = sum(1 for term in context_terms if term in content_lower)
        relevance_score += min(context_integration / 3, 0.5)
        
        # Check for business scenario relevance
        if context.initial_input:
            scenario_words = context.initial_input.lower().split()[:10]  # First 10 words
            scenario_mentions = sum(1 for word in scenario_words if len(word) > 3 and word in content_lower)
            relevance_score += min(scenario_mentions / 5, 0.5)
        
        return min(relevance_score, 1.0)
    
    def _extract_key_insights(self, content: str, step_type: ChainStepType) -> List[str]:
        """Extract key insights from step content."""
        insights = []
        
        # Split content into sentences
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        
        # Extract insights based on step type
        for sentence in sentences[:5]:  # Limit to first 5 sentences
            if step_type == ChainStepType.ANALYSIS and any(term in sentence.lower() for term in ['analysis', 'trend', 'data', 'metric']):
                insights.append(f"Analysis: {sentence}")
            elif step_type == ChainStepType.SYNTHESIS and any(term in sentence.lower() for term in ['overall', 'comprehensive', 'integrate']):
                insights.append(f"Synthesis: {sentence}")
            elif step_type == ChainStepType.EVALUATION and any(term in sentence.lower() for term in ['evaluate', 'compare', 'advantage']):
                insights.append(f"Evaluation: {sentence}")
            elif step_type == ChainStepType.RECOMMENDATION and any(term in sentence.lower() for term in ['recommend', 'suggest', 'should']):
                insights.append(f"Recommendation: {sentence}")
        
        return insights[:3]  # Return top 3 insights
    
    def _enhance_step_for_quality(self, step: ChainStep, failed_result: StepResult) -> ChainStep:
        """Enhance step configuration for quality recovery."""
        base_requirements = step.context_requirements or []
        enhanced_step = ChainStep(
            name=f"{step.name}_enhanced",
            step_type=step.step_type,
            prompt_template=step.prompt_template + "\n\nPlease provide a more detailed and comprehensive analysis with specific examples and concrete recommendations.",
            validation_level=step.validation_level,
            max_retries=1,  # Limit retries for enhanced step
            quality_threshold=step.quality_threshold * 0.9,  # Slightly lower threshold
            timeout_seconds=step.timeout_seconds,
            context_requirements=base_requirements + ["Enhanced detail required", "Specific examples needed"]
        )
        return enhanced_step
    
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
            validation_level=ValidationLevel.BASIC,
            quality_threshold=0.55,
            context_requirements=["Market size", "Growth trends", "Key players"]
        ),
        ChainStep(
            name="Competitive Analysis",
            step_type=ChainStepType.ANALYSIS,
            prompt_template="You are a market researcher. Analyze the competitive landscape in detail.",
            validation_level=ValidationLevel.BASIC,
            quality_threshold=0.55,
            context_requirements=["Competitor analysis", "Market positioning", "Differentiation"]
        ),
        ChainStep(
            name="Risk Assessment",
            step_type=ChainStepType.EVALUATION,
            prompt_template="You are a strategic consultant. Evaluate potential risks and challenges.",
            validation_level=ValidationLevel.BASIC,
            quality_threshold=0.55,
            context_requirements=["Risk identification", "Impact assessment", "Mitigation strategies"]
        ),
        ChainStep(
            name="Strategic Recommendations",
            step_type=ChainStepType.RECOMMENDATION,
            prompt_template="You are a strategic consultant. Provide specific strategic recommendations.",
            validation_level=ValidationLevel.BASIC,
            quality_threshold=0.55,
            context_requirements=["Action plan", "Implementation timeline", "Success metrics"]
        )
    ]


def run_comprehensive_test():
    """Demonstrate complete sequential chain workflow."""
    PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id")
    if PROJECT_ID == "your-project-id":
        print("❌ Please set PROJECT_ID environment variable")
        return
    
    # Initialize chain
    chain = SequentialChain(PROJECT_ID)
    
    # Load test scenario
    scenarios = load_business_scenarios()
    test_scenario = scenarios[0]
    
    # Create business scenario prompt
    scenario_prompt = f"""
Business Scenario:
Company: {test_scenario['company_name']}
Industry: {test_scenario['industry']}
Market Focus: {test_scenario['market_focus']}
Strategic Question: {test_scenario['strategic_question']}
Context: {test_scenario['additional_context']}

Please analyze this business scenario comprehensively.
"""
    
    # Create chain steps
    steps = create_sample_chain_steps()
    
    print("🔗 Testing Sequential Chain Implementation")
    print("=" * 60)
    
    # Execute chain
    try:
        result = chain.execute_chain(scenario_prompt, steps)
        
        if result.success:
            print("✅ Sequential chain executed successfully!")
            print(f"📊 Overall Quality: {result.overall_quality:.2f}")
            print(f"⏱️  Total Time: {result.total_execution_time:.1f}s")
            print(f"🎯 Steps Completed: {len(result.step_results)}")
            print(f"💰 Token Usage: {result.total_token_usage['total_input'] + result.total_token_usage['total_output']}")
            
            # Test specific TODOs
            print("\n🧪 TODO Validation:")
            print("✅ TODO 1: Chain step execution - IMPLEMENTED")
            print("✅ TODO 2: Context flow management - IMPLEMENTED") 
            print("✅ TODO 3: Chain quality validation - IMPLEMENTED")
            
        else:
            print("❌ Chain execution failed:")
            print(f"Error: {result.error_summary}")
            
    except Exception as e:
        print(f"❌ Chain execution error: {e}")


if __name__ == "__main__":
    run_comprehensive_test()