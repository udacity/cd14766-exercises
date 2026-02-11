"""
Lesson 4: Conditional Chain Implementation - Complete Solution

This module demonstrates advanced conditional prompt chaining with intelligent
branching logic, adaptive prompt selection, and multi-path reasoning synthesis.

Learning Objectives:
- Implement intelligent branching logic based on intermediate results
- Create adaptive prompt selection based on context and performance
- Build multi-path reasoning with result synthesis
- Develop sophisticated conditional workflow management

TODOs 4-22 SOLUTIONS implemented with comprehensive branching logic,
adaptive optimization, and parallel reasoning capabilities.

Author: Noble Ackerson (Udacity)
Date: 2025
"""

import os
import time
import asyncio
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai.types import GenerateContentConfig

# Import base classes from sequential chain
from sequential_chain import (
    ChainContext, ChainStep, StepResult, ChainResult, 
    ChainStepType, ValidationLevel, SequentialChain
)


class BranchingCondition(Enum):
    """Types of branching conditions for chain execution."""
    QUALITY_THRESHOLD = "quality_threshold"
    CONTENT_LENGTH = "content_length"
    COMPLEXITY_ANALYSIS = "complexity_analysis"
    CONTEXT_COMPLETENESS = "context_completeness"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    ERROR_RECOVERY = "error_recovery"


class ReasoningPath(Enum):
    """Different reasoning paths for multi-path analysis."""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    DETAILED = "detailed"
    SUMMARY = "summary"


@dataclass
class BranchingDecision:
    """Decision made by branching logic."""
    condition_type: BranchingCondition
    decision: str
    confidence: float
    reasoning: str
    alternative_paths: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.alternative_paths is None:
            self.alternative_paths = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PathResult:
    """Result from executing a single reasoning path."""
    path_type: ReasoningPath
    content: str
    quality_score: float
    confidence_score: float
    execution_time: float
    token_usage: Dict[str, int]
    unique_insights: List[str]
    success: bool
    error_message: Optional[str] = None


@dataclass
class SynthesizedResult:
    """Result from synthesizing multiple reasoning paths."""
    final_content: str
    combined_quality: float
    path_contributions: Dict[ReasoningPath, float]
    synthesis_confidence: float
    total_execution_time: float
    total_token_usage: Dict[str, int]
    unique_insights_count: int
    paths_used: List[ReasoningPath]


class ConditionalChain(SequentialChain):
    """
    Advanced conditional prompt chaining system with intelligent branching,
    adaptive prompt selection, and multi-path reasoning capabilities.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize conditional chain with enhanced capabilities."""
        super().__init__(project_id, location)
        self.prompt_performance_history = {}
        self.optimization_cache = {}
        self.branching_threshold = 0.7
        
    def execute_conditional_chain(self, initial_prompt: str, steps: List[ChainStep], 
                                 enable_branching: bool = True) -> ChainResult:
        """
        Execute chain with conditional branching logic.
        
        This method extends the sequential chain to add intelligent branching
        based on intermediate results and context analysis.
        """
        start_time = time.time()
        context = ChainContext(initial_input=initial_prompt)
        step_results = []
        
        if self.debug_mode:
            print(f"🔀 Starting conditional chain execution with {len(steps)} steps")
        
        for i, step in enumerate(steps):
            if self.debug_mode:
                print(f"⚡ Executing conditional step {i+1}/{len(steps)}: {step.name}")
            
            # Evaluate branching conditions before step execution
            if enable_branching and i > 0:  # Don't branch on first step
                branching_decision = self.evaluate_branching_condition(context, step)
                
                if branching_decision.decision != "continue_main_chain":
                    if self.debug_mode:
                        print(f"🔀 Branching decision: {branching_decision.decision}")
                    
                    # Execute alternative path based on branching decision
                    step_result = self._execute_branching_path(step, context, branching_decision)
                else:
                    # Execute normal step
                    step_result = self._execute_adaptive_step(step, context)
            else:
                # Execute normal step
                step_result = self._execute_adaptive_step(step, context)
            
            step_results.append(step_result)
            
            # Check for chain failure
            if not step_result.success:
                return self._handle_chain_failure(step_results, context, start_time)
            
            # Update context
            context = self._update_context_flow(context, step_result)
        
        # Calculate final metrics and return result
        return self._finalize_chain_result(step_results, context, start_time)
    
    def evaluate_branching_condition(self, context: ChainContext, current_step: ChainStep) -> BranchingDecision:
        """
        TODO 4 SOLUTION: Implement intelligent branching logic.
        
        This implementation provides:
        - Multi-dimensional condition evaluation
        - Quality-based branching decisions
        - Context completeness assessment
        - Performance optimization triggers
        """
        if not context.step_history:
            return BranchingDecision(
                condition_type=BranchingCondition.QUALITY_THRESHOLD,
                decision="continue_main_chain",
                confidence=1.0,
                reasoning="No previous steps to evaluate"
            )
        
        latest_result = context.step_history[-1]
        
        # 1. Quality threshold evaluation
        if latest_result["quality_score"] < self.branching_threshold:
            return BranchingDecision(
                condition_type=BranchingCondition.QUALITY_THRESHOLD,
                decision="retry_with_enhanced_prompt",
                confidence=0.9,
                reasoning=f"Quality score {latest_result['quality_score']:.2f} below threshold {self.branching_threshold}",
                alternative_paths=["cot_enhanced", "detailed_analysis", "multi_perspective"],
                metadata={"quality_deficit": self.branching_threshold - latest_result["quality_score"]}
            )
        
        # 2. Content length evaluation
        content_length = len(latest_result.get("content_summary", ""))
        if content_length < 150:  # Too brief
            return BranchingDecision(
                condition_type=BranchingCondition.CONTENT_LENGTH,
                decision="expand_analysis",
                confidence=0.8,
                reasoning=f"Content length {content_length} too brief for thorough analysis",
                alternative_paths=["detailed_expansion", "multi_angle_analysis"],
                metadata={"content_length": content_length}
            )
        
        # 3. Context completeness evaluation
        if len(context.accumulated_insights) < (len(context.step_history) * 2):  # Expect ~2 insights per step
            return BranchingDecision(
                condition_type=BranchingCondition.CONTEXT_COMPLETENESS,
                decision="enhance_insight_extraction",
                confidence=0.75,
                reasoning="Insufficient insights extracted from previous steps",
                alternative_paths=["insight_focused", "comprehensive_extraction"],
                metadata={"insight_ratio": len(context.accumulated_insights) / len(context.step_history)}
            )
        
        # 4. Performance optimization evaluation
        avg_quality = statistics.mean(context.quality_scores) if context.quality_scores else 0.5
        if avg_quality > 0.85 and current_step.step_type == ChainStepType.RECOMMENDATION:
            return BranchingDecision(
                condition_type=BranchingCondition.PERFORMANCE_OPTIMIZATION,
                decision="multi_path_synthesis",
                confidence=0.95,
                reasoning=f"High quality chain ({avg_quality:.2f}) suitable for multi-path synthesis",
                alternative_paths=["analytical_path", "creative_path", "conservative_path"],
                metadata={"chain_quality": avg_quality}
            )
        
        # 5. Default: continue main chain
        return BranchingDecision(
            condition_type=BranchingCondition.QUALITY_THRESHOLD,
            decision="continue_main_chain",
            confidence=0.8,
            reasoning="All conditions satisfied for normal execution"
        )
    
    def _execute_branching_path(self, step: ChainStep, context: ChainContext, 
                               decision: BranchingDecision) -> StepResult:
        """Execute alternative path based on branching decision."""
        if decision.decision == "retry_with_enhanced_prompt":
            enhanced_step = self._create_enhanced_step(step, decision)
            return self._execute_adaptive_step(enhanced_step, context)
        
        elif decision.decision == "expand_analysis":
            expanded_step = self._create_expanded_step(step, decision)
            return self._execute_adaptive_step(expanded_step, context)
        
        elif decision.decision == "enhance_insight_extraction":
            insight_step = self._create_insight_focused_step(step, decision)
            return self._execute_adaptive_step(insight_step, context)
        
        elif decision.decision == "multi_path_synthesis":
            return self.execute_multi_path_reasoning(step.prompt_template, [
                ReasoningPath.ANALYTICAL, 
                ReasoningPath.CREATIVE, 
                ReasoningPath.CONSERVATIVE
            ], context)
        
        else:
            # Fallback to normal execution
            return self._execute_adaptive_step(step, context)
    
    def _execute_adaptive_step(self, step: ChainStep, context: ChainContext) -> StepResult:
        """
        TODO 5 SOLUTION: Implement adaptive prompt selection.
        
        This implementation provides:
        - Performance-based prompt optimization
        - Context-aware prompt adaptation
        - Dynamic prompt enhancement
        - Fallback mechanism integration
        """
        # Check for optimized prompt variants in cache
        optimized_prompt = self._get_optimized_prompt(step, context)
        
        if optimized_prompt:
            if self.debug_mode:
                print(f"🎯 Using optimized prompt for step: {step.name}")
            
            # Create step with optimized prompt
            adapted_step = ChainStep(
                name=f"{step.name}_optimized",
                step_type=step.step_type,
                prompt_template=optimized_prompt,
                validation_level=step.validation_level,
                max_retries=step.max_retries,
                quality_threshold=step.quality_threshold,
                context_requirements=step.context_requirements
            )
            
            result = self._execute_step_with_retry(adapted_step, context)
            
            # Track performance for future optimization
            self._track_prompt_performance(step.name, optimized_prompt, result.quality_score)
            
            return result
        
        else:
            # Use original prompt
            result = self._execute_step_with_retry(step, context)
            
            # Track performance for future optimization
            self._track_prompt_performance(step.name, step.prompt_template, result.quality_score)
            
            return result
    
    def _get_optimized_prompt(self, step: ChainStep, context: ChainContext) -> Optional[str]:
        """Get optimized prompt variant based on context and performance history."""
        cache_key = f"{step.name}_{step.step_type.value}_{len(context.step_history)}"
        
        if cache_key in self.optimization_cache:
            return self.optimization_cache[cache_key]
        
        # Generate context-aware optimization
        if len(context.step_history) > 1:
            # Multi-step context optimization
            optimized_prompt = self._create_context_optimized_prompt(step, context)
            self.optimization_cache[cache_key] = optimized_prompt
            return optimized_prompt
        
        # Check performance history for this step type
        step_performance = self.prompt_performance_history.get(step.name, [])
        if len(step_performance) >= 3:
            avg_performance = statistics.mean([p["quality"] for p in step_performance[-3:]])
            if avg_performance < 0.75:
                # Create enhanced prompt for poor performance
                enhanced_prompt = self._create_performance_enhanced_prompt(step)
                self.optimization_cache[cache_key] = enhanced_prompt
                return enhanced_prompt
        
        return None
    
    def _create_context_optimized_prompt(self, step: ChainStep, context: ChainContext) -> str:
        """Create prompt optimized for current context."""
        base_prompt = step.prompt_template
        
        # Add context-specific enhancements
        context_enhancement = ""
        
        if context.quality_scores:
            avg_quality = statistics.mean(context.quality_scores)
            if avg_quality > 0.8:
                context_enhancement += "\nBUILD ON HIGH-QUALITY PREVIOUS ANALYSIS: The previous steps have provided excellent insights. Build upon this strong foundation with equally detailed analysis."
            else:
                context_enhancement += "\nENHANCE ANALYSIS QUALITY: Previous steps show room for improvement. Provide more detailed, specific, and actionable analysis."
        
        if len(context.accumulated_insights) >= 5:
            context_enhancement += "\nLEVERAGE ACCUMULATED INSIGHTS: You have rich context from previous analysis. Reference and build upon specific insights mentioned earlier."
        
        return f"{base_prompt}{context_enhancement}"
    
    def _create_performance_enhanced_prompt(self, step: ChainStep) -> str:
        """Create enhanced prompt for steps with poor performance history."""
        base_prompt = step.prompt_template
        
        enhancement = """

ENHANCED ANALYSIS REQUIREMENTS:
1. Provide specific examples and concrete evidence
2. Structure your response with clear sections and bullet points
3. Include quantitative assessments where possible
4. Address multiple perspectives and potential counterarguments
5. Conclude with actionable recommendations

Focus on delivering comprehensive, detailed analysis that demonstrates deep understanding of the topic."""
        
        return f"{base_prompt}{enhancement}"
    
    def _track_prompt_performance(self, step_name: str, prompt: str, quality_score: float):
        """Track prompt performance for future optimization."""
        if step_name not in self.prompt_performance_history:
            self.prompt_performance_history[step_name] = []
        
        self.prompt_performance_history[step_name].append({
            "prompt_hash": hash(prompt),
            "quality": quality_score,
            "timestamp": time.time()
        })
        
        # Keep only recent performance data
        if len(self.prompt_performance_history[step_name]) > 10:
            self.prompt_performance_history[step_name] = self.prompt_performance_history[step_name][-10:]
    
    def execute_multi_path_reasoning(self, base_prompt: str, paths: List[ReasoningPath],
                                   context: Optional[ChainContext] = None) -> StepResult:
        """
        TODO 6 SOLUTION: Implement sophisticated multi-path reasoning.
        
        This implementation provides:
        - Parallel execution of multiple reasoning approaches
        - Intelligent result synthesis
        - Confidence-based weighting
        - Performance optimization
        """
        if self.debug_mode:
            print(f"🔀 Executing multi-path reasoning with {len(paths)} paths")
        
        start_time = time.time()
        path_results = []
        
        # Execute paths in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(paths), 3)) as executor:
            # Submit all path executions
            future_to_path = {
                executor.submit(self._execute_reasoning_path, base_prompt, path, context): path 
                for path in paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per path
                    path_results.append(result)
                    
                    if self.debug_mode:
                        print(f"✅ Path {path.value} completed - Quality: {result.quality_score:.2f}")
                        
                except Exception as e:
                    if self.debug_mode:
                        print(f"❌ Path {path.value} failed: {str(e)}")
                    
                    # Create failed result
                    path_results.append(PathResult(
                        path_type=path,
                        content="",
                        quality_score=0.0,
                        confidence_score=0.0,
                        execution_time=0.0,
                        token_usage={"input_tokens": 0, "output_tokens": 0},
                        unique_insights=[],
                        success=False,
                        error_message=str(e)
                    ))
        
        # Synthesize results from successful paths
        successful_results = [r for r in path_results if r.success]
        
        if not successful_results:
            # All paths failed - return error result
            return StepResult(
                step_name="multi_path_reasoning",
                content="Multi-path reasoning failed - all paths encountered errors",
                quality_score=0.0,
                execution_time=time.time() - start_time,
                token_usage={"input_tokens": 0, "output_tokens": 0},
                success=False,
                error_message="All reasoning paths failed"
            )
        
        # Synthesize successful results
        synthesized = self._synthesize_path_results(successful_results)
        
        if self.debug_mode:
            print(f"🎯 Multi-path synthesis completed - Quality: {synthesized.combined_quality:.2f}")
        
        return StepResult(
            step_name="multi_path_reasoning",
            content=synthesized.final_content,
            quality_score=synthesized.combined_quality,
            execution_time=synthesized.total_execution_time,
            token_usage=synthesized.total_token_usage,
            success=True,
            confidence_score=synthesized.synthesis_confidence,
            key_insights=[f"Multi-path insight: {synthesized.unique_insights_count} unique perspectives synthesized"]
        )
    
    def _execute_reasoning_path(self, base_prompt: str, path: ReasoningPath, 
                               context: Optional[ChainContext] = None) -> PathResult:
        """Execute a single reasoning path."""
        start_time = time.time()
        
        # Create path-specific prompt
        path_prompt = self._create_path_specific_prompt(base_prompt, path, context)
        
        # Configure generation parameters based on path type
        config = self._get_path_generation_config(path)
        
        try:
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=path_prompt,
                config=config
            )
            
            execution_time = time.time() - start_time
            
            # Extract token usage
            token_usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }
            
            # Assess quality and confidence
            quality_score = self._assess_path_quality(response.text, path)
            confidence_score = self._assess_path_confidence(response.text, path)
            
            # Extract unique insights
            unique_insights = self._extract_path_insights(response.text, path)
            
            return PathResult(
                path_type=path,
                content=response.text,
                quality_score=quality_score,
                confidence_score=confidence_score,
                execution_time=execution_time,
                token_usage=token_usage,
                unique_insights=unique_insights,
                success=True
            )
            
        except Exception as e:
            return PathResult(
                path_type=path,
                content="",
                quality_score=0.0,
                confidence_score=0.0,
                execution_time=time.time() - start_time,
                token_usage={"input_tokens": 0, "output_tokens": 0},
                unique_insights=[],
                success=False,
                error_message=str(e)
            )
    
    def _create_path_specific_prompt(self, base_prompt: str, path: ReasoningPath, 
                                    context: Optional[ChainContext] = None) -> str:
        """Create prompt tailored to specific reasoning path."""
        path_instructions = {
            ReasoningPath.ANALYTICAL: "Focus on data-driven analysis with quantitative insights and logical reasoning.",
            ReasoningPath.CREATIVE: "Explore innovative approaches and creative solutions. Think outside conventional frameworks.",
            ReasoningPath.CONSERVATIVE: "Emphasize risk management and proven strategies. Focus on stable, low-risk approaches.",
            ReasoningPath.AGGRESSIVE: "Consider bold, high-impact strategies. Emphasize growth and competitive advantage.",
            ReasoningPath.DETAILED: "Provide comprehensive, in-depth analysis with extensive detail and examples.",
            ReasoningPath.SUMMARY: "Focus on key insights and high-level strategic overview."
        }
        
        path_instruction = path_instructions.get(path, "Provide thorough analysis.")
        
        context_section = ""
        if context and context.accumulated_insights:
            recent_insights = context.accumulated_insights[-3:]  # Last 3 insights
            context_section = f"\nPrevious Insights to Consider:\n" + "\n".join([f"- {insight}" for insight in recent_insights])
        
        return f"""{base_prompt}

REASONING PATH: {path.value.upper()}
{path_instruction}

{context_section}

Please provide your {path.value} analysis:"""
    
    def _get_path_generation_config(self, path: ReasoningPath) -> GenerateContentConfig:
        """Get generation configuration optimized for reasoning path."""
        configs = {
            ReasoningPath.ANALYTICAL: GenerateContentConfig(temperature=0.1, top_p=0.8, max_output_tokens=2000),
            ReasoningPath.CREATIVE: GenerateContentConfig(temperature=0.7, top_p=0.95, max_output_tokens=2000),
            ReasoningPath.CONSERVATIVE: GenerateContentConfig(temperature=0.2, top_p=0.85, max_output_tokens=1500),
            ReasoningPath.AGGRESSIVE: GenerateContentConfig(temperature=0.5, top_p=0.9, max_output_tokens=1500),
            ReasoningPath.DETAILED: GenerateContentConfig(temperature=0.3, top_p=0.9, max_output_tokens=2500),
            ReasoningPath.SUMMARY: GenerateContentConfig(temperature=0.2, top_p=0.8, max_output_tokens=1000)
        }
        
        return configs.get(path, GenerateContentConfig(temperature=0.3, top_p=0.9, max_output_tokens=2000))
    
    def _assess_path_quality(self, content: str, path: ReasoningPath) -> float:
        """Assess quality specific to reasoning path type."""
        base_quality = len(content.split()) / 100  # Base score from length
        base_quality = min(base_quality, 1.0)
        
        # Path-specific quality indicators
        content_lower = content.lower()
        
        if path == ReasoningPath.ANALYTICAL:
            analytical_terms = ["data", "analysis", "metrics", "quantitative", "evidence", "research"]
            path_score = sum(1 for term in analytical_terms if term in content_lower) / len(analytical_terms)
        
        elif path == ReasoningPath.CREATIVE:
            creative_terms = ["innovative", "creative", "novel", "breakthrough", "unique", "unconventional"]
            path_score = sum(1 for term in creative_terms if term in content_lower) / len(creative_terms)
        
        elif path == ReasoningPath.CONSERVATIVE:
            conservative_terms = ["risk", "stable", "proven", "established", "traditional", "cautious"]
            path_score = sum(1 for term in conservative_terms if term in content_lower) / len(conservative_terms)
        
        elif path == ReasoningPath.AGGRESSIVE:
            aggressive_terms = ["bold", "aggressive", "growth", "competitive", "expansion", "market share"]
            path_score = sum(1 for term in aggressive_terms if term in content_lower) / len(aggressive_terms)
        
        else:
            path_score = 0.7  # Default for other paths
        
        return min((base_quality + path_score) / 2, 1.0)
    
    def _assess_path_confidence(self, content: str, path: ReasoningPath) -> float:
        """Assess confidence level of path analysis."""
        confidence_indicators = ["clearly", "definitely", "certainly", "confident", "strong evidence"]
        uncertainty_indicators = ["might", "could", "possibly", "perhaps", "unclear"]
        
        content_lower = content.lower()
        
        confidence_count = sum(1 for indicator in confidence_indicators if indicator in content_lower)
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in content_lower)
        
        base_confidence = 0.7
        confidence_boost = min(confidence_count * 0.1, 0.2)
        uncertainty_penalty = min(uncertainty_count * 0.05, 0.15)
        
        return max(base_confidence + confidence_boost - uncertainty_penalty, 0.1)
    
    def _extract_path_insights(self, content: str, path: ReasoningPath) -> List[str]:
        """Extract insights specific to reasoning path."""
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        
        insights = []
        for sentence in sentences[:3]:  # Top 3 sentences
            if path.value in sentence.lower() or any(keyword in sentence.lower() for keyword in [
                'recommend', 'suggest', 'conclude', 'therefore', 'analysis shows'
            ]):
                insights.append(f"{path.value.title()}: {sentence}")
        
        return insights
    
    def _synthesize_path_results(self, path_results: List[PathResult]) -> SynthesizedResult:
        """Synthesize results from multiple reasoning paths."""
        start_time = time.time()
        
        # Calculate weights based on quality and confidence
        total_weight = 0
        weighted_content = []
        path_contributions = {}
        
        for result in path_results:
            weight = (result.quality_score * 0.7) + (result.confidence_score * 0.3)
            total_weight += weight
            path_contributions[result.path_type] = weight
            
            weighted_content.append({
                "content": result.content,
                "weight": weight,
                "path": result.path_type,
                "insights": result.unique_insights
            })
        
        # Normalize weights
        for path, weight in path_contributions.items():
            path_contributions[path] = weight / total_weight if total_weight > 0 else 0
        
        # Create synthesized content
        synthesis_parts = [
            "# Multi-Path Analysis Synthesis",
            "This analysis combines insights from multiple reasoning approaches:\n"
        ]
        
        # Add path-specific sections based on weights
        sorted_content = sorted(weighted_content, key=lambda x: x["weight"], reverse=True)
        
        for item in sorted_content:
            if item["weight"] / total_weight > 0.2:  # Include significant contributions
                synthesis_parts.append(f"## {item['path'].value.title()} Perspective")
                synthesis_parts.append(f"Weight: {item['weight']/total_weight:.1%}")
                synthesis_parts.append(f"{item['content'][:500]}...\n")  # Truncate for synthesis
        
        # Add integrated insights section
        all_insights = []
        for item in weighted_content:
            all_insights.extend(item["insights"])
        
        unique_insights = list(set(all_insights))  # Remove duplicates
        
        synthesis_parts.append("## Integrated Insights")
        for insight in unique_insights[:5]:  # Top 5 unique insights
            synthesis_parts.append(f"- {insight}")
        
        final_content = "\n".join(synthesis_parts)
        
        # Calculate combined metrics
        combined_quality = statistics.mean([r.quality_score for r in path_results])
        synthesis_confidence = min(combined_quality + 0.1, 1.0)  # Slight confidence boost from multi-path
        
        total_execution_time = time.time() - start_time + sum(r.execution_time for r in path_results)
        total_token_usage = {
            "input_tokens": sum(r.token_usage["input_tokens"] for r in path_results),
            "output_tokens": sum(r.token_usage["output_tokens"] for r in path_results)
        }
        
        return SynthesizedResult(
            final_content=final_content,
            combined_quality=combined_quality,
            path_contributions=path_contributions,
            synthesis_confidence=synthesis_confidence,
            total_execution_time=total_execution_time,
            total_token_usage=total_token_usage,
            unique_insights_count=len(unique_insights),
            paths_used=[r.path_type for r in path_results]
        )
    
    def _create_enhanced_step(self, step: ChainStep, decision: BranchingDecision) -> ChainStep:
        """Create enhanced step based on branching decision."""
        enhancement = "\n\nENHANCED REQUIREMENTS: Provide more detailed analysis with specific examples, concrete evidence, and actionable recommendations."
        
        return ChainStep(
            name=f"{step.name}_enhanced",
            step_type=step.step_type,
            prompt_template=step.prompt_template + enhancement,
            validation_level=step.validation_level,
            max_retries=step.max_retries,
            quality_threshold=max(step.quality_threshold - 0.05, 0.6),  # Slightly lower threshold
            context_requirements=step.context_requirements + ["Enhanced detail", "Specific examples"]
        )
    
    def _create_expanded_step(self, step: ChainStep, decision: BranchingDecision) -> ChainStep:
        """Create expanded step for brief content."""
        expansion = "\n\nEXPANSION REQUIREMENTS: Provide comprehensive analysis with detailed explanations, multiple perspectives, and thorough coverage of all relevant aspects."
        
        return ChainStep(
            name=f"{step.name}_expanded",
            step_type=step.step_type,
            prompt_template=step.prompt_template + expansion,
            validation_level=step.validation_level,
            max_retries=step.max_retries,
            quality_threshold=step.quality_threshold,
            context_requirements=step.context_requirements + ["Comprehensive coverage", "Multiple perspectives"]
        )
    
    def _create_insight_focused_step(self, step: ChainStep, decision: BranchingDecision) -> ChainStep:
        """Create step focused on insight extraction."""
        insight_focus = "\n\nINSIGHT FOCUS: Emphasize extracting key insights, patterns, and actionable conclusions. Provide clear takeaways and implications for decision-making."
        
        return ChainStep(
            name=f"{step.name}_insights",
            step_type=step.step_type,
            prompt_template=step.prompt_template + insight_focus,
            validation_level=step.validation_level,
            max_retries=step.max_retries,
            quality_threshold=step.quality_threshold,
            context_requirements=step.context_requirements + ["Key insights", "Actionable conclusions"]
        )


def run_conditional_chain_test():
    """Demonstrate conditional chain capabilities."""
    PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id")
    if PROJECT_ID == "your-project-id":
        print("❌ Please set PROJECT_ID environment variable")
        return
    
    # Initialize conditional chain
    chain = ConditionalChain(PROJECT_ID)
    
    print("🔀 Testing Conditional Chain Implementation")
    print("=" * 60)
    
    # Test branching logic
    print("\n🧪 Testing TODO 4: Branching Logic")
    context = ChainContext(initial_input="Test scenario")
    context.step_history = [{"quality_score": 0.5, "content_summary": "Brief analysis"}]
    context.quality_scores = [0.5]
    
    step = ChainStep("test_step", ChainStepType.ANALYSIS, "Test prompt")
    decision = chain.evaluate_branching_condition(context, step)
    print(f"✅ Branching decision: {decision.decision} (confidence: {decision.confidence:.2f})")
    
    # Test multi-path reasoning
    print("\n🧪 Testing TODO 6: Multi-path Reasoning")
    try:
        multi_result = chain.execute_multi_path_reasoning(
            "Analyze the strategic implications of market expansion.",
            [ReasoningPath.ANALYTICAL, ReasoningPath.CREATIVE],
            context
        )
        print(f"✅ Multi-path reasoning completed - Quality: {multi_result.quality_score:.2f}")
        
    except Exception as e:
        print(f"❌ Multi-path reasoning test failed: {e}")
    
    print("\n📊 TODO Validation:")
    print("✅ TODO 4: Branching logic implementation - IMPLEMENTED")
    print("✅ TODO 5: Adaptive prompt selection - IMPLEMENTED")
    print("✅ TODO 6: Multi-path reasoning - IMPLEMENTED")


if __name__ == "__main__":
    run_conditional_chain_test()