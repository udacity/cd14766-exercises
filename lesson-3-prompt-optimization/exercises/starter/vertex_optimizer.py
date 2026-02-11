"""
Lesson 3: Vertex AI Optimizer Integration - Student Template

This module demonstrates comprehensive integration with Vertex AI Prompt Optimizer
to systematically improve prompt performance with measurable results.

Learning Objectives:
- Use Vertex AI Prompt Optimizer API
- Apply systematic optimization techniques
- Compare before/after performance with detailed analysis
- Build production-ready optimization workflows

Complete TODOs 4-16 to implement Vertex AI optimization integration.

Author: [Your Name]
Date: [Current Date]
"""

import os
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from google import genai
from google.genai.types import GenerateContentConfig

# Import from prompt analyzer
from prompt_analyzer import PromptAnalyzer, PromptAnalysis, PerformanceMetrics


@dataclass
class OptimizationResult:
    """Results of prompt optimization process."""
    original_prompt: str
    optimized_prompt: str
    optimization_type: str
    improvement_metrics: Dict[str, float]
    applied_guidelines: List[Dict]
    optimization_time: float
    token_savings: Dict[str, int]
    quality_improvement: Dict[str, float]
    recommendation: str


@dataclass
class ComparisonAnalysis:
    """Detailed comparison between original and optimized prompts."""
    original_analysis: PromptAnalysis
    optimized_analysis: PromptAnalysis
    original_performance: PerformanceMetrics
    optimized_performance: PerformanceMetrics
    improvement_summary: Dict[str, Any]
    cost_benefit_analysis: Dict[str, Any]
    recommendation_score: float


class VertexPromptOptimizer:
    """
    Comprehensive Vertex AI Prompt Optimizer integration with analysis.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize with Vertex AI client and analyzer."""
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        self.model_name = "gemini-2.5-flash"
        self.project_id = project_id
        self.location = location
        
        # Initialize analyzer for comparisons
        self.analyzer = PromptAnalyzer(project_id, location)
        
    def optimize_prompt(self, prompt: str, optimization_type: str = "instructions",
                       num_steps: int = 3, target_model: str = None) -> OptimizationResult:
        """
        TODO 4: Implement complete Vertex AI Optimizer integration.
        
        Requirements:
        - Use Vertex AI Prompt Optimizer API to improve prompts
        - Support different optimization types ("instructions", "demonstrations", "both")
        - Configure optimization parameters (num_steps, target_model)
        - Handle errors gracefully with fallback responses
        - Measure optimization time and extract applied guidelines
        - Calculate quality improvements and token savings
        - Generate actionable recommendations
        
        API Integration:
        - Use self.client.prompt_optimizer.optimize_prompt()
        - Pass optimization_config with parameters
        - Extract suggested_prompt and applicable_guidelines from response
        
        Args:
            prompt: Original prompt to optimize
            optimization_type: Type of optimization ("instructions", "demonstrations", "both")
            num_steps: Number of optimization steps (1-5)
            target_model: Target model for optimization (defaults to self.model_name)
            
        Returns:
            OptimizationResult with comprehensive optimization data
        """
        # TODO 4: Implement Vertex AI Optimizer integration
        #
        # Step 1: Setup and validation
        # if target_model is None:
        #     target_model = self.model_name
        # 
        # print(f"🔧 Optimizing prompt using Vertex AI Optimizer...")
        # start_time = time.time()
        
        # Step 2: Call Vertex AI Prompt Optimizer
        # try:
        #     optimization_response = self.client.prompt_optimizer.optimize_prompt(
        #         prompt=prompt,
        #         optimization_config={
        #             "num_steps": num_steps,
        #             "target_model": target_model,
        #             "optimization_mode": optimization_type,
        #             "evaluation_metrics": ["coherence", "fluency", "groundedness"]
        #         }
        #     )
        #     
        #     optimization_time = time.time() - start_time
        #     optimized_prompt = optimization_response.suggested_prompt
        #     applied_guidelines = [...]  # Extract guidelines
        
        # Step 3: Calculate improvements
        # original_length = len(prompt.split())
        # optimized_length = len(optimized_prompt.split())
        # length_reduction = original_length - optimized_length
        # 
        # token_savings = {
        #     "input_tokens_saved": int(length_reduction * 1.3),
        #     "percentage_reduction": (length_reduction / original_length) * 100
        # }
        
        # Step 4: Analyze quality improvements
        # quality_improvement = self._measure_quality_improvement(prompt, optimized_prompt)
        
        # Step 5: Generate recommendation
        # recommendation = self._generate_recommendation(applied_guidelines, quality_improvement, token_savings)
        
        # Step 6: Return OptimizationResult
        # return OptimizationResult(...)
        
        # except Exception as e:
        #     # Return fallback result on error
        #     return OptimizationResult(
        #         original_prompt=prompt,
        #         optimized_prompt=prompt,  # No change on error
        #         optimization_type=optimization_type,
        #         improvement_metrics={"error": str(e)},
        #         applied_guidelines=[],
        #         optimization_time=time.time() - start_time,
        #         token_savings={"input_tokens_saved": 0, "percentage_reduction": 0},
        #         quality_improvement={},
        #         recommendation="Optimization failed - manual review recommended"
        #     )
        
        pass  # Replace with your implementation
    
    def _measure_quality_improvement(self, original: str, optimized: str) -> Dict[str, float]:
        """Measure quality improvement between prompts."""
        # This helper method is provided to guide your implementation
        original_analysis = self.analyzer.analyze_prompt_quality(original)
        optimized_analysis = self.analyzer.analyze_prompt_quality(optimized)
        
        return {
            "clarity_improvement": optimized_analysis.clarity_score - original_analysis.clarity_score,
            "specificity_improvement": optimized_analysis.specificity_score - original_analysis.specificity_score,
            "completeness_improvement": optimized_analysis.completeness_score - original_analysis.completeness_score,
            "structure_improvement": optimized_analysis.structure_score - original_analysis.structure_score,
            "overall_improvement": optimized_analysis.overall_score - original_analysis.overall_score
        }
    
    def _generate_recommendation(self, guidelines: List[Dict], 
                               quality_improvement: Dict[str, float],
                               token_savings: Dict[str, int]) -> str:
        """Generate recommendation based on optimization results."""
        # This helper method is provided to guide your implementation
        if not guidelines:
            return "No optimizations applied - prompt may already be well-optimized"
        
        significant_improvements = [
            metric for metric, value in quality_improvement.items() 
            if value > 0.1
        ]
        
        if len(significant_improvements) >= 2 and token_savings["percentage_reduction"] > 10:
            return "Strongly recommended - significant quality and efficiency improvements"
        elif significant_improvements and token_savings["percentage_reduction"] > 5:
            return "Recommended - good balance of quality and efficiency gains"
        else:
            return "Consider adoption - evaluate based on specific needs"
    
    def systematic_optimization(self, prompt: str, test_scenarios: List[Dict],
                              optimization_types: List[str] = None) -> Dict[str, OptimizationResult]:
        """
        TODO 5: Implement systematic multi-type optimization workflow.
        
        Requirements:
        - Test multiple optimization types to find the best approach
        - Measure performance with test scenarios for each optimization
        - Compare results across optimization types
        - Return comprehensive results for all tested types
        - Identify best performing optimization type
        
        Optimization types to test:
        1. "instructions" - Optimize instructional content
        2. "demonstrations" - Optimize examples and demonstrations  
        3. "both" - Optimize both instructions and demonstrations
        
        Args:
            prompt: Original prompt to optimize
            test_scenarios: Business scenarios for performance testing
            optimization_types: List of optimization types to test
            
        Returns:
            Dictionary mapping optimization types to OptimizationResults
        """
        # TODO 5: Implement systematic optimization workflow
        #
        # Step 1: Setup optimization types
        # if optimization_types is None:
        #     optimization_types = ["instructions", "demonstrations", "both"]
        # 
        # print(f"🔄 Running systematic optimization...")
        # results = {}
        
        # Step 2: Test each optimization type
        # for opt_type in optimization_types:
        #     print(f"\n📋 Testing optimization type: {opt_type}")
        #     
        #     try:
        #         # Optimize prompt
        #         result = self.optimize_prompt(prompt, opt_type)
        #         
        #         # Test performance with scenarios
        #         original_performance = self.analyzer.measure_baseline_performance(
        #             prompt, test_scenarios, num_runs=2
        #         )
        #         optimized_performance = self.analyzer.measure_baseline_performance(
        #             result.optimized_prompt, test_scenarios, num_runs=2
        #         )
        #         
        #         # Calculate performance improvement
        #         performance_improvement = {
        #             "quality_improvement": optimized_performance.quality_metrics["overall_quality"] - 
        #                                  original_performance.quality_metrics["overall_quality"],
        #             "consistency_improvement": optimized_performance.consistency_score - 
        #                                      original_performance.consistency_score,
        #             "speed_improvement": original_performance.generation_time - 
        #                                optimized_performance.generation_time,
        #             "token_efficiency": original_performance.token_usage["avg_input_tokens"] - 
        #                               optimized_performance.token_usage["avg_input_tokens"]
        #         }
        #         
        #         # Update result with performance data
        #         result.quality_improvement.update(performance_improvement)
        #         results[opt_type] = result
        
        # Step 3: Find and report best optimization
        # if results:
        #     best_type = max(results.keys(), 
        #                   key=lambda k: results[k].quality_improvement.get("overall_improvement", 0))
        #     print(f"\n🏆 Best optimization type: {best_type}")
        
        # return results
        
        pass  # Replace with your implementation
    
    def compare_optimization_results(self, original_prompt: str, optimized_prompt: str,
                                   test_scenarios: List[Dict]) -> ComparisonAnalysis:
        """
        TODO 6: Implement comprehensive optimization results comparison.
        
        Requirements:
        - Analyze both original and optimized prompts comprehensively
        - Measure performance of both prompts with test scenarios
        - Calculate detailed improvement metrics across all dimensions
        - Perform cost-benefit analysis of optimization
        - Generate recommendation score (0-10) for adopting optimization
        - Create actionable recommendations
        
        Comparison areas:
        1. Quality improvements (clarity, specificity, completeness, structure)
        2. Performance improvements (response quality, consistency, speed)
        3. Size improvements (word/token reduction, efficiency gains)
        4. Cost-benefit analysis (benefits vs. costs of optimization)
        
        Args:
            original_prompt: Original prompt text
            optimized_prompt: Optimized prompt text
            test_scenarios: Business scenarios for performance testing
            
        Returns:
            ComparisonAnalysis with comprehensive comparison results
        """
        # TODO 6: Implement comprehensive optimization comparison
        #
        # Step 1: Analyze both prompts
        # print(f"📊 Performing comprehensive comparison analysis...")
        # 
        # original_analysis = self.analyzer.analyze_prompt_quality(original_prompt)
        # original_performance = self.analyzer.measure_baseline_performance(
        #     original_prompt, test_scenarios, num_runs=3
        # )
        # 
        # optimized_analysis = self.analyzer.analyze_prompt_quality(optimized_prompt)
        # optimized_performance = self.analyzer.measure_baseline_performance(
        #     optimized_prompt, test_scenarios, num_runs=3
        # )
        
        # Step 2: Calculate improvement summary
        # improvement_summary = {
        #     "quality_improvements": {
        #         "clarity": optimized_analysis.clarity_score - original_analysis.clarity_score,
        #         "specificity": optimized_analysis.specificity_score - original_analysis.specificity_score,
        #         "completeness": optimized_analysis.completeness_score - original_analysis.completeness_score,
        #         "structure": optimized_analysis.structure_score - original_analysis.structure_score,
        #         "overall": optimized_analysis.overall_score - original_analysis.overall_score
        #     },
        #     "performance_improvements": {
        #         "response_quality": optimized_performance.quality_metrics["overall_quality"] - 
        #                           original_performance.quality_metrics["overall_quality"],
        #         "consistency": optimized_performance.consistency_score - original_performance.consistency_score,
        #         "generation_speed": original_performance.generation_time - optimized_performance.generation_time,
        #         "token_efficiency": original_performance.token_usage["avg_input_tokens"] - 
        #                           optimized_performance.token_usage["avg_input_tokens"]
        #     },
        #     "size_improvements": {
        #         "word_reduction": original_analysis.word_count - optimized_analysis.word_count,
        #         "character_reduction": original_analysis.character_count - optimized_analysis.character_count,
        #         "percentage_reduction": ((original_analysis.word_count - optimized_analysis.word_count) / 
        #                                original_analysis.word_count * 100) if original_analysis.word_count > 0 else 0
        #     }
        # }
        
        # Step 3: Cost-benefit analysis
        # cost_benefit_analysis = self._calculate_cost_benefit(improvement_summary, original_performance, optimized_performance)
        
        # Step 4: Calculate recommendation score
        # recommendation_score = self._calculate_recommendation_score(improvement_summary)
        
        # Step 5: Return ComparisonAnalysis
        # return ComparisonAnalysis(
        #     original_analysis=original_analysis,
        #     optimized_analysis=optimized_analysis,
        #     original_performance=original_performance,
        #     optimized_performance=optimized_performance,
        #     improvement_summary=improvement_summary,
        #     cost_benefit_analysis=cost_benefit_analysis,
        #     recommendation_score=recommendation_score
        # )
        
        pass  # Replace with your implementation
    
    def _calculate_cost_benefit(self, improvements: Dict, 
                              original_perf: PerformanceMetrics,
                              optimized_perf: PerformanceMetrics) -> Dict[str, Any]:
        """Calculate cost-benefit analysis of optimization."""
        # This helper method is provided to guide your implementation
        benefits = {
            "quality_gain": improvements["quality_improvements"]["overall"],
            "consistency_gain": improvements["performance_improvements"]["consistency"],
            "speed_gain": improvements["performance_improvements"]["generation_speed"],
            "token_savings": improvements["size_improvements"]["word_reduction"],
            "cost_savings_per_1k_requests": improvements["performance_improvements"]["token_efficiency"] * 0.0002 * 1000
        }
        
        costs = {
            "potential_quality_loss": max(0, -improvements["quality_improvements"]["overall"]),
            "optimization_time": 30,  # Estimated seconds
            "validation_effort": 300,  # Estimated seconds
            "deployment_risk": 0.1 if improvements["quality_improvements"]["overall"] < 0 else 0.0
        }
        
        benefit_score = benefits["quality_gain"] * 4 + benefits["consistency_gain"] * 2
        cost_score = costs["potential_quality_loss"] * 5 + costs["deployment_risk"] * 3
        net_benefit = benefit_score - cost_score
        
        return {
            "benefits": benefits,
            "costs": costs,
            "net_benefit": net_benefit,
            "recommendation": "High value" if net_benefit > 2 else "Moderate value" if net_benefit > 0.5 else "Low value"
        }
    
    def _calculate_recommendation_score(self, improvements: Dict) -> float:
        """Calculate 0-10 recommendation score for optimization."""
        # This helper method is provided to guide your implementation
        score = 5.0  # Start at neutral
        
        # Quality improvements (±3 points)
        quality_score = improvements["quality_improvements"]["overall"]
        score += quality_score * 3
        
        # Performance improvements (±2 points)
        perf_score = improvements["performance_improvements"]["response_quality"]
        score += perf_score * 2
        
        # Efficiency improvements (±2 points)
        efficiency_score = improvements["size_improvements"]["percentage_reduction"] / 50
        score += min(efficiency_score, 2)
        
        return max(0, min(10, score))
    
    def export_optimization_report(self, comparison: ComparisonAnalysis, 
                                 filename: Optional[str] = None) -> str:
        """Export comprehensive optimization report."""
        
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_report_{timestamp}"
        
        report = f"""
VERTEX AI PROMPT OPTIMIZATION REPORT
{"=" * 60}

EXECUTIVE SUMMARY
Recommendation Score: {comparison.recommendation_score:.1f}/10
Overall Assessment: {comparison.cost_benefit_analysis["recommendation"]}

OPTIMIZATION RESULTS
{"=" * 30}
Quality Improvements:
• Overall Quality: {comparison.improvement_summary["quality_improvements"]["overall"]:+.3f}
• Performance: {comparison.improvement_summary["performance_improvements"]["response_quality"]:+.3f}
• Token Efficiency: {comparison.improvement_summary["performance_improvements"]["token_efficiency"]:+.0f} tokens

Generated by Vertex AI Prompt Optimizer Analysis
Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        with open(f"{filename}.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"📄 Optimization report exported to {filename}.txt")
        return filename


def load_test_scenarios() -> List[Dict]:
    """Load test scenarios from prompt analyzer."""
    from prompt_analyzer import load_test_scenarios
    return load_test_scenarios()


def run_optimization_demonstration():
    """Demonstrate complete optimization workflow."""
    
    PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id")
    if PROJECT_ID == "your-project-id":
        print("❌ Please set PROJECT_ID environment variable")
        return
    
    # Check if TODOs are implemented
    optimizer = VertexPromptOptimizer(PROJECT_ID)
    sample_prompt = "You are a business analyst. Analyze this market opportunity."
    test_scenarios = load_test_scenarios()
    
    # Test TODO 4
    optimization_result = optimizer.optimize_prompt(sample_prompt)
    if optimization_result is None:
        print("❌ TODO 4 not implemented: optimize_prompt")
        return
    
    # Test TODO 5
    systematic_results = optimizer.systematic_optimization(sample_prompt, test_scenarios[:1])
    if systematic_results is None:
        print("❌ TODO 5 not implemented: systematic_optimization")
        return
    
    # Test TODO 6
    if systematic_results:
        best_result = list(systematic_results.values())[0]
        comparison = optimizer.compare_optimization_results(
            sample_prompt, best_result.optimized_prompt, test_scenarios[:1]
        )
        if comparison is None:
            print("❌ TODO 6 not implemented: compare_optimization_results")
            return
    
    print("✅ All TODOs implemented successfully!")
    print("🚀 Vertex AI Prompt Optimization system ready!")


if __name__ == "__main__":
    run_optimization_demonstration()