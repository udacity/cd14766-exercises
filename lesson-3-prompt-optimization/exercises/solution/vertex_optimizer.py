"""
Lesson 3: Vertex AI Optimizer Integration - Complete Solution

This module demonstrates comprehensive integration with Vertex AI Prompt Optimizer
to systematically improve prompt performance with measurable results.

Learning Objectives:
- Use Vertex AI Prompt Optimizer API
- Apply systematic optimization techniques
- Compare before/after performance with detailed analysis
- Build production-ready optimization workflows

Author: Noble Ackerson (Udacity)
Date: 2025
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
        TODO 4 SOLUTION: Complete Vertex AI Optimizer integration.
        
        Uses Vertex AI Prompt Optimizer to systematically improve prompts.
        """
        if target_model is None:
            target_model = self.model_name
            
        print(f"🔧 Optimizing prompt using Vertex AI Optimizer...")
        print(f"  Type: {optimization_type}")
        print(f"  Steps: {num_steps}")
        print(f"  Target Model: {target_model}")
        
        start_time = time.time()
        
        try:
            # Call Vertex AI Prompt Optimizer
            optimization_response = self.client.prompt_optimizer.optimize_prompt(
                prompt=prompt,
                optimization_config={
                    "num_steps": num_steps,
                    "target_model": target_model,
                    "optimization_mode": optimization_type,
                    "evaluation_metrics": ["coherence", "fluency", "groundedness"]
                }
            )
            
            optimization_time = time.time() - start_time
            
            # Extract optimization results
            optimized_prompt = optimization_response.suggested_prompt
            applied_guidelines = [
                {
                    "guideline": guideline.applicable_guideline,
                    "description": getattr(guideline, 'description', ''),
                    "impact": getattr(guideline, 'impact_score', 0.0)
                }
                for guideline in optimization_response.applicable_guidelines
            ]
            
            # Calculate basic improvements
            original_length = len(prompt.split())
            optimized_length = len(optimized_prompt.split())
            length_reduction = original_length - optimized_length
            
            # Analyze quality improvements
            quality_improvement = self._measure_quality_improvement(
                prompt, optimized_prompt
            )
            
            # Calculate token savings
            token_savings = {
                "input_tokens_saved": int(length_reduction * 1.3),  # Approximate
                "percentage_reduction": (length_reduction / original_length) * 100 if original_length > 0 else 0
            }
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                applied_guidelines, quality_improvement, token_savings
            )
            
            print(f"  ✅ Optimization completed in {optimization_time:.2f}s")
            print(f"  📊 Applied {len(applied_guidelines)} optimization guidelines")
            print(f"  📉 Length reduction: {length_reduction} words ({token_savings['percentage_reduction']:.1f}%)")
            
            return OptimizationResult(
                original_prompt=prompt,
                optimized_prompt=optimized_prompt,
                optimization_type=optimization_type,
                improvement_metrics={
                    "length_reduction": length_reduction,
                    "percentage_reduction": token_savings["percentage_reduction"],
                    "guidelines_applied": len(applied_guidelines)
                },
                applied_guidelines=applied_guidelines,
                optimization_time=optimization_time,
                token_savings=token_savings,
                quality_improvement=quality_improvement,
                recommendation=recommendation
            )
            
        except Exception as e:
            print(f"❌ Optimization failed: {str(e)}")
            
            # Return fallback result
            return OptimizationResult(
                original_prompt=prompt,
                optimized_prompt=prompt,  # No change
                optimization_type=optimization_type,
                improvement_metrics={"error": str(e)},
                applied_guidelines=[],
                optimization_time=time.time() - start_time,
                token_savings={"input_tokens_saved": 0, "percentage_reduction": 0},
                quality_improvement={},
                recommendation="Optimization failed - manual review recommended"
            )
    
    def _measure_quality_improvement(self, original: str, optimized: str) -> Dict[str, float]:
        """Measure quality improvement between prompts."""
        
        # Quick quality assessment without full API calls
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
        
        if not guidelines:
            return "No optimizations applied - prompt may already be well-optimized"
        
        # Analyze improvement significance
        significant_improvements = [
            metric for metric, value in quality_improvement.items() 
            if value > 0.1
        ]
        
        if len(significant_improvements) >= 2 and token_savings["percentage_reduction"] > 10:
            return "Strongly recommended - significant quality and efficiency improvements"
        elif significant_improvements and token_savings["percentage_reduction"] > 5:
            return "Recommended - good balance of quality and efficiency gains"
        elif significant_improvements:
            return "Consider adoption - quality improvements with minimal efficiency change"
        elif token_savings["percentage_reduction"] > 15:
            return "Consider for efficiency - significant token savings with maintained quality"
        else:
            return "Optional - minor improvements, evaluate based on specific needs"
    
    def systematic_optimization(self, prompt: str, test_scenarios: List[Dict],
                              optimization_types: List[str] = None) -> Dict[str, OptimizationResult]:
        """
        TODO 5 SOLUTION: Systematic multi-type optimization workflow.
        
        Tests different optimization approaches and finds the best one.
        """
        if optimization_types is None:
            optimization_types = ["instructions", "demonstrations", "both"]
        
        print(f"🔄 Running systematic optimization...")
        print(f"  Testing {len(optimization_types)} optimization types")
        print(f"  Using {len(test_scenarios)} test scenarios")
        
        results = {}
        
        for opt_type in optimization_types:
            print(f"\n  📋 Testing optimization type: {opt_type}")
            
            try:
                # Optimize prompt
                result = self.optimize_prompt(prompt, opt_type)
                
                # Test performance with scenarios
                print(f"    Testing performance with scenarios...")
                original_performance = self.analyzer.measure_baseline_performance(
                    prompt, test_scenarios, num_runs=2  # Reduced runs for efficiency
                )
                optimized_performance = self.analyzer.measure_baseline_performance(
                    result.optimized_prompt, test_scenarios, num_runs=2
                )
                
                # Calculate performance improvement
                performance_improvement = {
                    "quality_improvement": (
                        optimized_performance.quality_metrics["overall_quality"] - 
                        original_performance.quality_metrics["overall_quality"]
                    ),
                    "consistency_improvement": (
                        optimized_performance.consistency_score - 
                        original_performance.consistency_score
                    ),
                    "speed_improvement": (
                        original_performance.generation_time - 
                        optimized_performance.generation_time
                    ),
                    "token_efficiency": (
                        original_performance.token_usage["avg_input_tokens"] - 
                        optimized_performance.token_usage["avg_input_tokens"]
                    )
                }
                
                # Update result with performance data
                result.quality_improvement.update(performance_improvement)
                results[opt_type] = result
                
                print(f"    ✅ Quality improvement: {performance_improvement['quality_improvement']:+.3f}")
                print(f"    ⚡ Speed improvement: {performance_improvement['speed_improvement']:+.2f}s")
                
            except Exception as e:
                print(f"    ❌ Failed optimization type {opt_type}: {str(e)}")
                continue
        
        # Find best optimization
        if results:
            best_type = max(results.keys(), 
                          key=lambda k: results[k].quality_improvement.get("overall_improvement", 0))
            print(f"\n🏆 Best optimization type: {best_type}")
            print(f"  Overall improvement: {results[best_type].quality_improvement.get('overall_improvement', 0):.3f}")
        
        return results
    
    def compare_optimization_results(self, original_prompt: str, optimized_prompt: str,
                                   test_scenarios: List[Dict]) -> ComparisonAnalysis:
        """
        TODO 6 SOLUTION: Comprehensive optimization results comparison.
        
        Provides detailed analysis of optimization effectiveness.
        """
        print(f"📊 Performing comprehensive comparison analysis...")
        
        # Analyze both prompts
        print("  Analyzing original prompt...")
        original_analysis = self.analyzer.analyze_prompt_quality(original_prompt)
        original_performance = self.analyzer.measure_baseline_performance(
            original_prompt, test_scenarios, num_runs=3
        )
        
        print("  Analyzing optimized prompt...")
        optimized_analysis = self.analyzer.analyze_prompt_quality(optimized_prompt)
        optimized_performance = self.analyzer.measure_baseline_performance(
            optimized_prompt, test_scenarios, num_runs=3
        )
        
        # Calculate improvement summary
        improvement_summary = {
            "quality_improvements": {
                "clarity": optimized_analysis.clarity_score - original_analysis.clarity_score,
                "specificity": optimized_analysis.specificity_score - original_analysis.specificity_score,
                "completeness": optimized_analysis.completeness_score - original_analysis.completeness_score,
                "structure": optimized_analysis.structure_score - original_analysis.structure_score,
                "overall": optimized_analysis.overall_score - original_analysis.overall_score
            },
            "performance_improvements": {
                "response_quality": (
                    optimized_performance.quality_metrics["overall_quality"] - 
                    original_performance.quality_metrics["overall_quality"]
                ),
                "consistency": (
                    optimized_performance.consistency_score - 
                    original_performance.consistency_score
                ),
                "generation_speed": (
                    original_performance.generation_time - 
                    optimized_performance.generation_time
                ),
                "token_efficiency": (
                    original_performance.token_usage["avg_input_tokens"] - 
                    optimized_performance.token_usage["avg_input_tokens"]
                )
            },
            "size_improvements": {
                "word_reduction": original_analysis.word_count - optimized_analysis.word_count,
                "character_reduction": original_analysis.character_count - optimized_analysis.character_count,
                "percentage_reduction": (
                    (original_analysis.word_count - optimized_analysis.word_count) / 
                    original_analysis.word_count * 100 if original_analysis.word_count > 0 else 0
                )
            }
        }
        
        # Cost-benefit analysis
        cost_benefit_analysis = self._calculate_cost_benefit(
            improvement_summary, original_performance, optimized_performance
        )
        
        # Calculate recommendation score
        recommendation_score = self._calculate_recommendation_score(improvement_summary)
        
        print(f"  ✅ Comparison analysis complete")
        print(f"  📈 Overall quality improvement: {improvement_summary['quality_improvements']['overall']:+.3f}")
        print(f"  🚀 Performance improvement: {improvement_summary['performance_improvements']['response_quality']:+.3f}")
        print(f"  💰 Token savings: {improvement_summary['size_improvements']['word_reduction']} words")
        print(f"  🎯 Recommendation score: {recommendation_score:.2f}/10")
        
        return ComparisonAnalysis(
            original_analysis=original_analysis,
            optimized_analysis=optimized_analysis,
            original_performance=original_performance,
            optimized_performance=optimized_performance,
            improvement_summary=improvement_summary,
            cost_benefit_analysis=cost_benefit_analysis,
            recommendation_score=recommendation_score
        )
    
    def _calculate_cost_benefit(self, improvements: Dict, 
                              original_perf: PerformanceMetrics,
                              optimized_perf: PerformanceMetrics) -> Dict[str, Any]:
        """Calculate cost-benefit analysis of optimization."""
        
        # Benefits (positive impacts)
        benefits = {
            "quality_gain": improvements["quality_improvements"]["overall"],
            "consistency_gain": improvements["performance_improvements"]["consistency"],
            "speed_gain": improvements["performance_improvements"]["generation_speed"],
            "token_savings": improvements["size_improvements"]["word_reduction"],
            "cost_savings_per_1k_requests": (
                improvements["performance_improvements"]["token_efficiency"] * 0.0002 * 1000
            )  # Approximate cost savings
        }
        
        # Costs (negative impacts or risks)
        costs = {
            "potential_quality_loss": max(0, -improvements["quality_improvements"]["overall"]),
            "optimization_time": 30,  # Estimated seconds for optimization process
            "validation_effort": 300,  # Estimated seconds for validation
            "deployment_risk": 0.1 if improvements["quality_improvements"]["overall"] < 0 else 0.0
        }
        
        # Overall benefit score
        benefit_score = (
            benefits["quality_gain"] * 4 +
            benefits["consistency_gain"] * 2 +
            benefits["speed_gain"] * 1 +
            (benefits["token_savings"] / 100) * 1  # Normalize token savings
        )
        
        cost_score = (
            costs["potential_quality_loss"] * 5 +
            costs["deployment_risk"] * 3
        )
        
        net_benefit = benefit_score - cost_score
        
        return {
            "benefits": benefits,
            "costs": costs,
            "benefit_score": benefit_score,
            "cost_score": cost_score,
            "net_benefit": net_benefit,
            "recommendation": (
                "High value" if net_benefit > 2 else
                "Moderate value" if net_benefit > 0.5 else
                "Low value" if net_benefit > 0 else
                "Not recommended"
            )
        }
    
    def _calculate_recommendation_score(self, improvements: Dict) -> float:
        """Calculate 0-10 recommendation score for optimization."""
        
        score = 5.0  # Start at neutral
        
        # Quality improvements (±3 points)
        quality_score = improvements["quality_improvements"]["overall"]
        score += quality_score * 3
        
        # Performance improvements (±2 points)
        perf_score = improvements["performance_improvements"]["response_quality"]
        score += perf_score * 2
        
        # Efficiency improvements (±2 points)
        efficiency_score = improvements["size_improvements"]["percentage_reduction"] / 50  # Normalize
        score += min(efficiency_score, 2)
        
        # Consistency improvements (±1 point)
        consistency_score = improvements["performance_improvements"]["consistency"]
        score += consistency_score * 1
        
        # Clamp to 0-10 range
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
• Clarity: {comparison.improvement_summary["quality_improvements"]["clarity"]:+.3f}
• Specificity: {comparison.improvement_summary["quality_improvements"]["specificity"]:+.3f}
• Completeness: {comparison.improvement_summary["quality_improvements"]["completeness"]:+.3f}
• Structure: {comparison.improvement_summary["quality_improvements"]["structure"]:+.3f}
• Overall Quality: {comparison.improvement_summary["quality_improvements"]["overall"]:+.3f}

Performance Improvements:
• Response Quality: {comparison.improvement_summary["performance_improvements"]["response_quality"]:+.3f}
• Consistency: {comparison.improvement_summary["performance_improvements"]["consistency"]:+.3f}
• Generation Speed: {comparison.improvement_summary["performance_improvements"]["generation_speed"]:+.2f}s
• Token Efficiency: {comparison.improvement_summary["performance_improvements"]["token_efficiency"]:+.0f} tokens

Size Optimization:
• Word Reduction: {comparison.improvement_summary["size_improvements"]["word_reduction"]} words
• Percentage Reduction: {comparison.improvement_summary["size_improvements"]["percentage_reduction"]:.1f}%

COST-BENEFIT ANALYSIS
{"=" * 30}

Benefits:
• Quality Gain: {comparison.cost_benefit_analysis["benefits"]["quality_gain"]:.3f}
• Token Savings: {comparison.cost_benefit_analysis["benefits"]["token_savings"]} words
• Cost Savings (per 1K requests): ${comparison.cost_benefit_analysis["benefits"]["cost_savings_per_1k_requests"]:.4f}

Net Benefit Score: {comparison.cost_benefit_analysis["net_benefit"]:.2f}

DETAILED ANALYSIS
{"=" * 30}

Original Prompt Analysis:
• Word Count: {comparison.original_analysis.word_count}
• Overall Score: {comparison.original_analysis.overall_score:.3f}
• Optimization Targets: {", ".join(comparison.original_analysis.optimization_targets)}

Optimized Prompt Analysis:
• Word Count: {comparison.optimized_analysis.word_count}
• Overall Score: {comparison.optimized_analysis.overall_score:.3f}
• Optimization Targets: {", ".join(comparison.optimized_analysis.optimization_targets)}

Original Performance:
• Quality: {comparison.original_performance.quality_metrics["overall_quality"]:.3f}
• Consistency: {comparison.original_performance.consistency_score:.3f}
• Avg Generation Time: {comparison.original_performance.generation_time:.2f}s
• Avg Input Tokens: {comparison.original_performance.token_usage["avg_input_tokens"]}

Optimized Performance:
• Quality: {comparison.optimized_performance.quality_metrics["overall_quality"]:.3f}
• Consistency: {comparison.optimized_performance.consistency_score:.3f}
• Avg Generation Time: {comparison.optimized_performance.generation_time:.2f}s
• Avg Input Tokens: {comparison.optimized_performance.token_usage["avg_input_tokens"]}

RECOMMENDATION
{"=" * 30}
{comparison.cost_benefit_analysis["recommendation"]}

Based on the analysis, this optimization provides a net benefit score of {comparison.cost_benefit_analysis["net_benefit"]:.2f}.
The recommendation score is {comparison.recommendation_score:.1f}/10.

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
    
    # Sample prompt to optimize (business analyst from Lesson 1)
    sample_prompt = """
Role: You are a Senior Business Analyst with 15+ years of experience in market research and data-driven business strategy.

Expertise: Your specialization includes market sizing analysis, industry trend identification, competitive dynamics assessment, financial modeling, and strategic business planning. You have deep experience across technology, healthcare, financial services, and consumer goods sectors.

Communication Style: Professional, data-driven, and objective. You always support your insights with specific metrics, quantitative evidence, and clear reasoning chains. You present findings in logical sequence with supporting evidence and avoid subjective opinions without data backing.

Analytical Approach: You apply systematic frameworks including TAM/SAM/SOM analysis, industry lifecycle assessment, growth trajectory modeling, customer segmentation analysis, market penetration evaluation, and competitive benchmarking.

Task Context: When analyzing market opportunities, provide quantitative insights with clear reasoning chains. Focus on data-driven insights that inform strategic decision-making. Include specific metrics when possible and structure insights with clear cause-and-effect relationships.
"""
    
    print("🚀 VERTEX AI PROMPT OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = VertexPromptOptimizer(PROJECT_ID)
    test_scenarios = load_test_scenarios()
    
    # Single optimization
    print("\n🔧 Single Optimization Test")
    optimization_result = optimizer.optimize_prompt(sample_prompt, "instructions")
    
    print(f"\nOptimization Summary:")
    print(f"  Guidelines Applied: {len(optimization_result.applied_guidelines)}")
    print(f"  Length Reduction: {optimization_result.improvement_metrics['length_reduction']} words")
    print(f"  Recommendation: {optimization_result.recommendation}")
    
    # Systematic optimization
    print("\n🔄 Systematic Optimization Test")
    systematic_results = optimizer.systematic_optimization(sample_prompt, test_scenarios[:2])  # Limit for demo
    
    # Comprehensive comparison
    if systematic_results:
        best_result = max(systematic_results.values(), 
                         key=lambda r: r.quality_improvement.get("overall_improvement", 0))
        
        print("\n📊 Comprehensive Comparison Analysis")
        comparison = optimizer.compare_optimization_results(
            sample_prompt, 
            best_result.optimized_prompt, 
            test_scenarios[:1]  # Limit for demo
        )
        
        # Export report
        report_filename = optimizer.export_optimization_report(comparison)
        
        print(f"\n✅ Optimization demonstration complete!")
        print(f"📄 Detailed report saved to {report_filename}.txt")
    else:
        print("\n❌ No successful optimizations completed")


if __name__ == "__main__":
    run_optimization_demonstration()