"""
Lesson 3: Prompt Analysis and Baseline - Student Template

This module demonstrates comprehensive prompt analysis techniques to establish
baseline performance and identify optimization opportunities.

Learning Objectives:
- Analyze prompt quality systematically
- Measure baseline performance metrics
- Identify optimization targets
- Build foundation for systematic prompt improvement

Complete TODOs 1-13 to implement comprehensive prompt analysis.

Author: [Your Name]
Date: [Current Date]
"""

import os
import re
import time
import statistics
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from google import genai
from google.genai.types import GenerateContentConfig


@dataclass
class PromptAnalysis:
    """Results of prompt quality analysis."""
    clarity_score: float
    specificity_score: float
    completeness_score: float
    structure_score: float
    overall_score: float
    optimization_targets: List[str]
    word_count: int
    character_count: int
    readability_issues: List[str]


@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    quality_metrics: Dict[str, float]
    token_usage: Dict[str, int]
    generation_time: float
    consistency_score: float
    response_length: int
    test_runs: int


class PromptAnalyzer:
    """
    Comprehensive prompt analysis and baseline performance measurement.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize with Vertex AI client."""
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        self.model_name = "gemini-2.5-flash"
        
    def analyze_prompt_quality(self, prompt: str, prompt_type: str = "general") -> PromptAnalysis:
        """
        TODO 1: Implement comprehensive prompt quality assessment.
        
        Requirements:
        - Analyze clarity, specificity, completeness, and structure
        - Calculate overall quality score (weighted average)
        - Identify specific optimization targets
        - Find readability issues
        - Return PromptAnalysis dataclass with all metrics
        
        Scoring guidelines:
        - Each metric should be 0.0 to 1.0
        - Overall score = (clarity + specificity + completeness + structure) / 4
        - Optimization targets should be specific (e.g., "improve_clarity", "reduce_verbosity")
        
        Analysis areas:
        1. Clarity: Clear instructions, unambiguous language, appropriate sentence length
        2. Specificity: Detailed roles, specific frameworks, context requirements
        3. Completeness: All necessary elements present, sufficient context
        4. Structure: Logical organization, proper sections, good flow
        
        Args:
            prompt: The prompt text to analyze
            prompt_type: Type of prompt ("persona", "task", "analysis", "general")
            
        Returns:
            PromptAnalysis with all quality metrics and recommendations
        """
        # TODO 1: Implement prompt quality analysis
        #
        # Step 1: Calculate basic metrics
        # word_count = len(prompt.split())
        # character_count = len(prompt)
        
        # Step 2: Analyze each quality dimension
        # clarity_score = self._analyze_clarity(prompt)
        # specificity_score = self._analyze_specificity(prompt, prompt_type)
        # completeness_score = self._analyze_completeness(prompt, prompt_type)
        # structure_score = self._analyze_structure(prompt)
        
        # Step 3: Calculate overall score (weighted average)
        # overall_score = (clarity_score + specificity_score + completeness_score + structure_score) / 4
        
        # Step 4: Identify optimization targets
        # optimization_targets = self._identify_optimization_targets(prompt, clarity_score, ...)
        
        # Step 5: Find readability issues
        # readability_issues = self._find_readability_issues(prompt)
        
        # Step 6: Return PromptAnalysis dataclass
        # return PromptAnalysis(...)
        
        pass  # Replace with your implementation
    
    def _analyze_clarity(self, prompt: str) -> float:
        """
        Analyze prompt clarity and understandability.
        
        This helper method is provided to guide your implementation.
        Check for:
        - Clear instruction words ("you are", "your task", "please")
        - Absence of ambiguous language ("maybe", "perhaps", "might")
        - Appropriate sentence length (≤15 words is good, ≤25 is acceptable)
        - Explained jargon terms
        """
        score = 0.0
        
        # Check for clear instructions
        instruction_words = ["you are", "your task", "please", "should", "must", "need to"]
        if any(word in prompt.lower() for word in instruction_words):
            score += 0.3
        
        # Check for ambiguous language (fewer is better)
        ambiguous_phrases = ["maybe", "perhaps", "might", "could", "sometimes", "possibly"]
        ambiguous_count = sum(1 for phrase in ambiguous_phrases if phrase in prompt.lower())
        if ambiguous_count == 0:
            score += 0.3
        elif ambiguous_count <= 2:
            score += 0.15
        
        # Add more clarity checks here...
        
        return min(score, 1.0)
    
    def _analyze_specificity(self, prompt: str, prompt_type: str) -> float:
        """Analyze how specific and detailed the prompt is."""
        # TODO: Implement specificity analysis
        # Check for: specific roles, frameworks mentioned, output format, context details
        pass
    
    def _analyze_completeness(self, prompt: str, prompt_type: str) -> float:
        """Analyze if prompt contains all necessary elements."""
        # TODO: Implement completeness analysis
        # Check for: essential elements by type, task context, success criteria, examples
        pass
    
    def _analyze_structure(self, prompt: str) -> float:
        """Analyze prompt organization and logical flow."""
        # TODO: Implement structure analysis
        # Check for: clear sections, logical flow, paragraphs, lists, proper formatting
        pass
    
    def _identify_optimization_targets(self, prompt: str, clarity: float, 
                                     specificity: float, completeness: float, 
                                     structure: float) -> List[str]:
        """Identify specific areas for optimization."""
        # TODO: Implement optimization target identification
        # Based on low scores, identify specific improvement areas
        pass
    
    def _find_readability_issues(self, prompt: str) -> List[str]:
        """Find specific readability issues."""
        # TODO: Implement readability issue detection
        # Check for: long sentences, repeated words, passive voice
        pass
    
    def measure_baseline_performance(self, prompt: str, test_scenarios: List[Dict], 
                                   num_runs: int = 3) -> PerformanceMetrics:
        """
        TODO 2: Measure baseline performance with statistical validity.
        
        Requirements:
        - Test prompt with multiple scenarios and runs for consistency
        - Measure quality metrics, token usage, generation time
        - Calculate consistency score (lower variance = higher consistency)
        - Return PerformanceMetrics with comprehensive data
        
        Performance areas to measure:
        1. Quality metrics: Response coherence, relevance, completeness
        2. Token usage: Input tokens, output tokens, efficiency
        3. Generation time: Average response time across runs
        4. Consistency: Variance in quality across multiple runs
        
        Args:
            prompt: The prompt to test
            test_scenarios: List of business scenarios for testing
            num_runs: Number of runs per scenario for statistical validity
            
        Returns:
            PerformanceMetrics with comprehensive performance data
        """
        # TODO 2: Implement baseline performance measurement
        #
        # Step 1: Initialize tracking variables
        # all_results = []
        # total_tokens = []
        # generation_times = []
        # response_lengths = []
        
        # Step 2: Loop through scenarios and runs
        # for scenario in test_scenarios:
        #     for run in range(num_runs):
        #         # Create full prompt with scenario
        #         full_prompt = self._create_scenario_prompt(prompt, scenario)
        #         
        #         # Measure generation time
        #         start_time = time.time()
        #         response = self.client.models.generate_content(...)
        #         generation_time = time.time() - start_time
        #         
        #         # Analyze response quality
        #         quality_metrics = self._analyze_response_quality(response.text, scenario)
        #         
        #         # Track all metrics
        #         all_results.append({...})
        
        # Step 3: Calculate aggregate metrics
        # - Average quality across all runs
        # - Token usage statistics
        # - Consistency score (1 - standard_deviation)
        
        # Step 4: Return PerformanceMetrics
        # return PerformanceMetrics(...)
        
        pass  # Replace with your implementation
    
    def _create_scenario_prompt(self, prompt: str, scenario: Dict) -> str:
        """Create full prompt combining persona with scenario."""
        return f"""{prompt}

Business Scenario:
Company: {scenario.get('company_name', 'Test Company')}
Industry: {scenario.get('industry', 'Technology')}
Market Focus: {scenario.get('market_focus', 'general market')}
Strategic Question: {scenario.get('strategic_question', 'What should be our next strategic move?')}
Context: {scenario.get('additional_context', 'Limited additional context provided.')}

Based on your expertise, provide your analysis of this business scenario.
"""
    
    def _analyze_response_quality(self, response: str, scenario: Dict) -> Dict[str, float]:
        """
        Analyze quality of generated response.
        
        This helper method is provided to guide your implementation.
        """
        word_count = len(response.split())
        
        # Coherence (logical flow and structure)
        coherence_score = 0.0
        if word_count >= 50:
            coherence_score += 0.3
        
        structure_indicators = ["first", "second", "third", "therefore", "however", "additionally"]
        structure_count = sum(1 for indicator in structure_indicators if indicator in response.lower())
        coherence_score += min(structure_count / 3, 0.4)
        
        if len(response.split('\n\n')) >= 2:  # Multiple paragraphs
            coherence_score += 0.3
        
        coherence_score = min(coherence_score, 1.0)
        
        # Add relevance and completeness analysis...
        
        return {
            "coherence": coherence_score,
            "relevance": 0.7,  # Placeholder - implement relevance analysis
            "completeness": 0.7,  # Placeholder - implement completeness analysis
            "overall_quality": coherence_score * 0.4 + 0.7 * 0.6  # Weighted average
        }
    
    def detect_optimization_opportunities(self, analysis: PromptAnalysis, 
                                        performance: PerformanceMetrics) -> Dict[str, Any]:
        """
        TODO 3: Detect specific optimization opportunities.
        
        Requirements:
        - Analyze quality scores and performance metrics
        - Identify priority optimization targets
        - Suggest specific optimization strategies
        - Calculate expected improvements
        - Determine optimization urgency (low/medium/high)
        
        Analysis areas:
        1. Quality gaps: Where scores are below thresholds (0.7, 0.8)
        2. Performance issues: Poor consistency, slow generation, high token usage
        3. Optimization strategies: Specific techniques to address issues
        4. Expected improvements: Quantitative estimates of potential gains
        
        Args:
            analysis: PromptAnalysis results from analyze_prompt_quality
            performance: PerformanceMetrics from measure_baseline_performance
            
        Returns:
            Dictionary with optimization opportunities analysis
        """
        # TODO 3: Implement optimization opportunity detection
        #
        # Step 1: Initialize opportunities structure
        # opportunities = {
        #     "priority_targets": [],
        #     "optimization_strategies": [],
        #     "expected_improvements": {},
        #     "optimization_urgency": "low"  # low, medium, high
        # }
        
        # Step 2: Analyze quality scores for optimization targets
        # if analysis.overall_score < 0.7:
        #     opportunities["optimization_urgency"] = "high"
        #     
        #     if analysis.clarity_score < 0.6:
        #         opportunities["priority_targets"].append("clarity_improvement")
        #         opportunities["optimization_strategies"].append("Use Vertex AI Optimizer to clarify instructions")
        #         opportunities["expected_improvements"]["clarity"] = 0.8 - analysis.clarity_score
        
        # Step 3: Analyze performance metrics
        # if performance.quality_metrics["overall_quality"] < 0.7:
        #     opportunities["priority_targets"].append("response_quality_improvement")
        
        # Step 4: Check for efficiency opportunities
        # if performance.token_usage["avg_input_tokens"] > 1000:
        #     opportunities["priority_targets"].append("token_efficiency")
        
        # Step 5: Return complete analysis
        # return opportunities
        
        pass  # Replace with your implementation


def load_test_scenarios() -> List[Dict]:
    """Load business scenarios for testing."""
    return [
        {
            "company_name": "TechFlow Solutions",
            "industry": "Software Technology",
            "market_focus": "enterprise workflow automation",
            "strategic_question": "Should we expand into small business markets or focus on enterprise growth?",
            "additional_context": "Strong enterprise presence, limited SMB experience, considering platform simplification.",
            "expected_elements": ["market", "competitive", "strategy", "recommendation"]
        },
        {
            "company_name": "GreenEnergy Corp",
            "industry": "Renewable Energy",
            "market_focus": "commercial solar installations", 
            "strategic_question": "How should we respond to increased competition in the commercial solar market?",
            "additional_context": "Market leader for 5 years, new competitors entering with lower prices.",
            "expected_elements": ["competitive", "positioning", "pricing", "differentiation"]
        },
        {
            "company_name": "DataInsight Analytics",
            "industry": "Business Intelligence",
            "market_focus": "predictive analytics for retail",
            "strategic_question": "What new product features should we prioritize for next quarter?",
            "additional_context": "Growing customer base requesting AI features, limited development resources.",
            "expected_elements": ["prioritization", "resource", "customer", "development"]
        }
    ]


def run_comprehensive_analysis():
    """Demonstrate comprehensive prompt analysis workflow."""
    
    PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id")
    if PROJECT_ID == "your-project-id":
        print("❌ Please set PROJECT_ID environment variable")
        return
    
    # Check if TODOs are implemented
    analyzer = PromptAnalyzer(PROJECT_ID)
    sample_prompt = "You are a business analyst. Analyze this market opportunity."
    
    # Test TODO 1
    analysis = analyzer.analyze_prompt_quality(sample_prompt)
    if analysis is None:
        print("❌ TODO 1 not implemented: analyze_prompt_quality")
        return
    
    # Test TODO 2
    test_scenarios = load_test_scenarios()
    performance = analyzer.measure_baseline_performance(sample_prompt, test_scenarios[:1], num_runs=1)
    if performance is None:
        print("❌ TODO 2 not implemented: measure_baseline_performance")
        return
    
    # Test TODO 3
    opportunities = analyzer.detect_optimization_opportunities(analysis, performance)
    if opportunities is None:
        print("❌ TODO 3 not implemented: detect_optimization_opportunities")
        return
    
    print("✅ All TODOs implemented successfully!")
    print("🎯 Ready for vertex_optimizer.py!")


if __name__ == "__main__":
    run_comprehensive_analysis()