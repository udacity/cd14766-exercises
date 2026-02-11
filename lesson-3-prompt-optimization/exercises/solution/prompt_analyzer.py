"""
Lesson 3: Prompt Analysis and Baseline - Complete Solution

This module demonstrates comprehensive prompt analysis techniques to establish
baseline performance and identify optimization opportunities.

Learning Objectives:
- Analyze prompt quality systematically
- Measure baseline performance metrics
- Identify optimization targets
- Build foundation for systematic prompt improvement

Author: Noble Ackerson (Udacity)
Date: 2025
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
        TODO 1 SOLUTION: Comprehensive prompt quality assessment.
        
        Analyzes prompt structure, clarity, and identifies optimization opportunities.
        """
        # Basic metrics
        word_count = len(prompt.split())
        character_count = len(prompt)
        
        # Clarity Analysis
        clarity_score = self._analyze_clarity(prompt)
        
        # Specificity Analysis  
        specificity_score = self._analyze_specificity(prompt, prompt_type)
        
        # Completeness Analysis
        completeness_score = self._analyze_completeness(prompt, prompt_type)
        
        # Structure Analysis
        structure_score = self._analyze_structure(prompt)
        
        # Overall score (weighted average)
        overall_score = (
            clarity_score * 0.25 +
            specificity_score * 0.25 +
            completeness_score * 0.25 +
            structure_score * 0.25
        )
        
        # Identify optimization targets
        optimization_targets = self._identify_optimization_targets(
            prompt, clarity_score, specificity_score, completeness_score, structure_score
        )
        
        # Find readability issues
        readability_issues = self._find_readability_issues(prompt)
        
        return PromptAnalysis(
            clarity_score=clarity_score,
            specificity_score=specificity_score,
            completeness_score=completeness_score,
            structure_score=structure_score,
            overall_score=overall_score,
            optimization_targets=optimization_targets,
            word_count=word_count,
            character_count=character_count,
            readability_issues=readability_issues
        )
    
    def _analyze_clarity(self, prompt: str) -> float:
        """Analyze prompt clarity and understandability."""
        score = 0.0
        
        # Check for clear instructions
        instruction_words = ["you are", "your task", "please", "should", "must", "need to"]
        if any(word in prompt.lower() for word in instruction_words):
            score += 0.3
        
        # Check for ambiguous language
        ambiguous_phrases = ["maybe", "perhaps", "might", "could", "sometimes", "possibly"]
        ambiguous_count = sum(1 for phrase in ambiguous_phrases if phrase in prompt.lower())
        if ambiguous_count == 0:
            score += 0.3
        elif ambiguous_count <= 2:
            score += 0.15
        
        # Check sentence length (shorter sentences are clearer)
        sentences = re.split(r'[.!?]+', prompt)
        avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / len([s for s in sentences if s.strip()])
        if avg_sentence_length <= 15:
            score += 0.2
        elif avg_sentence_length <= 25:
            score += 0.1
        
        # Check for jargon without explanation
        jargon_terms = ["roi", "kpi", "tam", "sam", "som", "cagr"]
        explained_jargon = 0
        for term in jargon_terms:
            if term in prompt.lower():
                # Check if it's explained nearby
                if any(explanation in prompt.lower() for explanation in [
                    f"{term} (", f"{term}:", f"{term} -", f"{term} is", f"{term} means"
                ]):
                    explained_jargon += 1
        
        if explained_jargon > 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_specificity(self, prompt: str, prompt_type: str) -> float:
        """Analyze how specific and detailed the prompt is."""
        score = 0.0
        
        # Check for specific roles/expertise
        role_indicators = ["senior", "expert", "specialist", "years", "experience", "background"]
        role_specificity = sum(1 for indicator in role_indicators if indicator in prompt.lower())
        if role_specificity >= 3:
            score += 0.3
        elif role_specificity >= 1:
            score += 0.15
        
        # Check for specific frameworks/methodologies
        frameworks = [
            "porter", "swot", "pestle", "tam", "sam", "som", "roi", "npv", 
            "cagr", "five forces", "bcg matrix", "ansoff matrix"
        ]
        framework_mentions = sum(1 for framework in frameworks if framework in prompt.lower())
        if framework_mentions >= 2:
            score += 0.3
        elif framework_mentions >= 1:
            score += 0.2
        
        # Check for specific output format requirements
        format_indicators = ["format:", "structure:", "include:", "sections:", "organize"]
        if any(indicator in prompt.lower() for indicator in format_indicators):
            score += 0.2
        
        # Check for context-specific details
        context_indicators = ["company", "industry", "market", "customer", "competitor"]
        context_count = sum(1 for indicator in context_indicators if indicator in prompt.lower())
        if context_count >= 3:
            score += 0.2
        elif context_count >= 1:
            score += 0.1
        
        return min(score, 1.0)
    
    def _analyze_completeness(self, prompt: str, prompt_type: str) -> float:
        """Analyze if prompt contains all necessary elements."""
        score = 0.0
        
        # Essential elements for different prompt types
        essential_elements = {
            "persona": ["role:", "expertise:", "communication", "approach"],
            "task": ["task", "goal", "objective", "output"],
            "analysis": ["analyze", "evaluate", "assess", "examine"],
            "general": ["you", "task", "provide", "focus"]
        }
        
        required_elements = essential_elements.get(prompt_type, essential_elements["general"])
        found_elements = sum(1 for element in required_elements if element in prompt.lower())
        completeness_ratio = found_elements / len(required_elements)
        score += completeness_ratio * 0.4
        
        # Check for task context
        if any(context in prompt.lower() for context in ["when", "context", "situation", "scenario"]):
            score += 0.2
        
        # Check for success criteria
        if any(criteria in prompt.lower() for criteria in ["should", "must", "ensure", "include"]):
            score += 0.2
        
        # Check for examples or guidance
        if any(guide in prompt.lower() for guide in ["example", "such as", "including", "like"]):
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_structure(self, prompt: str) -> float:
        """Analyze prompt organization and logical flow."""
        score = 0.0
        
        # Check for clear sections
        section_markers = ["role:", "expertise:", "task:", "objective:", "approach:", "style:"]
        sections_found = sum(1 for marker in section_markers if marker in prompt.lower())
        if sections_found >= 4:
            score += 0.3
        elif sections_found >= 2:
            score += 0.2
        elif sections_found >= 1:
            score += 0.1
        
        # Check for logical flow (introductory sentences)
        flow_indicators = ["you are", "your role", "you will", "you should", "you must"]
        if any(indicator in prompt.lower() for indicator in flow_indicators):
            score += 0.2
        
        # Check for paragraph structure
        paragraphs = prompt.split('\n\n')
        if len(paragraphs) >= 3:
            score += 0.2
        elif len(paragraphs) >= 2:
            score += 0.1
        
        # Check for bullet points or lists
        if '•' in prompt or '-' in prompt or re.search(r'\d+\.', prompt):
            score += 0.15
        
        # Check for proper capitalization and punctuation
        sentences = re.split(r'[.!?]+', prompt)
        properly_capitalized = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
        if properly_capitalized >= len([s for s in sentences if s.strip()]) * 0.8:
            score += 0.15
        
        return min(score, 1.0)
    
    def _identify_optimization_targets(self, prompt: str, clarity: float, 
                                     specificity: float, completeness: float, 
                                     structure: float) -> List[str]:
        """Identify specific areas for optimization."""
        targets = []
        
        if clarity < 0.7:
            targets.append("improve_clarity")
            if len(prompt.split()) > 200:
                targets.append("reduce_verbosity")
            
        if specificity < 0.7:
            targets.append("add_specificity")
            targets.append("include_frameworks")
            
        if completeness < 0.7:
            targets.append("add_context")
            targets.append("clarify_objectives")
            
        if structure < 0.7:
            targets.append("improve_organization")
            targets.append("add_sections")
            
        # Check for common issues
        if "including but not limited to" in prompt:
            targets.append("remove_filler_phrases")
            
        if prompt.count("and") > 10:
            targets.append("simplify_conjunctions")
            
        if len(prompt.split()) > 300:
            targets.append("reduce_length")
            
        return list(set(targets))  # Remove duplicates
    
    def _find_readability_issues(self, prompt: str) -> List[str]:
        """Find specific readability issues."""
        issues = []
        
        # Long sentences
        sentences = re.split(r'[.!?]+', prompt)
        long_sentences = [s for s in sentences if s.strip() and len(s.split()) > 30]
        if long_sentences:
            issues.append(f"long_sentences ({len(long_sentences)} found)")
        
        # Repeated words
        words = prompt.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        overused_words = [word for word, count in word_freq.items() if count > 5 and len(word) > 3]
        if overused_words:
            issues.append(f"overused_words ({', '.join(overused_words[:3])})")
        
        # Passive voice
        passive_indicators = ["is being", "was being", "are being", "were being", "be used", "be done"]
        passive_count = sum(1 for indicator in passive_indicators if indicator in prompt.lower())
        if passive_count > 2:
            issues.append(f"excessive_passive_voice ({passive_count} instances)")
        
        return issues
    
    def measure_baseline_performance(self, prompt: str, test_scenarios: List[Dict], 
                                   num_runs: int = 3) -> PerformanceMetrics:
        """
        TODO 2 SOLUTION: Measure baseline performance with statistical validity.
        
        Tests prompt performance across multiple scenarios and runs.
        """
        all_results = []
        total_tokens = []
        generation_times = []
        response_lengths = []
        
        print(f"Measuring baseline performance with {num_runs} runs per scenario...")
        
        for scenario_idx, scenario in enumerate(test_scenarios):
            print(f"  Testing scenario {scenario_idx + 1}/{len(test_scenarios)}")
            
            scenario_results = []
            for run in range(num_runs):
                try:
                    # Create full prompt with scenario
                    full_prompt = self._create_scenario_prompt(prompt, scenario)
                    
                    # Measure generation time
                    start_time = time.time()
                    
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=full_prompt,
                        config=GenerateContentConfig(
                            temperature=0.3,  # Consistent temperature for baseline
                            max_output_tokens=800,
                            candidate_count=1
                        )
                    )
                    
                    generation_time = time.time() - start_time
                    
                    # Analyze response quality
                    response_text = response.text or ""
                    quality_metrics = self._analyze_response_quality(
                        response_text, scenario.get("expected_elements", [])
                    )

                    # Track metrics
                    token_count = len(full_prompt.split()) * 1.3  # Approximate input tokens
                    response_length = len(response_text.split())
                    
                    result = {
                        "quality_metrics": quality_metrics,
                        "generation_time": generation_time,
                        "token_count": token_count,
                        "response_length": response_length,
                        "scenario": scenario_idx,
                        "run": run
                    }
                    
                    scenario_results.append(result)
                    all_results.append(result)
                    total_tokens.append(token_count)
                    generation_times.append(generation_time)
                    response_lengths.append(response_length)
                    
                except Exception as e:
                    print(f"    Error in run {run + 1}: {str(e)}")
                    continue
        
        if not all_results:
            raise ValueError("No successful test runs completed")
        
        # Calculate aggregate metrics
        avg_quality = {}
        quality_keys = all_results[0]["quality_metrics"].keys()
        for key in quality_keys:
            values = [r["quality_metrics"][key] for r in all_results]
            avg_quality[key] = sum(values) / len(values)
        
        # Calculate consistency (lower standard deviation = higher consistency)
        quality_values = [r["quality_metrics"]["overall_quality"] for r in all_results]
        consistency_score = max(0, 1 - (statistics.stdev(quality_values) if len(quality_values) > 1 else 0))
        
        return PerformanceMetrics(
            quality_metrics=avg_quality,
            token_usage={
                "avg_input_tokens": int(sum(total_tokens) / len(total_tokens)),
                "total_tokens": int(sum(total_tokens)),
                "avg_response_tokens": int(sum(response_lengths) / len(response_lengths))
            },
            generation_time=sum(generation_times) / len(generation_times),
            consistency_score=consistency_score,
            response_length=int(sum(response_lengths) / len(response_lengths)),
            test_runs=len(all_results)
        )
    
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
    
    def _analyze_response_quality(self, response: str, expected_elements: List[str]) -> Dict[str, float]:
        """Analyze quality of generated response."""
        
        # Content quality metrics
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
        
        # Relevance (contains expected elements)
        relevance_score = 0.0
        if expected_elements:
            found_elements = sum(1 for element in expected_elements 
                               if element.lower() in response.lower())
            relevance_score = found_elements / len(expected_elements)
        else:
            # Default business relevance check
            business_terms = ["market", "strategy", "competitive", "analysis", "recommendation"]
            found_terms = sum(1 for term in business_terms if term in response.lower())
            relevance_score = min(found_terms / 3, 1.0)
        
        # Completeness (adequate detail)
        completeness_score = 0.0
        if word_count >= 100:
            completeness_score += 0.4
        if word_count >= 150:
            completeness_score += 0.3
        if "recommendation" in response.lower() or "suggest" in response.lower():
            completeness_score += 0.3
        
        completeness_score = min(completeness_score, 1.0)
        
        # Overall quality (weighted average)
        overall_quality = (
            coherence_score * 0.4 +
            relevance_score * 0.4 +
            completeness_score * 0.2
        )
        
        return {
            "coherence": coherence_score,
            "relevance": relevance_score,
            "completeness": completeness_score,
            "overall_quality": overall_quality
        }
    
    def detect_optimization_opportunities(self, analysis: PromptAnalysis, 
                                        performance: PerformanceMetrics) -> Dict[str, Any]:
        """
        TODO 3 SOLUTION: Detect specific optimization opportunities.
        
        Analyzes quality and performance data to recommend optimizations.
        """
        opportunities = {
            "priority_targets": [],
            "optimization_strategies": [],
            "expected_improvements": {},
            "optimization_urgency": "low"  # low, medium, high
        }
        
        # Analyze quality scores for optimization targets
        if analysis.overall_score < 0.7:
            opportunities["optimization_urgency"] = "high"
            
            if analysis.clarity_score < 0.6:
                opportunities["priority_targets"].append("clarity_improvement")
                opportunities["optimization_strategies"].append(
                    "Use Vertex AI Optimizer to clarify instructions and reduce ambiguous language"
                )
                opportunities["expected_improvements"]["clarity"] = 0.8 - analysis.clarity_score
            
            if analysis.specificity_score < 0.6:
                opportunities["priority_targets"].append("specificity_enhancement")
                opportunities["optimization_strategies"].append(
                    "Add specific frameworks, methodologies, and detailed role definitions"
                )
                opportunities["expected_improvements"]["specificity"] = 0.8 - analysis.specificity_score
            
            if analysis.structure_score < 0.6:
                opportunities["priority_targets"].append("structural_reorganization")
                opportunities["optimization_strategies"].append(
                    "Reorganize content into clear sections with logical flow"
                )
                opportunities["expected_improvements"]["structure"] = 0.8 - analysis.structure_score
                
        elif analysis.overall_score < 0.8:
            opportunities["optimization_urgency"] = "medium"
        
        # Analyze performance metrics
        if performance.quality_metrics["overall_quality"] < 0.7:
            opportunities["priority_targets"].append("response_quality_improvement")
            opportunities["optimization_strategies"].append(
                "Optimize prompt to generate higher quality, more relevant responses"
            )
            opportunities["expected_improvements"]["response_quality"] = (
                0.8 - performance.quality_metrics["overall_quality"]
            )
        
        if performance.consistency_score < 0.8:
            opportunities["priority_targets"].append("consistency_improvement")
            opportunities["optimization_strategies"].append(
                "Refine prompt to reduce variance in response quality across runs"
            )
            opportunities["expected_improvements"]["consistency"] = 0.9 - performance.consistency_score
        
        # Token efficiency opportunities
        if performance.token_usage["avg_input_tokens"] > 1000:
            opportunities["priority_targets"].append("token_efficiency")
            opportunities["optimization_strategies"].append(
                "Reduce prompt length while maintaining quality to improve efficiency"
            )
            opportunities["expected_improvements"]["token_reduction"] = "15-25%"
        
        # Length optimization
        if analysis.word_count > 250:
            opportunities["priority_targets"].append("length_optimization")
            opportunities["optimization_strategies"].append(
                "Use Vertex AI Optimizer to remove redundant language and improve conciseness"
            )
            opportunities["expected_improvements"]["length_reduction"] = f"{analysis.word_count - 200} words"
        
        return opportunities


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
    
    # Sample prompt to analyze (using business analyst persona from Lesson 1)
    sample_prompt = """
Role: You are a Senior Business Analyst with 15+ years of experience in market research and data-driven business strategy.

Expertise: Your specialization includes market sizing analysis, industry trend identification, competitive dynamics assessment, financial modeling, and strategic business planning. You have deep experience across technology, healthcare, financial services, and consumer goods sectors.

Communication Style: Professional, data-driven, and objective. You always support your insights with specific metrics, quantitative evidence, and clear reasoning chains. You present findings in logical sequence with supporting evidence and avoid subjective opinions without data backing.

Analytical Approach: You apply systematic frameworks including TAM/SAM/SOM analysis, industry lifecycle assessment, growth trajectory modeling, customer segmentation analysis, market penetration evaluation, and competitive benchmarking.

Task Context: When analyzing market opportunities, provide quantitative insights with clear reasoning chains. Focus on data-driven insights that inform strategic decision-making. Include specific metrics when possible and structure insights with clear cause-and-effect relationships.
"""
    
    print("🔍 COMPREHENSIVE PROMPT ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = PromptAnalyzer(PROJECT_ID)
    
    # Analyze prompt quality
    print("\n📊 Analyzing Prompt Quality...")
    analysis = analyzer.analyze_prompt_quality(sample_prompt, "persona")
    
    print(f"Quality Analysis Results:")
    print(f"  Overall Score: {analysis.overall_score:.2f}")
    print(f"  Clarity: {analysis.clarity_score:.2f}")
    print(f"  Specificity: {analysis.specificity_score:.2f}")
    print(f"  Completeness: {analysis.completeness_score:.2f}")
    print(f"  Structure: {analysis.structure_score:.2f}")
    print(f"  Word Count: {analysis.word_count}")
    print(f"  Optimization Targets: {', '.join(analysis.optimization_targets)}")
    
    # Measure baseline performance
    print("\n⚡ Measuring Baseline Performance...")
    test_scenarios = load_test_scenarios()
    performance = analyzer.measure_baseline_performance(sample_prompt, test_scenarios)
    
    print(f"Performance Metrics:")
    print(f"  Average Quality: {performance.quality_metrics['overall_quality']:.2f}")
    print(f"  Consistency: {performance.consistency_score:.2f}")
    print(f"  Avg Generation Time: {performance.generation_time:.2f}s")
    print(f"  Avg Input Tokens: {performance.token_usage['avg_input_tokens']}")
    print(f"  Test Runs Completed: {performance.test_runs}")
    
    # Detect optimization opportunities
    print("\n🎯 Detecting Optimization Opportunities...")
    opportunities = analyzer.detect_optimization_opportunities(analysis, performance)
    
    print(f"Optimization Assessment:")
    print(f"  Urgency Level: {opportunities['optimization_urgency'].upper()}")
    print(f"  Priority Targets: {', '.join(opportunities['priority_targets'])}")
    print(f"  Strategies:")
    for i, strategy in enumerate(opportunities['optimization_strategies'], 1):
        print(f"    {i}. {strategy}")
    
    print(f"\n✅ Analysis complete! Ready for optimization in vertex_optimizer.py")


if __name__ == "__main__":
    run_comprehensive_analysis()