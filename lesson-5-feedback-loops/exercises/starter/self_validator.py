"""
Lesson 5: Self-Validation and Error Detection - Student Template

This module demonstrates advanced self-validation systems with automated error
detection, confidence scoring, and intelligent retry mechanisms for production AI systems.

Learning Objectives:
- Build comprehensive self-assessment capabilities
- Implement automated error detection and quality validation
- Create intelligent retry mechanisms with learning integration
- Develop quality gate systems with escalation workflows

Complete TODOs 1-27 to implement comprehensive self-validation functionality.

Author: [Your Name]
Date: [Current Date]
"""

import os
import time
import json
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from google import genai
from google.genai.types import GenerateContentConfig

# Import previous lesson components for integration
import sys
sys.path.append("../../lesson-4-prompt-chaining/exercises/solution")
sys.path.append("../../lesson-3-prompt-optimization/exercises/solution")

try:
    from sequential_chain import ChainContext, ChainStep, StepResult
    from bi_chain_agent import BusinessScenario, BIReport
    LESSON_4_AVAILABLE = True
except ImportError:
    print("⚠️  Lesson 4 components not available - using fallback")
    LESSON_4_AVAILABLE = False

try:
    from vertex_optimizer import VertexPromptOptimizer
    LESSON_3_AVAILABLE = True
except ImportError:
    print("⚠️  Lesson 3 optimizer not available - using fallback")
    LESSON_3_AVAILABLE = False


class ValidationLevel(Enum):
    """Validation strictness levels."""
    PERMISSIVE = "permissive"    # 0.6 threshold
    STANDARD = "standard"        # 0.75 threshold  
    STRICT = "strict"           # 0.85 threshold
    CRITICAL = "critical"       # 0.95 threshold


class ConfidenceType(Enum):
    """Types of confidence assessments."""
    CONTENT_QUALITY = "content_quality"
    CONTEXT_RELEVANCE = "context_relevance"
    FACTUAL_ACCURACY = "factual_accuracy"
    STRUCTURAL_COHERENCE = "structural_coherence"
    COMPLETENESS = "completeness"


class RetryStrategy(Enum):
    """Retry strategies for failed validations."""
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    PROMPT_ENHANCEMENT = "prompt_enhancement"
    CONTEXT_ENRICHMENT = "context_enrichment"
    ALTERNATIVE_APPROACH = "alternative_approach"
    ESCALATION = "escalation"


@dataclass
class ConfidenceScore:
    """Multi-dimensional confidence assessment."""
    content_quality: float
    context_relevance: float
    factual_accuracy: float
    structural_coherence: float
    completeness: float
    overall_confidence: float
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    @classmethod
    def weighted_average(cls, scores: List[Tuple['ConfidenceScore', float]]) -> 'ConfidenceScore':
        """Calculate weighted average of multiple confidence scores."""
        total_weight = sum(weight for _, weight in scores)
        
        weighted_content = sum(score.content_quality * weight for score, weight in scores) / total_weight
        weighted_context = sum(score.context_relevance * weight for score, weight in scores) / total_weight
        weighted_factual = sum(score.factual_accuracy * weight for score, weight in scores) / total_weight
        weighted_structural = sum(score.structural_coherence * weight for score, weight in scores) / total_weight
        weighted_completeness = sum(score.completeness * weight for score, weight in scores) / total_weight
        
        overall = (weighted_content + weighted_context + weighted_factual + 
                  weighted_structural + weighted_completeness) / 5
        
        return cls(
            content_quality=weighted_content,
            context_relevance=weighted_context,
            factual_accuracy=weighted_factual,
            structural_coherence=weighted_structural,
            completeness=weighted_completeness,
            overall_confidence=overall
        )


@dataclass
class ValidationIssue:
    """Identified validation issue with details."""
    issue_type: str
    severity: str  # low, medium, high, critical
    description: str
    location: str  # where in the content
    suggestion: str
    confidence: float


@dataclass
class ValidationContext:
    """Context for validation operations."""
    original_prompt: str
    expected_elements: List[str]
    business_scenario: Optional[Dict] = None
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    previous_attempts: List[Dict] = None
    timeout_seconds: int = 30
    
    def __post_init__(self):
        if self.previous_attempts is None:
            self.previous_attempts = []


@dataclass
class ValidationResult:
    """Comprehensive validation result."""
    overall_confidence: float
    confidence_breakdown: ConfidenceScore
    detected_issues: List[ValidationIssue]
    quality_metrics: Dict[str, float]
    improvement_recommendations: List[str]
    validation_passed: bool
    validation_level: ValidationLevel
    processing_time: float
    retry_recommended: bool = False
    retry_strategy: Optional[RetryStrategy] = None


@dataclass
class RetryAttempt:
    """Details of a retry attempt."""
    attempt_number: int
    strategy_used: RetryStrategy
    parameter_adjustments: Dict[str, Any]
    prompt_modifications: List[str]
    result: Optional[ValidationResult] = None
    success: bool = False
    execution_time: float = 0.0


@dataclass
class RetryResult:
    """Result from retry operations."""
    final_success: bool
    total_attempts: int
    successful_attempt: Optional[RetryAttempt]
    all_attempts: List[RetryAttempt]
    total_time: float
    improvement_achieved: float
    final_validation: Optional[ValidationResult] = None


class SelfValidator:
    """
    Advanced self-validation system with error detection, confidence scoring,
    and intelligent retry mechanisms.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize self-validator with Vertex AI integration."""
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        self.model_name = "gemini-2.5-flash"
        self.debug_mode = os.getenv("VALIDATION_DEBUG", "false").lower() == "true"
        
        # Validation thresholds by level
        self.validation_thresholds = {
            ValidationLevel.PERMISSIVE: 0.6,
            ValidationLevel.STANDARD: 0.75,
            ValidationLevel.STRICT: 0.85,
            ValidationLevel.CRITICAL: 0.95
        }
        
        # Performance tracking
        self.validation_history = []
        self.retry_statistics = {
            "total_retries": 0,
            "successful_retries": 0,
            "strategy_effectiveness": {}
        }
        
        # Initialize optimizer if available
        self.optimizer = None
        if LESSON_3_AVAILABLE:
            try:
                self.optimizer = VertexPromptOptimizer(project_id)
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️  Optimizer initialization failed: {e}")
    
    def validate_response(self, response: str, context: ValidationContext) -> ValidationResult:
        """
        TODO 1: Implement comprehensive self-validation framework.
        
        Requirements:
        - Multi-dimensional confidence scoring across 5 key areas
        - Automated error detection with detailed issue analysis
        - Context-aware quality assessment based on validation level
        - Improvement recommendations with actionable insights
        
        Confidence scoring should include:
        1. Content Quality: Professional language, appropriate length, structure
        2. Context Relevance: Addresses expected elements and business scenario
        3. Factual Accuracy: Consistency, evidence-based claims, hedge word analysis
        4. Structural Coherence: Logical flow, transitions, organization
        5. Completeness: Comprehensive coverage of required topics
        
        Error detection should identify:
        - Length issues (too short/long)
        - Missing required elements
        - Structural problems (poor organization)
        - Language issues (vague terms, repetition)
        - Quality concerns based on validation level
        
        Args:
            response: The generated response to validate
            context: ValidationContext with requirements and settings
            
        Returns:
            ValidationResult with comprehensive assessment and recommendations
        """
        # TODO 1: Implement comprehensive self-validation
        #
        # Step 1: Initialize validation process
        # start_time = time.time()
        #
        # Step 2: Calculate multi-dimensional confidence scores
        # confidence_scores = self._calculate_confidence_scores(response, context)
        # 
        # Step 3: Detect errors and issues
        # detected_issues = self._detect_errors(response, context)
        #
        # Step 4: Assess quality metrics
        # quality_metrics = self._assess_quality_metrics(response, context)
        #
        # Step 5: Generate improvement recommendations
        # recommendations = self._generate_improvement_recommendations(
        #     response, context, confidence_scores, detected_issues
        # )
        #
        # Step 6: Determine validation pass/fail and retry strategy
        # threshold = self.validation_thresholds[context.validation_level]
        # validation_passed = confidence_scores.overall_confidence >= threshold
        # retry_recommended = not validation_passed
        # retry_strategy = self._determine_retry_strategy(...) if retry_recommended else None
        #
        # Step 7: Create and return ValidationResult
        # return ValidationResult(...)
        
        return ValidationResult(
            overall_confidence=0.0,
            confidence_breakdown=ConfidenceScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            detected_issues=[],
            quality_metrics={},
            improvement_recommendations=[],
            validation_passed=False,
            validation_level=context.validation_level,
            processing_time=0.0
        )
    
    def _calculate_confidence_scores(self, response: str, context: ValidationContext) -> ConfidenceScore:
        """Calculate multi-dimensional confidence scores."""
        # TODO: Implement confidence scoring
        # Check content quality, context relevance, factual accuracy, etc.
        return ConfidenceScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def _detect_errors(self, response: str, context: ValidationContext) -> List[ValidationIssue]:
        """Detect various types of errors and issues in the response."""
        # TODO: Implement error detection
        # Check for length issues, missing elements, structural problems, etc.
        return []
    
    def _assess_quality_metrics(self, response: str, context: ValidationContext) -> Dict[str, float]:
        """Assess various quality metrics for the response."""
        # TODO: Implement quality assessment
        # Calculate readability, professional language ratio, structure metrics, etc.
        return {}
    
    def _generate_improvement_recommendations(self, response: str, context: ValidationContext,
                                           confidence: ConfidenceScore, issues: List[ValidationIssue]) -> List[str]:
        """Generate actionable improvement recommendations."""
        # TODO: Implement recommendation generation
        # Based on confidence scores and detected issues
        return []
    
    def _determine_retry_strategy(self, confidence: ConfidenceScore, 
                                 issues: List[ValidationIssue], 
                                 context: ValidationContext) -> RetryStrategy:
        """Determine the best retry strategy based on validation results."""
        # TODO: Implement retry strategy selection
        # Based on issue types and confidence deficits
        return RetryStrategy.PARAMETER_ADJUSTMENT
    
    def auto_retry_with_improvement(self, response: str, context: ValidationContext, 
                                   validation_result: ValidationResult, 
                                   max_attempts: int = 3) -> RetryResult:
        """
        TODO 2: Implement intelligent retry mechanism with learning.
        
        Requirements:
        - Multiple retry strategies based on failure analysis
        - Parameter adjustment and prompt enhancement
        - Learning integration to improve future attempts
        - Comprehensive attempt tracking and success measurement
        
        Retry strategies to implement:
        1. Parameter Adjustment: Modify temperature, top_p, etc.
        2. Prompt Enhancement: Add specific guidance and requirements
        3. Context Enrichment: Provide additional context and examples
        4. Alternative Approach: Try different reasoning methods
        5. Escalation: Flag for manual review
        
        Learning integration should:
        - Track which strategies work best for different failure types
        - Adjust parameters based on previous attempt results
        - Accumulate knowledge about successful improvements
        - Apply lessons learned to future retry attempts
        
        Args:
            response: Original response that failed validation
            context: ValidationContext for the operation
            validation_result: Failed validation result with retry strategy
            max_attempts: Maximum number of retry attempts
            
        Returns:
            RetryResult with final success status and attempt details
        """
        # TODO 2: Implement intelligent retry mechanism
        #
        # Step 1: Initialize retry process
        # start_time = time.time()
        # attempts = []
        #
        # Step 2: Execute retry attempts with different strategies
        # for attempt_num in range(1, max_attempts + 1):
        #     attempt = RetryAttempt(
        #         attempt_number=attempt_num,
        #         strategy_used=validation_result.retry_strategy,
        #         parameter_adjustments={},
        #         prompt_modifications=[]
        #     )
        #     
        #     # Apply strategy-specific improvements
        #     if validation_result.retry_strategy == RetryStrategy.PARAMETER_ADJUSTMENT:
        #         improved_context = self._adjust_parameters(context, validation_result, attempt_num)
        #     elif validation_result.retry_strategy == RetryStrategy.PROMPT_ENHANCEMENT:
        #         improved_context = self._enhance_prompt(context, validation_result)
        #     # ... handle other strategies
        #     
        #     # Execute retry and validate result
        #     # Track success and update statistics
        #     
        # Step 3: Return comprehensive retry result
        # return RetryResult(...)
        
        return RetryResult(
            final_success=False,
            total_attempts=0,
            successful_attempt=None,
            all_attempts=[],
            total_time=0.0,
            improvement_achieved=0.0
        )
    
    def _adjust_parameters(self, context: ValidationContext, validation: ValidationResult, attempt: int) -> ValidationContext:
        """Adjust parameters for retry attempt."""
        # TODO: Implement parameter adjustment
        return context
    
    def _enhance_prompt(self, context: ValidationContext, validation: ValidationResult) -> ValidationContext:
        """Enhance prompt based on validation feedback."""
        # TODO: Implement prompt enhancement
        return context
    
    def implement_quality_gates(self, validation_result: ValidationResult, 
                               escalation_threshold: float = 0.5) -> Dict[str, Any]:
        """
        TODO 3: Implement quality gate systems with escalation.
        
        Requirements:
        - Multi-level quality gates with different validation thresholds
        - Automatic escalation for persistent failures
        - Fallback strategies for different failure types
        - Comprehensive quality control workflow with decision logic
        
        Quality gate levels to implement:
        1. Primary Gate: Standard threshold validation
        2. Secondary Gate: Escalation threshold check
        3. Tertiary Gate: Critical issue detection
        4. Final Gate: Comprehensive quality assessment
        
        Escalation logic should consider:
        - Confidence deficit magnitude
        - Number and severity of detected issues
        - Failure pattern analysis
        - Resource and time constraints
        
        Fallback strategies should include:
        - Manual review requirement
        - Structured template usage
        - Enhanced retry with additional guidance
        - Alternative content generation approaches
        
        Args:
            validation_result: ValidationResult from primary validation
            escalation_threshold: Threshold for triggering escalation
            
        Returns:
            Dict with gate results, escalation decisions, and recommendations
        """
        # TODO 3: Implement quality gate systems
        #
        # Step 1: Initialize quality gate result structure
        # quality_gate_result = {
        #     "gate_passed": False,
        #     "gate_level": validation_result.validation_level.value,
        #     "confidence_score": validation_result.overall_confidence,
        #     "threshold": self.validation_thresholds[validation_result.validation_level],
        #     "escalation_required": False,
        #     "escalation_reason": None,
        #     "fallback_strategy": None,
        #     "recommendations": []
        # }
        #
        # Step 2: Primary quality gate check
        # threshold = self.validation_thresholds[validation_result.validation_level]
        # if validation_result.overall_confidence >= threshold:
        #     quality_gate_result["gate_passed"] = True
        #     return quality_gate_result
        #
        # Step 3: Escalation decision logic
        # confidence_deficit = threshold - validation_result.overall_confidence
        # if confidence_deficit > escalation_threshold:
        #     quality_gate_result["escalation_required"] = True
        #     quality_gate_result["escalation_reason"] = f"Confidence deficit: {confidence_deficit:.2f}"
        #
        # Step 4: Determine fallback strategy
        # Based on issue types and confidence levels
        #
        # Step 5: Generate specific recommendations
        # return quality_gate_result
        
        return {}
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        if not self.validation_history:
            return {"error": "No validation history available"}
        
        recent_validations = self.validation_history[-50:]  # Last 50 validations
        
        stats = {
            "total_validations": len(self.validation_history),
            "recent_validations": len(recent_validations),
            "average_confidence": statistics.mean([v["confidence"] for v in recent_validations]),
            "pass_rate": sum(1 for v in recent_validations if v["passed"]) / len(recent_validations),
            "validation_levels": {},
            "retry_statistics": self.retry_statistics,
            "trend_analysis": self._analyze_validation_trends()
        }
        
        # Breakdown by validation level
        for level in ValidationLevel:
            level_validations = [v for v in recent_validations if v["level"] == level.value]
            if level_validations:
                stats["validation_levels"][level.value] = {
                    "count": len(level_validations),
                    "pass_rate": sum(1 for v in level_validations if v["passed"]) / len(level_validations),
                    "avg_confidence": statistics.mean([v["confidence"] for v in level_validations])
                }
        
        return stats
    
    def _analyze_validation_trends(self) -> Dict[str, Any]:
        """Analyze trends in validation performance."""
        if len(self.validation_history) < 10:
            return {"insufficient_data": True}
        
        recent_10 = self.validation_history[-10:]
        previous_10 = self.validation_history[-20:-10] if len(self.validation_history) >= 20 else []
        
        current_avg = statistics.mean([v["confidence"] for v in recent_10])
        
        trends = {"current_average_confidence": current_avg}
        
        if previous_10:
            previous_avg = statistics.mean([v["confidence"] for v in previous_10])
            trends["previous_average_confidence"] = previous_avg
            trends["confidence_trend"] = "improving" if current_avg > previous_avg else "declining"
            trends["trend_magnitude"] = abs(current_avg - previous_avg)
        
        return trends


def load_test_scenarios() -> List[Dict]:
    """Load test scenarios for validation testing."""
    return [
        {
            "company_name": "TechFlow Solutions",
            "industry": "Software Technology", 
            "market_focus": "enterprise workflow automation",
            "strategic_question": "Should we expand into small business markets?",
            "additional_context": "Strong enterprise presence, considering SMB expansion.",
            "expected_elements": ["market", "competitive", "financial", "recommendation"]
        },
        {
            "company_name": "GreenEnergy Corp",
            "industry": "Renewable Energy",
            "market_focus": "commercial solar installations",
            "strategic_question": "How should we respond to increased competition?",
            "additional_context": "Market leader for 5 years, new competitors entering.",
            "expected_elements": ["competitive", "positioning", "pricing", "strategy"]
        }
    ]


def run_validation_test():
    """Demonstrate self-validation capabilities."""
    PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id")
    if PROJECT_ID == "your-project-id":
        print("❌ Please set PROJECT_ID environment variable")
        return
    
    # Check if TODOs are implemented
    validator = SelfValidator(PROJECT_ID)
    scenarios = load_test_scenarios()
    test_scenario = scenarios[0]
    
    print("🔍 Testing Self-Validation System")
    print("=" * 60)
    
    test_response = """This comprehensive market analysis reveals significant opportunities for TechFlow Solutions' expansion into small business markets. 

Market Overview: The SMB workflow automation market has grown 15% annually, reaching $2.3B globally. Current enterprise solutions are over-engineered for smaller businesses, creating a clear market gap.

Competitive Analysis: Main competitors focus on enterprise segments, leaving SMB underserved. Our simplified platform approach could capture 12% market share within 24 months.

Financial Projections: SMB expansion requires $2M investment but projects $8M revenue by year 2, with 40% gross margins.

Strategic Recommendation: Proceed with SMB expansion using a simplified product offering, targeting 100-500 employee companies initially."""
    
    context = ValidationContext(
        original_prompt="Analyze the strategic implications of market expansion",
        expected_elements=test_scenario["expected_elements"],
        business_scenario=test_scenario,
        validation_level=ValidationLevel.STANDARD
    )
    
    # Test TODO 1
    try:
        validation_result = validator.validate_response(test_response, context)
        if validation_result is None:
            print("❌ TODO 1 not implemented: validate_response")
            return
        print("✅ TODO 1: Self-validation framework - IMPLEMENTED")
    except Exception as e:
        print(f"❌ TODO 1: Self-validation error - {e}")
        return
    
    # Test TODO 2
    try:
        if not validation_result.validation_passed:
            retry_result = validator.auto_retry_with_improvement(
                test_response, context, validation_result, max_attempts=2
            )
            if retry_result is None:
                print("❌ TODO 2 not implemented: auto_retry_with_improvement")
                return
        print("✅ TODO 2: Automatic retry and refinement - IMPLEMENTED")
    except Exception as e:
        print(f"❌ TODO 2: Retry mechanism error - {e}")
        return
    
    # Test TODO 3
    try:
        gate_result = validator.implement_quality_gates(validation_result)
        if gate_result is None:
            print("❌ TODO 3 not implemented: implement_quality_gates")
            return
        print("✅ TODO 3: Quality gate systems - IMPLEMENTED")
    except Exception as e:
        print(f"❌ TODO 3: Quality gates error - {e}")
        return
    
    print("\n🎯 All TODOs implemented successfully!")
    print("🚀 Ready for improvement_engine.py!")


if __name__ == "__main__":
    run_validation_test()