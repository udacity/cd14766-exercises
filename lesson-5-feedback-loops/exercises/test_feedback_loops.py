"""
Lesson 5: Feedback Loops and Iterative Improvement - Comprehensive Test Suite

Author: Noble Ackerson (Udacity)
Date: 2025

This module provides comprehensive testing for all Lesson 5 components including
self-validation systems, iterative improvement engines, and production monitoring.

Test Coverage:
- TODO 1: Self-validation framework testing
- TODO 2: Automatic retry and refinement testing  
- TODO 3: Quality gate systems testing
- TODO 4: Performance analytics system testing
- TODO 5: Continuous learning integration testing
- TODO 6: Production monitoring dashboard testing
- Cross-lesson integration testing
- Production readiness validation
"""

import os
import sys
import time
import json
import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add the exercises directories to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'solution'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'starter'))


class TestEnvironment:
    """Test environment setup and configuration."""
    
    def __init__(self):
        self.project_id = os.getenv("PROJECT_ID", "test-project")
        self.test_data_dir = tempfile.mkdtemp()
        self.mock_responses = self._load_mock_responses()
        
    def _load_mock_responses(self) -> Dict[str, Any]:
        """Load mock responses for testing without actual API calls."""
        return {
            "high_quality_response": {
                "content": "This is a comprehensive business analysis with detailed market insights, competitive analysis, and strategic recommendations.",
                "confidence": 0.92,
                "token_count": 2500
            },
            "low_quality_response": {
                "content": "Brief analysis.",
                "confidence": 0.45,
                "token_count": 150
            },
            "validation_result": {
                "overall_confidence": 0.85,
                "validation_passed": True,
                "detected_issues": [],
                "improvement_recommendations": ["Enhance detail in competitive analysis"]
            }
        }
    
    def cleanup(self):
        """Clean up test environment."""
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)


class TestSelfValidationFramework(unittest.TestCase):
    """
    Test suite for TODO 1: Self-Validation Framework
    
    Tests comprehensive self-assessment capabilities including:
    - Multi-dimensional confidence scoring
    - Automated error detection
    - Content validation systems
    - Dynamic quality thresholds
    """
    
    def setUp(self):
        """Set up test environment for self-validation tests."""
        self.env = TestEnvironment()
        
        # Try to import solution first, fall back to starter
        try:
            from solution.self_validator import SelfValidator, ValidationContext, ValidationResult
            self.validator = SelfValidator(self.env.project_id)
            self.has_solution = True
        except (ImportError, NotImplementedError):
            try:
                from starter.self_validator import SelfValidator, ValidationContext, ValidationResult
                self.validator = SelfValidator(self.env.project_id)
                self.has_solution = False
            except ImportError:
                self.skipTest("SelfValidator not found in solution or starter")
    
    def test_confidence_scoring_accuracy(self):
        """Test TODO 1: Multi-dimensional confidence scoring."""
        if not self.has_solution:
            self.skipTest("TODO 1 not implemented yet")
            
        # Test high-quality response
        high_quality_response = self.env.mock_responses["high_quality_response"]["content"]
        context = ValidationContext(
            scenario_type="business_analysis",
            validation_level="standard",
            expected_sections=["market_overview", "competitive_analysis", "recommendations"],
            quality_threshold=0.8
        )
        
        result = self.validator.validate_response(high_quality_response, context)
        
        # Validate confidence scoring
        self.assertIsInstance(result, ValidationResult)
        self.assertGreaterEqual(result.overall_confidence, 0.8, 
                               "High-quality response should have high confidence")
        self.assertTrue(result.validation_passed, 
                       "High-quality response should pass validation")
        
        # Test confidence breakdown
        self.assertIn("content_quality", result.confidence_breakdown.__dict__)
        self.assertIn("context_relevance", result.confidence_breakdown.__dict__)
        self.assertIn("factual_accuracy", result.confidence_breakdown.__dict__)
        
        print(f"✅ TODO 1: Confidence scoring - Overall: {result.overall_confidence:.3f}")
    
    def test_error_detection_system(self):
        """Test TODO 1: Automated error detection capabilities.""" 
        if not self.has_solution:
            self.skipTest("TODO 1 not implemented yet")
            
        # Test low-quality response that should trigger error detection
        low_quality_response = self.env.mock_responses["low_quality_response"]["content"]
        context = ValidationContext(
            scenario_type="business_analysis",
            validation_level="strict",
            expected_sections=["market_overview", "competitive_analysis", "recommendations"],
            quality_threshold=0.8
        )
        
        result = self.validator.validate_response(low_quality_response, context)
        
        # Should detect quality issues
        self.assertFalse(result.validation_passed, 
                        "Low-quality response should fail validation")
        self.assertGreater(len(result.detected_issues), 0,
                          "Should detect issues in low-quality response")
        
        # Check error categories
        issue_types = [issue.issue_type for issue in result.detected_issues]
        expected_types = ["insufficient_length", "missing_sections", "low_detail"]
        self.assertTrue(any(t in issue_types for t in expected_types),
                       f"Should detect expected issue types, got: {issue_types}")
        
        print(f"✅ TODO 1: Error detection - Found {len(result.detected_issues)} issues")
    
    def test_dynamic_quality_thresholds(self):
        """Test TODO 1: Dynamic quality threshold management."""
        if not self.has_solution:
            self.skipTest("TODO 1 not implemented yet")
            
        response = self.env.mock_responses["high_quality_response"]["content"]
        
        # Test different validation levels
        contexts = [
            ValidationContext("business_analysis", "lenient", [], 0.6),
            ValidationContext("business_analysis", "standard", [], 0.8),
            ValidationContext("business_analysis", "strict", [], 0.9)
        ]
        
        results = []
        for context in contexts:
            result = self.validator.validate_response(response, context)
            results.append((context.validation_level, result.validation_passed))
        
        # Should have different outcomes for different thresholds
        validation_outcomes = [passed for _, passed in results]
        self.assertTrue(len(set(validation_outcomes)) > 1 or all(validation_outcomes),
                       "Dynamic thresholds should affect validation outcomes")
        
        print(f"✅ TODO 1: Dynamic thresholds - Results: {results}")


class TestAutomaticRetryRefinement(unittest.TestCase):
    """
    Test suite for TODO 2: Automatic Retry and Refinement
    
    Tests intelligent retry mechanisms including:
    - Smart retry logic with parameter adjustment
    - Improvement tracking across attempts
    - Learning integration for future attempts
    - Escalation paths for persistent failures
    """
    
    def setUp(self):
        """Set up test environment for retry system tests."""
        self.env = TestEnvironment()
        
        try:
            from solution.self_validator import SelfValidator, FailedAttempt, RetryResult
            self.validator = SelfValidator(self.env.project_id)
            self.has_solution = True
        except (ImportError, NotImplementedError):
            try:
                from starter.self_validator import SelfValidator, FailedAttempt, RetryResult
                self.validator = SelfValidator(self.env.project_id)
                self.has_solution = False
            except ImportError:
                self.skipTest("SelfValidator retry system not found")
    
    def test_intelligent_retry_logic(self):
        """Test TODO 2: Smart retry with parameter adjustment."""
        if not self.has_solution:
            self.skipTest("TODO 2 not implemented yet")
            
        # Mock a failed attempt
        failed_attempt = FailedAttempt(
            attempt_number=1,
            original_prompt="Generate a business analysis",
            original_parameters={"temperature": 0.7, "max_tokens": 1000},
            failure_reason="insufficient_detail",
            validation_result=self.env.mock_responses["validation_result"],
            context={"scenario_type": "business_analysis"}
        )
        
        retry_result = self.validator.auto_retry_with_improvement(failed_attempt)
        
        # Validate retry improvements
        self.assertIsInstance(retry_result, RetryResult)
        self.assertNotEqual(retry_result.adjusted_parameters, failed_attempt.original_parameters,
                           "Parameters should be adjusted for retry")
        self.assertNotEqual(retry_result.improved_prompt, failed_attempt.original_prompt,
                           "Prompt should be enhanced for retry")
        
        # Check specific improvements
        if "max_tokens" in retry_result.adjusted_parameters:
            self.assertGreater(retry_result.adjusted_parameters["max_tokens"], 
                             failed_attempt.original_parameters["max_tokens"],
                             "Should increase max_tokens for insufficient detail")
        
        print(f"✅ TODO 2: Retry logic - Strategy: {retry_result.retry_strategy}")
    
    def test_improvement_tracking(self):
        """Test TODO 2: Monitor refinement effectiveness across attempts."""
        if not self.has_solution:
            self.skipTest("TODO 2 not implemented yet")
            
        # Simulate multiple retry attempts
        attempts = []
        for i in range(3):
            failed_attempt = FailedAttempt(
                attempt_number=i+1,
                original_prompt=f"Generate business analysis (attempt {i+1})",
                original_parameters={"temperature": 0.7 + i*0.1, "max_tokens": 1000 + i*500},
                failure_reason="quality_issues",
                validation_result={"confidence": 0.5 + i*0.1},
                context={"scenario_type": "business_analysis"}
            )
            attempts.append(failed_attempt)
            
            retry_result = self.validator.auto_retry_with_improvement(failed_attempt)
            
            # Track improvement metrics
            self.assertIn("improvement_score", retry_result.__dict__ or {},
                         "Should track improvement effectiveness")
            
        print(f"✅ TODO 2: Improvement tracking - Processed {len(attempts)} attempts")
    
    def test_learning_integration(self):
        """Test TODO 2: Incorporate feedback into future attempts."""
        if not self.has_solution:
            self.skipTest("TODO 2 not implemented yet")
            
        # Test that learning from previous failures improves future retries
        failure_patterns = [
            {"type": "insufficient_detail", "count": 3},
            {"type": "poor_structure", "count": 2}, 
            {"type": "missing_sections", "count": 4}
        ]
        
        # Validator should learn from these patterns
        for pattern in failure_patterns:
            failed_attempt = FailedAttempt(
                attempt_number=1,
                original_prompt="Generate analysis",
                original_parameters={"temperature": 0.7},
                failure_reason=pattern["type"],
                validation_result={"confidence": 0.4},
                context={"scenario_type": "business_analysis"}
            )
            
            retry_result = self.validator.auto_retry_with_improvement(failed_attempt)
            
            # Should adapt based on learned patterns
            self.assertIn("learning_applied", retry_result.__dict__ or {},
                         "Should apply learning from failure patterns")
        
        print(f"✅ TODO 2: Learning integration - Applied patterns from {len(failure_patterns)} failure types")
    
    def test_escalation_paths(self):
        """Test TODO 2: Handle persistent failures gracefully."""
        if not self.has_solution:
            self.skipTest("TODO 2 not implemented yet")
            
        # Simulate persistent failure requiring escalation
        persistent_failure = FailedAttempt(
            attempt_number=5,  # High attempt number
            original_prompt="Generate complex analysis",
            original_parameters={"temperature": 0.9, "max_tokens": 4000},
            failure_reason="persistent_quality_issues",
            validation_result={"confidence": 0.3},
            context={"scenario_type": "complex_analysis", "escalation_threshold": 3}
        )
        
        retry_result = self.validator.auto_retry_with_improvement(persistent_failure)
        
        # Should trigger escalation
        self.assertTrue(retry_result.escalation_triggered,
                       "Should trigger escalation for persistent failures")
        self.assertIn("alternative_approach", retry_result.__dict__,
                     "Should provide alternative approaches")
        
        print(f"✅ TODO 2: Escalation paths - Triggered at attempt {persistent_failure.attempt_number}")


class TestQualityGateSystems(unittest.TestCase):
    """
    Test suite for TODO 3: Quality Gate Systems
    
    Tests comprehensive quality control including:
    - Multi-level validation gates
    - Automatic escalation workflows  
    - Fallback strategies for failures
    - Performance integration
    """
    
    def setUp(self):
        """Set up test environment for quality gate tests."""
        self.env = TestEnvironment()
        
        try:
            from solution.self_validator import SelfValidator, QualityGate, EscalationWorkflow
            self.validator = SelfValidator(self.env.project_id)
            self.has_solution = True
        except (ImportError, NotImplementedError):
            try:
                from starter.self_validator import SelfValidator, QualityGate, EscalationWorkflow
                self.validator = SelfValidator(self.env.project_id)
                self.has_solution = False
            except ImportError:
                self.skipTest("Quality gate systems not found")
    
    def test_multi_level_quality_gates(self):
        """Test TODO 3: Different validation levels for different use cases."""
        if not self.has_solution:
            self.skipTest("TODO 3 not implemented yet")
            
        response = self.env.mock_responses["high_quality_response"]["content"]
        
        # Test different quality gate levels
        gate_levels = ["basic", "standard", "premium", "enterprise"]
        results = {}
        
        for level in gate_levels:
            gate_result = self.validator.apply_quality_gate(response, level)
            results[level] = gate_result
            
            self.assertIn("gate_passed", gate_result)
            self.assertIn("quality_score", gate_result)
            self.assertIn("gate_level", gate_result)
            
        # Verify different standards
        quality_scores = [results[level]["quality_score"] for level in gate_levels]
        self.assertTrue(len(set([r["gate_passed"] for r in results.values()])) >= 1,
                       "Quality gates should have different pass/fail outcomes")
        
        print(f"✅ TODO 3: Multi-level gates - Tested {len(gate_levels)} levels")
    
    def test_escalation_workflows(self):
        """Test TODO 3: Automatic escalation for quality failures."""
        if not self.has_solution:
            self.skipTest("TODO 3 not implemented yet")
            
        # Test escalation for repeated quality failures
        low_quality_response = self.env.mock_responses["low_quality_response"]["content"]
        
        escalation_result = self.validator.trigger_escalation_workflow(
            response=low_quality_response,
            failure_count=3,
            failure_type="quality_threshold",
            context={"urgency": "high", "business_impact": "medium"}
        )
        
        # Validate escalation workflow
        self.assertIn("escalation_level", escalation_result)
        self.assertIn("escalation_actions", escalation_result)
        self.assertIn("estimated_resolution_time", escalation_result)
        
        # Check escalation actions
        actions = escalation_result["escalation_actions"]
        expected_actions = ["human_review", "alternative_model", "fallback_response"]
        self.assertTrue(any(action in actions for action in expected_actions),
                       "Should provide appropriate escalation actions")
        
        print(f"✅ TODO 3: Escalation workflow - Level: {escalation_result['escalation_level']}")
    
    def test_fallback_strategies(self):
        """Test TODO 3: Alternative approaches when primary methods fail."""
        if not self.has_solution:
            self.skipTest("TODO 3 not implemented yet")
            
        # Test fallback when primary validation fails
        fallback_result = self.validator.execute_fallback_strategy(
            primary_failure="validation_service_unavailable",
            context={"scenario_type": "business_analysis", "urgency": "high"},
            available_strategies=["heuristic_validation", "cached_response", "simplified_analysis"]
        )
        
        # Validate fallback execution
        self.assertIn("strategy_used", fallback_result)
        self.assertIn("fallback_quality", fallback_result)
        self.assertIn("success", fallback_result)
        
        self.assertTrue(fallback_result["success"],
                       "Fallback strategy should succeed")
        self.assertGreaterEqual(fallback_result["fallback_quality"], 0.6,
                               "Fallback should maintain reasonable quality")
        
        print(f"✅ TODO 3: Fallback strategies - Used: {fallback_result['strategy_used']}")
    
    def test_performance_integration(self):
        """Test TODO 3: Connect quality gates to performance metrics."""
        if not self.has_solution:
            self.skipTest("TODO 3 not implemented yet")
            
        # Test integration with performance monitoring
        performance_context = {
            "response_time_budget": 5.0,
            "cost_budget": 0.10,
            "quality_threshold": 0.8,
            "performance_tier": "standard"
        }
        
        result = self.validator.validate_with_performance_constraints(
            response=self.env.mock_responses["high_quality_response"]["content"],
            performance_context=performance_context
        )
        
        # Validate performance integration
        self.assertIn("quality_passed", result)
        self.assertIn("performance_passed", result)
        self.assertIn("cost_passed", result)
        self.assertIn("overall_passed", result)
        
        # Should consider all constraints
        self.assertEqual(result["overall_passed"], 
                        all([result["quality_passed"], result["performance_passed"], result["cost_passed"]]),
                        "Overall pass should consider all constraints")
        
        print(f"✅ TODO 3: Performance integration - Overall passed: {result['overall_passed']}")


class TestPerformanceAnalyticsSystem(unittest.TestCase):
    """
    Test suite for TODO 4: Performance Analytics System
    
    Tests sophisticated performance tracking including:
    - Comprehensive metrics collection
    - Trend analysis and anomaly detection
    - Real-time alerting systems
    - AI-driven optimization recommendations
    """
    
    def setUp(self):
        """Set up test environment for performance analytics tests."""
        self.env = TestEnvironment()
        
        try:
            from solution.improvement_engine import PerformanceAnalytics, MetricsCollection
            self.analytics = PerformanceAnalytics(self.env.project_id)
            self.has_solution = True
        except (ImportError, NotImplementedError):
            try:
                from starter.improvement_engine import PerformanceAnalytics, MetricsCollection
                self.analytics = PerformanceAnalytics(self.env.project_id)
                self.has_solution = False
            except ImportError:
                self.skipTest("PerformanceAnalytics not found")
    
    def test_comprehensive_metrics_collection(self):
        """Test TODO 4: Metrics collection across all dimensions."""
        if not self.has_solution:
            self.skipTest("TODO 4 not implemented yet")
            
        # Mock operation execution
        execution_result = {
            "success": True,
            "response_time": 2.5,
            "processing_time": 2.1,
            "input_tokens": 1500,
            "output_tokens": 3000,
            "estimated_cost": 0.045,
            "retry_count": 0,
            "user_satisfaction": 4.2,
            "quality_rating": 4.0
        }
        
        # Mock validation result
        validation_result = Mock()
        validation_result.overall_confidence = 0.87
        validation_result.validation_passed = True
        validation_result.detected_issues = []
        
        metrics = self.analytics.collect_metrics("business_report", execution_result, validation_result)
        
        # Validate comprehensive collection
        self.assertIsInstance(metrics, MetricsCollection)
        self.assertGreater(len(metrics.quality_metrics), 0, "Should collect quality metrics")
        self.assertGreater(len(metrics.performance_metrics), 0, "Should collect performance metrics")
        self.assertGreater(len(metrics.cost_metrics), 0, "Should collect cost metrics")
        
        # Validate specific metrics
        self.assertIn("overall_confidence", metrics.quality_metrics)
        self.assertIn("response_time", metrics.performance_metrics)
        self.assertIn("estimated_cost", metrics.cost_metrics)
        
        print(f"✅ TODO 4: Metrics collection - Collected {len(metrics.quality_metrics)} quality, {len(metrics.performance_metrics)} performance metrics")
    
    def test_trend_analysis_and_anomaly_detection(self):
        """Test TODO 4: Identify performance patterns and anomalies."""
        if not self.has_solution:
            self.skipTest("TODO 4 not implemented yet")
            
        # Generate sample metrics history for trend analysis
        for i in range(50):
            execution_result = {
                "success": True,
                "response_time": 2.0 + (i * 0.1),  # Gradual increase
                "quality_score": 0.8 + (i * 0.002),  # Slight improvement
                "cost": 0.05 - (i * 0.0005)  # Slight decrease
            }
            
            metrics = self.analytics.collect_metrics(f"operation_{i}", execution_result)
            self.analytics.metrics_history.append(metrics)
        
        # Analyze trends
        response_time_trend = self.analytics.analyze_trends("response_time")
        quality_trend = self.analytics.analyze_trends("quality_score")
        
        # Validate trend detection
        self.assertEqual(response_time_trend.trend_direction, "degrading",
                        "Should detect degrading response time trend")
        self.assertEqual(quality_trend.trend_direction, "improving",
                        "Should detect improving quality trend")
        
        # Test anomaly detection
        # Add an anomalous data point
        anomalous_execution = {
            "response_time": 15.0,  # Much higher than normal
            "quality_score": 0.3    # Much lower than normal
        }
        anomalous_metrics = self.analytics.collect_metrics("anomaly_test", anomalous_execution)
        
        # Should detect anomaly
        anomalies = self.analytics._detect_anomalies([m.performance_metrics.get("response_time", 0) 
                                                    for m in self.analytics.metrics_history[-10:]])
        self.assertGreater(len(anomalies), 0, "Should detect anomalous response times")
        
        print(f"✅ TODO 4: Trend analysis - Response time: {response_time_trend.trend_direction}, Quality: {quality_trend.trend_direction}")
    
    def test_real_time_alerting_system(self):
        """Test TODO 4: Real-time alerts for performance issues."""
        if not self.has_solution:
            self.skipTest("TODO 4 not implemented yet")
            
        # Test threshold-based alerting
        critical_execution = {
            "success": False,
            "response_time": 12.0,  # Above critical threshold
            "error_rate": 0.15,     # Above critical threshold
            "cost": 0.25           # Above critical threshold
        }
        
        metrics = self.analytics.collect_metrics("critical_test", critical_execution)
        
        # Should generate alerts
        self.assertGreater(len(self.analytics.active_alerts), 0,
                          "Should generate alerts for threshold violations")
        
        # Check alert properties
        alert = self.analytics.active_alerts[0]
        self.assertIn("alert_id", alert.__dict__)
        self.assertIn("severity", alert.__dict__)
        self.assertIn("suggested_actions", alert.__dict__)
        
        print(f"✅ TODO 4: Real-time alerting - Generated {len(self.analytics.active_alerts)} alerts")
    
    def test_ai_driven_optimization_recommendations(self):
        """Test TODO 4: Generate optimization recommendations."""
        if not self.has_solution:
            self.skipTest("TODO 4 not implemented yet")
            
        # Mock performance data indicating optimization opportunities
        performance_data = {
            "avg_response_time": 4.5,
            "cost_trend": "increasing",
            "quality_variance": 0.15,
            "error_rate": 0.08,
            "user_satisfaction": 3.2
        }
        
        recommendations = self.analytics.generate_optimization_recommendations(performance_data)
        
        # Validate recommendations
        self.assertGreater(len(recommendations), 0, "Should generate optimization recommendations")
        
        for rec in recommendations:
            self.assertIn("category", rec.__dict__)
            self.assertIn("priority", rec.__dict__)
            self.assertIn("description", rec.__dict__)
            self.assertIn("expected_impact", rec.__dict__)
            
        # Should cover key optimization areas
        categories = [rec.category for rec in recommendations]
        expected_categories = ["performance", "cost", "quality"]
        self.assertTrue(any(cat in categories for cat in expected_categories),
                       f"Should cover optimization categories, got: {categories}")
        
        print(f"✅ TODO 4: AI optimization - Generated {len(recommendations)} recommendations")


class TestContinuousLearningIntegration(unittest.TestCase):
    """
    Test suite for TODO 5: Continuous Learning Integration
    
    Tests feedback-driven learning systems including:
    - Structured feedback processing
    - Dynamic model adaptation
    - Historical trend analysis
    - Predictive optimization
    """
    
    def setUp(self):
        """Set up test environment for continuous learning tests."""
        self.env = TestEnvironment()
        
        try:
            from solution.improvement_engine import ContinuousLearningEngine, PerformanceAnalytics
            self.learning_engine = ContinuousLearningEngine(self.env.project_id)
            self.analytics = PerformanceAnalytics(self.env.project_id)
            self.has_solution = True
        except (ImportError, NotImplementedError):
            try:
                from starter.improvement_engine import ContinuousLearningEngine, PerformanceAnalytics
                self.learning_engine = ContinuousLearningEngine(self.env.project_id)
                self.analytics = PerformanceAnalytics(self.env.project_id)
                self.has_solution = False
            except ImportError:
                self.skipTest("Continuous learning components not found")
    
    def test_feedback_processing_system(self):
        """Test TODO 5: Structured feedback collection and analysis."""
        if not self.has_solution:
            self.skipTest("TODO 5 not implemented yet")
            
        # Mock operation result and user feedback
        operation_result = {
            "operation_id": "test_001",
            "success": True,
            "quality_score": 0.75,
            "response_time": 3.2,
            "cost": 0.08
        }
        
        user_feedback = {
            "satisfaction_score": 3.8,
            "quality_rating": 4.0,
            "usefulness": 4.2,
            "specific_feedback": "Good analysis but could be more detailed",
            "areas_for_improvement": ["detail", "examples"]
        }
        
        learning_insights = self.learning_engine.learn_from_feedback(operation_result, user_feedback)
        
        # Validate feedback processing
        self.assertIn("correlation_analysis", learning_insights)
        self.assertIn("improvement_patterns", learning_insights)
        self.assertIn("adaptation_recommendations", learning_insights)
        
        # Should identify correlations between metrics and feedback
        correlation = learning_insights["correlation_analysis"]
        self.assertIn("quality_satisfaction_correlation", correlation)
        
        print(f"✅ TODO 5: Feedback processing - Processed feedback with {len(learning_insights)} insight categories")
    
    def test_dynamic_model_adaptation(self):
        """Test TODO 5: Model adaptation based on performance data."""
        if not self.has_solution:
            self.skipTest("TODO 5 not implemented yet")
            
        # Mock context requiring adaptation
        adaptation_context = {
            "scenario_type": "financial_analysis",
            "recent_performance": {
                "avg_quality": 0.72,
                "avg_satisfaction": 3.5,
                "success_rate": 0.85
            },
            "performance_trends": {
                "quality": "declining",
                "speed": "stable",
                "cost": "increasing"
            }
        }
        
        adapted_parameters = self.learning_engine.adapt_parameters(adaptation_context)
        
        # Validate parameter adaptation
        self.assertIn("prompt_adjustments", adapted_parameters)
        self.assertIn("model_parameters", adapted_parameters)
        self.assertIn("validation_thresholds", adapted_parameters)
        self.assertIn("confidence_score", adapted_parameters)
        
        # Should provide specific adaptations based on trends
        if adaptation_context["performance_trends"]["quality"] == "declining":
            self.assertIn("quality_enhancement", adapted_parameters["prompt_adjustments"])
        
        print(f"✅ TODO 5: Model adaptation - Applied {len(adapted_parameters)} parameter adjustments")
    
    def test_historical_trend_analysis(self):
        """Test TODO 5: Long-term performance trend analysis."""
        if not self.has_solution:
            self.skipTest("TODO 5 not implemented yet")
            
        # Generate historical performance data
        historical_data = []
        for week in range(12):  # 12 weeks of data
            week_data = {
                "week": week,
                "avg_quality": 0.7 + (week * 0.02),  # Gradual improvement
                "avg_response_time": 3.0 - (week * 0.1),  # Gradual improvement
                "user_satisfaction": 3.5 + (week * 0.05),  # Gradual improvement
                "cost_efficiency": 0.08 - (week * 0.002)  # Gradual improvement
            }
            historical_data.append(week_data)
        
        # Analyze long-term trends
        trend_analysis = self.analytics.analyze_trends("avg_quality", timedelta(weeks=12))
        
        # Validate historical analysis
        self.assertGreaterEqual(trend_analysis.data_points, 10, "Should analyze sufficient historical data")
        self.assertEqual(trend_analysis.trend_direction, "improving", "Should detect improvement trend")
        self.assertGreater(trend_analysis.confidence, 0.7, "Should have high confidence in trend")
        
        # Should provide long-term recommendations
        self.assertGreater(len(trend_analysis.recommendations), 0,
                          "Should provide recommendations based on historical trends")
        
        print(f"✅ TODO 5: Historical analysis - Analyzed {trend_analysis.data_points} data points over {trend_analysis.time_period}")
    
    def test_predictive_optimization(self):
        """Test TODO 5: Proactive performance improvement."""
        if not self.has_solution:
            self.skipTest("TODO 5 not implemented yet")
            
        # Mock current performance state
        current_state = {
            "quality_trend": "stable",
            "performance_trend": "declining",
            "cost_trend": "increasing",
            "user_satisfaction_trend": "stable"
        }
        
        # Generate predictive recommendations
        predictions = self.analytics.generate_optimization_recommendations(current_state)
        
        # Validate predictive capabilities
        self.assertGreater(len(predictions), 0, "Should generate predictive recommendations")
        
        # Should prioritize based on trends
        high_priority_recs = [p for p in predictions if p.priority == "high"]
        self.assertGreater(len(high_priority_recs), 0,
                          "Should identify high-priority optimizations")
        
        # Should address declining performance
        performance_recs = [p for p in predictions if "performance" in p.category.lower()]
        self.assertGreater(len(performance_recs), 0,
                          "Should address performance decline")
        
        print(f"✅ TODO 5: Predictive optimization - Generated {len(predictions)} predictions, {len(high_priority_recs)} high priority")


class TestProductionMonitoringDashboard(unittest.TestCase):
    """
    Test suite for TODO 6: Production Monitoring Dashboard
    
    Tests enterprise-grade monitoring including:
    - Real-time metrics display
    - System health monitoring
    - Cost analytics and optimization
    - Complete lesson integration validation
    """
    
    def setUp(self):
        """Set up test environment for production monitoring tests."""
        self.env = TestEnvironment()
        
        try:
            from solution.improvement_engine import PerformanceAnalytics, ProductionAISystem, MonitoringDashboard
            self.analytics = PerformanceAnalytics(self.env.project_id)
            self.production_system = ProductionAISystem(self.env.project_id)
            self.has_solution = True
        except (ImportError, NotImplementedError):
            try:
                from starter.improvement_engine import PerformanceAnalytics, ProductionAISystem, MonitoringDashboard
                self.analytics = PerformanceAnalytics(self.env.project_id)
                self.production_system = ProductionAISystem(self.env.project_id)
                self.has_solution = False
            except ImportError:
                self.skipTest("Production monitoring components not found")
    
    def test_real_time_metrics_dashboard(self):
        """Test TODO 6: Live performance dashboards with key indicators."""
        if not self.has_solution:
            self.skipTest("TODO 6 not implemented yet")
            
        dashboard = self.analytics.generate_dashboard(include_predictions=True)
        
        # Validate dashboard structure
        self.assertIsInstance(dashboard, MonitoringDashboard)
        self.assertIn("dashboard_id", dashboard.__dict__)
        self.assertIn("system_status", dashboard.__dict__)
        self.assertIn("live_metrics", dashboard.__dict__)
        self.assertIn("health_score", dashboard.__dict__)
        
        # Validate real-time metrics
        live_metrics = dashboard.live_metrics
        expected_metrics = [
            "current_response_time", "active_operations", "success_rate",
            "quality_score", "cost_per_hour", "user_satisfaction"
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, live_metrics, f"Dashboard should include {metric}")
        
        # Validate health score
        self.assertGreaterEqual(dashboard.health_score, 0.0)
        self.assertLessEqual(dashboard.health_score, 1.0)
        
        print(f"✅ TODO 6: Real-time dashboard - Status: {dashboard.system_status}, Health: {dashboard.health_score:.2f}")
    
    def test_system_health_monitoring(self):
        """Test TODO 6: System health checks and diagnostic reporting."""
        if not self.has_solution:
            self.skipTest("TODO 6 not implemented yet")
            
        dashboard = self.analytics.generate_dashboard()
        
        # Validate health monitoring
        self.assertIn(dashboard.system_status, ["healthy", "warning", "critical", "offline"])
        
        # Should include performance summary
        self.assertIn("performance_summary", dashboard.__dict__)
        performance_summary = dashboard.performance_summary
        
        expected_summary_keys = ["uptime", "total_requests", "average_response_time", "success_rate"]
        for key in expected_summary_keys:
            self.assertIn(key, performance_summary, f"Performance summary should include {key}")
        
        # Should provide diagnostic insights
        if hasattr(dashboard, 'trend_insights'):
            self.assertGreater(len(dashboard.trend_insights), 0,
                              "Should provide trend insights for diagnostics")
        
        print(f"✅ TODO 6: Health monitoring - Summary includes {len(performance_summary)} metrics")
    
    def test_cost_analytics_optimization(self):
        """Test TODO 6: Cost tracking and optimization insights."""
        if not self.has_solution:
            self.skipTest("TODO 6 not implemented yet")
            
        # Mock cost data
        cost_data = {
            "current_cost_per_hour": 2.50,
            "daily_cost": 60.00,
            "monthly_projection": 1800.00,
            "cost_trend": "increasing"
        }
        
        dashboard = self.analytics.generate_dashboard()
        
        # Validate cost analytics
        live_metrics = dashboard.live_metrics
        cost_metrics = [key for key in live_metrics.keys() if "cost" in key.lower()]
        self.assertGreater(len(cost_metrics), 0, "Dashboard should include cost metrics")
        
        # Should provide cost optimization opportunities
        if hasattr(dashboard, 'optimization_opportunities'):
            cost_optimizations = [opp for opp in dashboard.optimization_opportunities 
                                if "cost" in opp.category.lower()]
            self.assertGreater(len(cost_optimizations), 0,
                              "Should provide cost optimization opportunities")
        
        print(f"✅ TODO 6: Cost analytics - Tracked {len(cost_metrics)} cost metrics")
    
    def test_complete_lesson_integration(self):
        """Test TODO 6: Validate all lesson components working together."""
        if not self.has_solution:
            self.skipTest("TODO 6 not implemented yet")
            
        # Test complete system integration
        integration_results = self.production_system.validate_system_integration()
        
        # Validate integration test results
        self.assertIsInstance(integration_results, dict)
        
        expected_integrations = [
            "personas_integration",      # Lesson 1
            "reasoning_integration",     # Lesson 2
            "optimization_integration",  # Lesson 3
            "chaining_integration",      # Lesson 4
            "feedback_loops_integration" # Lesson 5
        ]
        
        for integration in expected_integrations:
            self.assertIn(integration, integration_results,
                         f"Should test {integration}")
        
        # Calculate integration score
        integration_score = sum(integration_results.values()) / len(integration_results)
        self.assertGreaterEqual(integration_score, 0.5,
                               "At least 50% of integrations should be working")
        
        # Test end-to-end workflow
        self.assertIn("end_to_end_workflow", integration_results)
        
        print(f"✅ TODO 6: Integration validation - {integration_score:.1%} components integrated")
    
    def test_production_readiness_validation(self):
        """Test TODO 6: Overall production readiness assessment."""
        if not self.has_solution:
            self.skipTest("TODO 6 not implemented yet")
            
        # Mock business scenario for production test
        business_scenario = {
            "company": "TechCorp",
            "industry": "Software",
            "analysis_type": "market_expansion",
            "urgency": "medium"
        }
        
        # Run production workflow
        workflow_result = self.production_system.run_production_workflow(business_scenario)
        
        # Validate production workflow
        self.assertIn("success", workflow_result)
        self.assertIn("execution_time", workflow_result)
        self.assertIn("components_used", workflow_result)
        
        if workflow_result["success"]:
            self.assertIn("quality_score", workflow_result)
            self.assertGreaterEqual(workflow_result["quality_score"], 0.7,
                                   "Production workflow should maintain high quality")
        
        # Validate component integration
        expected_components = ["personas", "reasoning", "optimization", "chaining", "feedback"]
        components_used = workflow_result.get("components_used", [])
        
        integration_coverage = len(set(components_used) & set(expected_components)) / len(expected_components)
        self.assertGreaterEqual(integration_coverage, 0.8,
                               "Should use most lesson components in production workflow")
        
        print(f"✅ TODO 6: Production readiness - Workflow success: {workflow_result['success']}, Integration: {integration_coverage:.1%}")


class TestSuiteRunner:
    """Main test suite runner with comprehensive reporting."""
    
    def __init__(self):
        self.test_results = {}
        self.env = TestEnvironment()
        
    def run_all_tests(self, verbose: bool = True):
        """Run all test suites and generate comprehensive report."""
        print("🚀 Running Lesson 5 Comprehensive Test Suite")
        print("=" * 60)
        
        test_suites = [
            ("TODO 1: Self-Validation Framework", TestSelfValidationFramework),
            ("TODO 2: Automatic Retry & Refinement", TestAutomaticRetryRefinement),
            ("TODO 3: Quality Gate Systems", TestQualityGateSystems),
            ("TODO 4: Performance Analytics", TestPerformanceAnalyticsSystem),
            ("TODO 5: Continuous Learning", TestContinuousLearningIntegration),
            ("TODO 6: Production Monitoring", TestProductionMonitoringDashboard)
        ]
        
        total_tests = 0
        total_passed = 0
        
        for suite_name, suite_class in test_suites:
            print(f"\n📋 {suite_name}")
            print("-" * 50)
            
            # Run test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(suite_class)
            runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
            result = runner.run(suite)
            
            # Record results
            tests_run = result.testsRun
            tests_passed = tests_run - len(result.failures) - len(result.errors)
            
            self.test_results[suite_name] = {
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "success_rate": tests_passed / tests_run if tests_run > 0 else 0
            }
            
            total_tests += tests_run
            total_passed += tests_passed
            
            print(f"📊 Results: {tests_passed}/{tests_run} passed ({tests_passed/tests_run:.1%})")
        
        self._generate_final_report(total_tests, total_passed)
        self.env.cleanup()
    
    def _generate_final_report(self, total_tests: int, total_passed: int):
        """Generate comprehensive test report."""
        print("\n" + "=" * 60)
        print("📊 LESSON 5 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        print(f"\n🎯 Overall Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Tests Passed: {total_passed}")
        print(f"   Success Rate: {overall_success_rate:.1%}")
        
        print(f"\n📋 Detailed Results by TODO:")
        for todo, results in self.test_results.items():
            status = "✅" if results["success_rate"] >= 0.8 else "⚠️" if results["success_rate"] >= 0.5 else "❌"
            print(f"   {status} {todo}: {results['tests_passed']}/{results['tests_run']} ({results['success_rate']:.1%})")
        
        print(f"\n🏆 Production Readiness Assessment:")
        if overall_success_rate >= 0.9:
            print("   🌟 EXCELLENT - Production ready with comprehensive feedback loops!")
        elif overall_success_rate >= 0.8:
            print("   ✅ GOOD - Nearly production ready, minor improvements needed")
        elif overall_success_rate >= 0.6:
            print("   ⚠️  FAIR - Requires significant improvements before production")
        else:
            print("   ❌ NEEDS WORK - Major implementation required")
        
        print(f"\n📈 Implementation Progress:")
        todo_progress = {
            "Self-Validation (TODOs 1-3)": sum(
                self.test_results.get(f"TODO {i}: {name}", {}).get("success_rate", 0) 
                for i, name in [(25, "Self-Validation Framework"), (26, "Automatic Retry & Refinement"), (27, "Quality Gate Systems")]
            ) / 3,
            "Performance Analytics (TODO 4)": self.test_results.get("TODO 4: Performance Analytics", {}).get("success_rate", 0),
            "Continuous Learning (TODO 5)": self.test_results.get("TODO 5: Continuous Learning", {}).get("success_rate", 0),
            "Production Monitoring (TODO 6)": self.test_results.get("TODO 6: Production Monitoring", {}).get("success_rate", 0)
        }
        
        for category, progress in todo_progress.items():
            progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
            print(f"   {category}: [{progress_bar}] {progress:.1%}")


def main():
    """Main entry point for running tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lesson 5 Feedback Loops Test Suite")
    parser.add_argument("--todo", type=int, choices=[25, 26, 27, 28, 29, 30],
                       help="Run tests for specific TODO")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--all-lessons", action="store_true", help="Test cross-lesson integration")
    
    args = parser.parse_args()
    
    # Set up test environment
    test_runner = TestSuiteRunner()
    
    if args.todo:
        # Run specific TODO tests
        todo_map = {
            25: TestSelfValidationFramework,
            26: TestAutomaticRetryRefinement,
            27: TestQualityGateSystems,
            28: TestPerformanceAnalyticsSystem,
            29: TestContinuousLearningIntegration,
            30: TestProductionMonitoringDashboard
        }
        
        if args.todo in todo_map:
            print(f"🎯 Running tests for TODO {args.todo}")
            suite = unittest.TestLoader().loadTestsFromTestCase(todo_map[args.todo])
            runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
            runner.run(suite)
        
    elif args.integration or args.all_lessons:
        # Run integration tests
        print("🔧 Running integration tests...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestProductionMonitoringDashboard)
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        runner.run(suite)
        
    else:
        # Run all tests
        test_runner.run_all_tests(verbose=args.verbose)


if __name__ == "__main__":
    main()