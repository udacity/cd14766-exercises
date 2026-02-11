"""
Lesson 5: Iterative Improvement Engine - Complete Solution

This module demonstrates advanced performance analytics, continuous learning,
and production monitoring for AI systems with comprehensive feedback integration.

Learning Objectives:
- Build sophisticated performance tracking and analytics systems
- Implement continuous learning with feedback integration
- Create production-grade monitoring dashboards with real-time metrics
- Develop predictive optimization and proactive maintenance

TODOs 4-30 SOLUTIONS implemented with comprehensive analytics frameworks,
learning systems, and production monitoring capabilities.

Author: Noble Ackerson (Udacity)
Date: 2025
"""

import os
import time
import json
import statistics
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
from google import genai

# Import previous lesson components and self-validator
from self_validator import SelfValidator, ValidationResult, ValidationContext, ValidationLevel


class MetricType(Enum):
    """Types of metrics collected by the system."""
    QUALITY = "quality"
    PERFORMANCE = "performance" 
    COST = "cost"
    USER_FEEDBACK = "user_feedback"
    SYSTEM_HEALTH = "system_health"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TrendDirection(Enum):
    """Trend analysis directions."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"


@dataclass
class MetricsCollection:
    """Comprehensive metrics from a single operation."""
    timestamp: float
    operation_id: str
    operation_type: str
    quality_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    cost_metrics: Dict[str, float]
    user_feedback: Dict[str, Any]
    system_context: Dict[str, Any]
    execution_time: float
    token_usage: Dict[str, int]


@dataclass
class TrendAnalysis:
    """Analysis of performance trends over time."""
    metric_name: str
    trend_direction: TrendDirection
    trend_strength: float  # 0.0 to 1.0
    current_value: float
    historical_average: float
    change_rate: float
    prediction_next_period: float
    confidence: float
    anomalies_detected: List[Dict[str, Any]]
    recommendations: List[str]


@dataclass
class AlertRule:
    """Configuration for automated alerting."""
    rule_id: str
    name: str
    metric_path: str
    condition: str  # "greater_than", "less_than", "change_rate", etc.
    threshold: float
    severity: AlertSeverity
    cooldown_seconds: int = 300
    enabled: bool = True
    last_triggered: Optional[float] = None


@dataclass
class Alert:
    """Generated alert from monitoring system."""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    message: str
    timestamp: float
    metric_value: float
    threshold: float
    context: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class MonitoringDashboard:
    """Real-time monitoring dashboard data."""
    timestamp: float
    system_status: str
    live_metrics: Dict[str, Any]
    health_indicators: Dict[str, float]
    active_alerts: List[Alert]
    performance_summary: Dict[str, Any]
    cost_analysis: Dict[str, Any]
    quality_trends: Dict[str, TrendAnalysis]
    recommendations: List[str]


@dataclass
class LearningInsight:
    """Insight generated from continuous learning."""
    insight_id: str
    timestamp: float
    insight_type: str
    description: str
    evidence: List[Dict[str, Any]]
    confidence: float
    actionable_recommendations: List[str]
    estimated_impact: float


class PerformanceAnalytics:
    """
    Sophisticated performance analytics system with trend analysis,
    anomaly detection, and optimization recommendations.
    """
    
    def __init__(self, project_id: str):
        """Initialize performance analytics system."""
        self.project_id = project_id
        self.metrics_history = deque(maxlen=10000)  # Keep last 10k metrics
        self.alert_rules = {}
        self.active_alerts = []
        self.trend_cache = {}
        self.learning_insights = []
        
        # Performance tracking
        self.operation_counters = defaultdict(int)
        self.performance_baselines = {}
        self.anomaly_threshold = 0.15  # 15% deviation for anomaly detection
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup default alert rules
        self._setup_default_alert_rules()
    
    def collect_metrics(self, operation_type: str, execution_result: Dict[str, Any], 
                       validation_result: Optional[ValidationResult] = None) -> MetricsCollection:
        """
        TODO 4 SOLUTION: Implement comprehensive metrics collection system.
        
        This implementation provides:
        - Multi-dimensional metrics collection across all performance areas
        - Real-time data aggregation and storage
        - Integration with validation results for quality tracking
        - Contextual metadata for comprehensive analysis
        """
        operation_id = f"{operation_type}_{int(time.time())}_{hash(str(execution_result)) % 10000}"
        
        # Quality metrics from validation
        quality_metrics = {}
        if validation_result:
            quality_metrics = {
                "overall_confidence": validation_result.overall_confidence,
                "content_quality": validation_result.confidence_breakdown.content_quality,
                "context_relevance": validation_result.confidence_breakdown.context_relevance,
                "factual_accuracy": validation_result.confidence_breakdown.factual_accuracy,
                "structural_coherence": validation_result.confidence_breakdown.structural_coherence,
                "completeness": validation_result.confidence_breakdown.completeness,
                "validation_passed": 1.0 if validation_result.validation_passed else 0.0,
                "issues_count": len(validation_result.detected_issues),
                "critical_issues": len([i for i in validation_result.detected_issues if i.severity == "critical"]),
                "processing_time": validation_result.processing_time
            }
        elif "quality_score" in execution_result:
            # Fallback: use quality_score from operation if no validation_result provided
            quality_score = execution_result.get("quality_score", 0.0)
            quality_metrics = {
                "overall_confidence": quality_score,
                "validation_passed": 1.0 if quality_score >= 0.75 else 0.0,
                "issues_count": 0 if quality_score >= 0.75 else int((0.75 - quality_score) * 10),
                "critical_issues": 0
            }
        
        # Performance metrics
        performance_metrics = {
            "execution_time": execution_result.get("execution_time", 0.0),
            "response_length": len(str(execution_result.get("content", ""))),
            "token_efficiency": self._calculate_token_efficiency(execution_result),
            "memory_usage": execution_result.get("memory_usage", 0.0),
            "cpu_utilization": execution_result.get("cpu_utilization", 0.0),
            "api_calls_made": execution_result.get("api_calls", 1),
            "retry_count": execution_result.get("retry_count", 0),
            "cache_hit_rate": execution_result.get("cache_hit_rate", 0.0)
        }
        
        # Cost metrics
        token_usage = execution_result.get("token_usage", {})
        total_tokens = token_usage.get("input_tokens", 0) + token_usage.get("output_tokens", 0)
        cost_per_1k_tokens = 0.0002  # Gemini 2.5 Flash pricing
        
        cost_metrics = {
            "total_tokens": total_tokens,
            "input_tokens": token_usage.get("input_tokens", 0),
            "output_tokens": token_usage.get("output_tokens", 0),
            "estimated_cost": (total_tokens / 1000) * cost_per_1k_tokens,
            "cost_per_quality_point": 0.0,
            "token_per_word": 0.0
        }
        
        # Calculate derived cost metrics
        if quality_metrics.get("overall_confidence", 0) > 0:
            cost_metrics["cost_per_quality_point"] = cost_metrics["estimated_cost"] / quality_metrics["overall_confidence"]
        
        word_count = len(str(execution_result.get("content", "")).split())
        if word_count > 0:
            cost_metrics["token_per_word"] = total_tokens / word_count
        
        # User feedback (simulated - would come from real user interactions)
        user_feedback = {
            "satisfaction_score": execution_result.get("user_satisfaction", 0.0),
            "usefulness_rating": execution_result.get("usefulness", 0.0),
            "clarity_rating": execution_result.get("clarity", 0.0),
            "completeness_rating": execution_result.get("completeness_rating", 0.0),
            "feedback_provided": execution_result.get("has_feedback", False)
        }
        
        # System context
        system_context = {
            "operation_count": self.operation_counters[operation_type],
            "system_load": execution_result.get("system_load", 0.0),
            "concurrent_operations": execution_result.get("concurrent_ops", 1),
            "model_version": execution_result.get("model_version", "gemini-2.5-flash"),
            "optimization_applied": execution_result.get("optimization_applied", False),
            "lesson_integration": execution_result.get("lesson_integration", [])
        }
        
        metrics = MetricsCollection(
            timestamp=time.time(),
            operation_id=operation_id,
            operation_type=operation_type,
            quality_metrics=quality_metrics,
            performance_metrics=performance_metrics,
            cost_metrics=cost_metrics,
            user_feedback=user_feedback,
            system_context=system_context,
            execution_time=performance_metrics["execution_time"],
            token_usage=token_usage
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        self.operation_counters[operation_type] += 1
        
        # Check alert rules
        self._check_alert_rules(metrics)
        
        if len(self.metrics_history) % 10 == 0:  # Log every 10th operation
            self.logger.info(f"Collected metrics for {operation_type}: quality={quality_metrics.get('overall_confidence', 0):.3f}")
        
        return metrics
    
    def _calculate_token_efficiency(self, execution_result: Dict[str, Any]) -> float:
        """Calculate token usage efficiency."""
        token_usage = execution_result.get("token_usage", {})
        total_tokens = token_usage.get("input_tokens", 0) + token_usage.get("output_tokens", 0)
        
        if total_tokens == 0:
            return 0.0
        
        # Efficiency based on output quality per token
        quality_score = execution_result.get("quality_score", 0.5)
        return quality_score / (total_tokens / 1000)  # Quality per 1k tokens
    
    def analyze_trends(self, metric_path: str, time_window_hours: int = 24) -> TrendAnalysis:
        """
        TODO 5 SOLUTION: Implement sophisticated trend analysis and learning.
        
        This implementation provides:
        - Multi-timeframe trend analysis with statistical validation
        - Anomaly detection using statistical methods
        - Predictive modeling for future performance
        - Actionable optimization recommendations
        """
        if not self.metrics_history:
            return TrendAnalysis(
                metric_name=metric_path,
                trend_direction=TrendDirection.STABLE,
                trend_strength=0.0,
                current_value=0.0,
                historical_average=0.0,
                change_rate=0.0,
                prediction_next_period=0.0,
                confidence=0.0,
                anomalies_detected=[],
                recommendations=["Insufficient data for trend analysis"]
            )
        
        # Filter metrics by time window
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if len(recent_metrics) < 5:
            return TrendAnalysis(
                metric_name=metric_path,
                trend_direction=TrendDirection.STABLE,
                trend_strength=0.0,
                current_value=0.0,
                historical_average=0.0,
                change_rate=0.0,
                prediction_next_period=0.0,
                confidence=0.0,
                anomalies_detected=[],
                recommendations=["Insufficient recent data for trend analysis"]
            )
        
        # Extract metric values
        values = []
        timestamps = []
        
        for metrics in recent_metrics:
            value = self._extract_metric_value(metrics, metric_path)
            if value is not None:
                values.append(value)
                timestamps.append(metrics.timestamp)
        
        if len(values) < 3:
            return TrendAnalysis(
                metric_name=metric_path,
                trend_direction=TrendDirection.STABLE,
                trend_strength=0.0,
                current_value=0.0,
                historical_average=0.0,
                change_rate=0.0,
                prediction_next_period=0.0,
                confidence=0.0,
                anomalies_detected=[],
                recommendations=[f"Metric {metric_path} not found in recent data"]
            )
        
        # Statistical analysis
        current_value = values[-1]
        historical_average = statistics.mean(values)
        
        # Trend analysis using linear regression (simplified)
        trend_slope = self._calculate_trend_slope(values, timestamps)
        
        # Determine trend direction and strength
        trend_direction = TrendDirection.STABLE
        trend_strength = abs(trend_slope)
        
        if trend_slope > 0.05:
            trend_direction = TrendDirection.IMPROVING
        elif trend_slope < -0.05:
            trend_direction = TrendDirection.DECLINING
        
        # Check for volatility
        if len(values) >= 5:
            volatility = statistics.stdev(values) / historical_average if historical_average > 0 else 0
            if volatility > 0.3:
                trend_direction = TrendDirection.VOLATILE
        
        # Calculate change rate
        if len(values) >= 2:
            change_rate = (values[-1] - values[0]) / values[0] if values[0] != 0 else 0.0
        else:
            change_rate = 0.0
        
        # Prediction (simple linear extrapolation)
        prediction_next_period = current_value + trend_slope
        
        # Confidence based on data quality and consistency
        confidence = min(len(values) / 20, 1.0)  # More data = higher confidence
        if trend_direction == TrendDirection.VOLATILE:
            confidence *= 0.5
        
        # Anomaly detection
        anomalies = self._detect_anomalies(values, timestamps)
        
        # Generate recommendations
        recommendations = self._generate_trend_recommendations(
            metric_path, trend_direction, trend_strength, current_value, historical_average, anomalies
        )
        
        return TrendAnalysis(
            metric_name=metric_path,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            current_value=current_value,
            historical_average=historical_average,
            change_rate=change_rate,
            prediction_next_period=prediction_next_period,
            confidence=confidence,
            anomalies_detected=anomalies,
            recommendations=recommendations
        )
    
    def _extract_metric_value(self, metrics: MetricsCollection, metric_path: str) -> Optional[float]:
        """Extract metric value from metrics collection using dot notation path."""
        path_parts = metric_path.split('.')
        current = asdict(metrics)
        
        try:
            for part in path_parts:
                current = current[part]
            return float(current) if current is not None else None
        except (KeyError, TypeError, ValueError):
            return None
    
    def _calculate_trend_slope(self, values: List[float], timestamps: List[float]) -> float:
        """Calculate trend slope using simple linear regression."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        
        # Normalize timestamps to start from 0
        time_normalized = [(t - timestamps[0]) / 3600 for t in timestamps]  # Convert to hours
        
        # Calculate slope using least squares
        sum_x = sum(time_normalized)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(time_normalized, values))
        sum_x2 = sum(x * x for x in time_normalized)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _detect_anomalies(self, values: List[float], timestamps: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies in metric values."""
        if len(values) < 5:
            return []
        
        anomalies = []
        mean_val = statistics.mean(values)
        
        if len(values) >= 3:
            std_val = statistics.stdev(values)
            threshold = std_val * 2  # 2 standard deviations
            
            for i, (value, timestamp) in enumerate(zip(values, timestamps)):
                if abs(value - mean_val) > threshold:
                    anomalies.append({
                        "timestamp": timestamp,
                        "value": value,
                        "expected_range": [mean_val - threshold, mean_val + threshold],
                        "deviation": abs(value - mean_val),
                        "severity": "high" if abs(value - mean_val) > threshold * 1.5 else "medium"
                    })
        
        return anomalies
    
    def _generate_trend_recommendations(self, metric_path: str, trend_direction: TrendDirection,
                                      trend_strength: float, current_value: float,
                                      historical_average: float, anomalies: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on trend analysis."""
        recommendations = []
        
        # Trend-based recommendations
        if trend_direction == TrendDirection.DECLINING:
            if "quality" in metric_path.lower():
                recommendations.append("Quality declining - review prompt optimization and validation thresholds")
            elif "performance" in metric_path.lower():
                recommendations.append("Performance declining - check system resources and optimization settings")
            elif "cost" in metric_path.lower():
                recommendations.append("Costs increasing - review token usage and optimization strategies")
        
        elif trend_direction == TrendDirection.IMPROVING:
            recommendations.append(f"Positive trend in {metric_path} - maintain current practices")
        
        elif trend_direction == TrendDirection.VOLATILE:
            recommendations.append(f"High volatility in {metric_path} - investigate root causes and stabilize processes")
        
        # Value-based recommendations
        if current_value < historical_average * 0.8:
            recommendations.append(f"Current {metric_path} significantly below average - immediate attention required")
        
        # Anomaly-based recommendations
        if anomalies:
            high_severity_anomalies = [a for a in anomalies if a["severity"] == "high"]
            if high_severity_anomalies:
                recommendations.append(f"Critical anomalies detected in {metric_path} - investigate immediately")
        
        # Specific metric recommendations
        if "confidence" in metric_path.lower() and current_value < 0.7:
            recommendations.append("Low confidence scores - enhance validation criteria and retry mechanisms")
        
        if "cost" in metric_path.lower() and trend_strength > 0.1:
            recommendations.append("Implement cost optimization strategies and token usage monitoring")
        
        return recommendations[:3]  # Limit to top 3 recommendations
    
    def generate_dashboard(self, include_predictions: bool = True) -> MonitoringDashboard:
        """
        TODO 6 SOLUTION: Generate comprehensive production monitoring dashboard.
        
        This implementation provides:
        - Real-time system health monitoring with key performance indicators
        - Live metrics dashboard with critical operational data
        - Automated alerting system with severity-based notifications
        - Performance insights and optimization recommendations
        """
        timestamp = time.time()
        
        # Calculate system status
        system_status = self._calculate_system_status()
        
        # Generate live metrics
        live_metrics = self._generate_live_metrics()
        
        # Calculate health indicators
        health_indicators = self._calculate_health_indicators()
        
        # Get active alerts
        active_alerts = [alert for alert in self.active_alerts if not alert.resolved]
        
        # Generate performance summary
        performance_summary = self._generate_performance_summary()
        
        # Cost analysis
        cost_analysis = self._generate_cost_analysis()
        
        # Quality trends
        quality_trends = {}
        key_quality_metrics = [
            "quality_metrics.overall_confidence",
            "quality_metrics.validation_passed",
            "performance_metrics.execution_time",
            "cost_metrics.estimated_cost"
        ]
        
        for metric in key_quality_metrics:
            try:
                trend = self.analyze_trends(metric, time_window_hours=6)
                quality_trends[metric] = trend
            except Exception as e:
                self.logger.warning(f"Failed to analyze trend for {metric}: {e}")
        
        # Generate dashboard recommendations
        recommendations = self._generate_dashboard_recommendations(
            system_status, health_indicators, active_alerts, quality_trends
        )
        
        return MonitoringDashboard(
            timestamp=timestamp,
            system_status=system_status,
            live_metrics=live_metrics,
            health_indicators=health_indicators,
            active_alerts=active_alerts,
            performance_summary=performance_summary,
            cost_analysis=cost_analysis,
            quality_trends=quality_trends,
            recommendations=recommendations
        )
    
    def _calculate_system_status(self) -> str:
        """Calculate overall system status."""
        if not self.metrics_history:
            return "INITIALIZING"
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 operations
        
        # Check for critical alerts
        critical_alerts = [a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL and not a.resolved]
        if critical_alerts:
            return "CRITICAL"
        
        # Check success rate
        if recent_metrics:
            success_rate = sum(1 for m in recent_metrics 
                             if m.quality_metrics.get("validation_passed", 0) > 0) / len(recent_metrics)
            
            if success_rate < 0.8:
                return "DEGRADED"
            elif success_rate < 0.95:
                return "WARNING"
        
        # Check for error alerts
        error_alerts = [a for a in self.active_alerts if a.severity == AlertSeverity.ERROR and not a.resolved]
        if error_alerts:
            return "WARNING"
        
        return "HEALTHY"
    
    def _generate_live_metrics(self) -> Dict[str, Any]:
        """Generate live metrics for dashboard."""
        if not self.metrics_history:
            return {"status": "No data available"}
        
        recent_metrics = list(self.metrics_history)[-20:]  # Last 20 operations
        
        live_metrics = {}
        
        # Quality metrics
        if recent_metrics:
            confidence_scores = [m.quality_metrics.get("overall_confidence", 0) for m in recent_metrics]
            live_metrics["average_confidence"] = statistics.mean(confidence_scores)
            live_metrics["min_confidence"] = min(confidence_scores)
            live_metrics["max_confidence"] = max(confidence_scores)
            
            validation_passes = [m.quality_metrics.get("validation_passed", 0) for m in recent_metrics]
            live_metrics["success_rate"] = statistics.mean(validation_passes)
        
        # Performance metrics
        execution_times = [m.performance_metrics.get("execution_time", 0) for m in recent_metrics]
        if execution_times:
            live_metrics["avg_execution_time"] = statistics.mean(execution_times)
            live_metrics["p95_execution_time"] = sorted(execution_times)[int(len(execution_times) * 0.95)]
        
        # Cost metrics
        costs = [m.cost_metrics.get("estimated_cost", 0) for m in recent_metrics]
        if costs:
            live_metrics["total_cost_recent"] = sum(costs)
            live_metrics["avg_cost_per_operation"] = statistics.mean(costs)
        
        # System metrics
        live_metrics["total_operations"] = len(self.metrics_history)
        live_metrics["operations_last_hour"] = len([m for m in self.metrics_history 
                                                   if m.timestamp >= time.time() - 3600])
        live_metrics["active_alerts_count"] = len([a for a in self.active_alerts if not a.resolved])
        
        return live_metrics
    
    def _calculate_health_indicators(self) -> Dict[str, float]:
        """Calculate system health indicators (0.0 to 1.0)."""
        if not self.metrics_history:
            return {"overall_health": 0.0}
        
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 operations
        
        health_indicators = {}
        
        # Quality health
        confidence_scores = [m.quality_metrics.get("overall_confidence", 0) for m in recent_metrics]
        if confidence_scores:
            health_indicators["quality_health"] = statistics.mean(confidence_scores)
        
        # Performance health
        execution_times = [m.performance_metrics.get("execution_time", 0) for m in recent_metrics]
        if execution_times:
            avg_time = statistics.mean(execution_times)
            # Health decreases as execution time increases (assuming 5s is baseline)
            health_indicators["performance_health"] = max(0.0, min(1.0, (5.0 - avg_time) / 5.0))
        
        # Cost health (based on efficiency)
        cost_efficiency = [m.cost_metrics.get("cost_per_quality_point", float('inf')) for m in recent_metrics]
        cost_efficiency = [c for c in cost_efficiency if c != float('inf')]
        if cost_efficiency:
            avg_efficiency = statistics.mean(cost_efficiency)
            # Health is better with lower cost per quality point (assuming 0.1 is good)
            health_indicators["cost_health"] = max(0.0, min(1.0, (0.1 - avg_efficiency) / 0.1 + 0.5))
        
        # System stability (based on alert frequency)
        recent_alerts = [a for a in self.active_alerts if a.timestamp >= time.time() - 3600]
        alert_penalty = min(len(recent_alerts) * 0.1, 0.5)
        health_indicators["stability_health"] = max(0.0, 1.0 - alert_penalty)
        
        # Overall health (weighted average)
        weights = {"quality_health": 0.4, "performance_health": 0.3, "cost_health": 0.2, "stability_health": 0.1}
        overall_health = 0.0
        total_weight = 0.0
        
        for indicator, weight in weights.items():
            if indicator in health_indicators:
                overall_health += health_indicators[indicator] * weight
                total_weight += weight
        
        if total_weight > 0:
            health_indicators["overall_health"] = overall_health / total_weight
        else:
            health_indicators["overall_health"] = 0.0
        
        return health_indicators
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary statistics."""
        if not self.metrics_history:
            return {"status": "No performance data"}
        
        # Time windows
        hour_ago = time.time() - 3600
        day_ago = time.time() - 86400
        
        recent_hour = [m for m in self.metrics_history if m.timestamp >= hour_ago]
        recent_day = [m for m in self.metrics_history if m.timestamp >= day_ago]
        
        summary = {}
        
        # Operations count
        summary["operations_last_hour"] = len(recent_hour)
        summary["operations_last_day"] = len(recent_day)
        
        # Success rates
        if recent_hour:
            summary["success_rate_hour"] = sum(1 for m in recent_hour 
                                             if m.quality_metrics.get("validation_passed", 0)) / len(recent_hour)
        
        if recent_day:
            summary["success_rate_day"] = sum(1 for m in recent_day 
                                            if m.quality_metrics.get("validation_passed", 0)) / len(recent_day)
        
        # Performance statistics
        if recent_day:
            exec_times = [m.performance_metrics.get("execution_time", 0) for m in recent_day]
            summary["avg_execution_time"] = statistics.mean(exec_times)
            summary["median_execution_time"] = statistics.median(exec_times)
            summary["p95_execution_time"] = sorted(exec_times)[int(len(exec_times) * 0.95)]
        
        return summary
    
    def _generate_cost_analysis(self) -> Dict[str, Any]:
        """Generate cost analysis and optimization insights."""
        if not self.metrics_history:
            return {"status": "No cost data"}
        
        day_ago = time.time() - 86400
        recent_day = [m for m in self.metrics_history if m.timestamp >= day_ago]
        
        if not recent_day:
            return {"status": "No recent cost data"}
        
        analysis = {}
        
        # Total costs
        total_cost = sum(m.cost_metrics.get("estimated_cost", 0) for m in recent_day)
        analysis["total_cost_24h"] = total_cost
        
        # Cost breakdown
        total_tokens = sum(m.cost_metrics.get("total_tokens", 0) for m in recent_day)
        analysis["total_tokens_24h"] = total_tokens
        analysis["avg_tokens_per_operation"] = total_tokens / len(recent_day) if recent_day else 0
        
        # Efficiency metrics
        cost_per_ops = [m.cost_metrics.get("cost_per_quality_point", 0) for m in recent_day]
        cost_per_ops = [c for c in cost_per_ops if c > 0]
        if cost_per_ops:
            analysis["avg_cost_efficiency"] = statistics.mean(cost_per_ops)
            analysis["best_cost_efficiency"] = min(cost_per_ops)
            analysis["worst_cost_efficiency"] = max(cost_per_ops)
        
        # Optimization opportunities
        analysis["optimization_opportunities"] = []
        
        if total_tokens > 0:
            high_token_ops = [m for m in recent_day if m.cost_metrics.get("total_tokens", 0) > total_tokens / len(recent_day) * 1.5]
            if high_token_ops:
                analysis["optimization_opportunities"].append(
                    f"Token optimization: {len(high_token_ops)} operations used 50% more tokens than average"
                )
        
        return analysis
    
    def _generate_dashboard_recommendations(self, system_status: str, health_indicators: Dict[str, float],
                                          active_alerts: List[Alert], quality_trends: Dict[str, TrendAnalysis]) -> List[str]:
        """Generate actionable recommendations for the dashboard."""
        recommendations = []
        
        # System status recommendations
        if system_status == "CRITICAL":
            recommendations.append("🚨 CRITICAL: Immediate action required - check active alerts and system health")
        elif system_status == "DEGRADED":
            recommendations.append("⚠️ DEGRADED: System performance below acceptable levels - investigate recent changes")
        elif system_status == "WARNING":
            recommendations.append("⚠️ WARNING: Monitor closely - address active alerts to prevent degradation")
        
        # Health indicator recommendations
        overall_health = health_indicators.get("overall_health", 0.0)
        if overall_health < 0.7:
            recommendations.append(f"Health score low ({overall_health:.1%}) - review quality, performance, and cost metrics")
        
        quality_health = health_indicators.get("quality_health", 0.0)
        if quality_health < 0.8:
            recommendations.append("Quality health declining - review validation thresholds and retry mechanisms")
        
        # Trend-based recommendations
        for metric_path, trend in quality_trends.items():
            if trend.trend_direction == TrendDirection.DECLINING and trend.confidence > 0.6:
                recommendations.append(f"Declining trend in {metric_path.split('.')[-1]} - investigate causes")
        
        # Alert-based recommendations
        if len(active_alerts) > 5:
            recommendations.append("High alert volume - review alert thresholds and address underlying issues")
        
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            recommendations.append(f"Resolve {len(critical_alerts)} critical alerts immediately")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules for common monitoring scenarios."""
        default_rules = [
            AlertRule(
                rule_id="quality_degradation",
                name="Quality Degradation",
                metric_path="quality_metrics.overall_confidence",
                condition="less_than",
                threshold=0.6,
                severity=AlertSeverity.WARNING,
                cooldown_seconds=300
            ),
            AlertRule(
                rule_id="critical_quality_failure",
                name="Critical Quality Failure",
                metric_path="quality_metrics.overall_confidence",
                condition="less_than",
                threshold=0.4,
                severity=AlertSeverity.CRITICAL,
                cooldown_seconds=60
            ),
            AlertRule(
                rule_id="high_execution_time",
                name="High Execution Time",
                metric_path="performance_metrics.execution_time",
                condition="greater_than",
                threshold=10.0,
                severity=AlertSeverity.WARNING,
                cooldown_seconds=600
            ),
            AlertRule(
                rule_id="cost_spike",
                name="Cost Spike",
                metric_path="cost_metrics.estimated_cost",
                condition="greater_than",
                threshold=0.1,
                severity=AlertSeverity.WARNING,
                cooldown_seconds=900
            ),
            AlertRule(
                rule_id="validation_failure_rate",
                name="High Validation Failure Rate",
                metric_path="quality_metrics.validation_passed",
                condition="less_than",
                threshold=0.8,
                severity=AlertSeverity.ERROR,
                cooldown_seconds=300
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    def _check_alert_rules(self, metrics: MetricsCollection):
        """Check all alert rules against new metrics."""
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule.last_triggered and time.time() - rule.last_triggered < rule.cooldown_seconds:
                continue
            
            # Extract metric value
            metric_value = self._extract_metric_value(metrics, rule.metric_path)
            if metric_value is None:
                continue
            
            # Check condition
            triggered = False
            if rule.condition == "greater_than" and metric_value > rule.threshold:
                triggered = True
            elif rule.condition == "less_than" and metric_value < rule.threshold:
                triggered = True
            
            if triggered:
                alert = Alert(
                    alert_id=f"{rule.rule_id}_{int(time.time())}",
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    message=f"{rule.name}: {rule.metric_path} = {metric_value:.3f} (threshold: {rule.threshold})",
                    timestamp=time.time(),
                    metric_value=metric_value,
                    threshold=rule.threshold,
                    context={"operation_id": metrics.operation_id, "operation_type": metrics.operation_type}
                )
                
                self.active_alerts.append(alert)
                rule.last_triggered = time.time()
                
                self.logger.warning(f"ALERT: {alert.message}")


class ContinuousLearningEngine:
    """
    Advanced continuous learning system that processes feedback,
    identifies patterns, and generates actionable insights.
    """
    
    def __init__(self, analytics: PerformanceAnalytics):
        """Initialize continuous learning engine."""
        self.analytics = analytics
        self.learning_insights = []
        self.feedback_buffer = deque(maxlen=1000)
        self.pattern_cache = {}
        
    def process_feedback(self, feedback: Dict[str, Any]) -> Optional[LearningInsight]:
        """Process user feedback and generate learning insights."""
        self.feedback_buffer.append(feedback)

        # Analyze patterns in feedback
        insights = self._analyze_feedback_patterns()

        if insights:
            self.learning_insights.extend(insights)
            return insights[0]  # Return most recent insight

        return None
    
    def _analyze_feedback_patterns(self) -> List[LearningInsight]:
        """Analyze patterns in accumulated feedback."""
        if len(self.feedback_buffer) < 10:
            return []
        
        insights = []
        
        # Analyze satisfaction trends
        satisfaction_scores = [f.get("satisfaction", 0) for f in self.feedback_buffer if "satisfaction" in f]
        if len(satisfaction_scores) >= 5:
            recent_satisfaction = statistics.mean(satisfaction_scores[-5:])
            historical_satisfaction = statistics.mean(satisfaction_scores[:-5])
            
            if recent_satisfaction < historical_satisfaction * 0.9:
                insights.append(LearningInsight(
                    insight_id=f"satisfaction_decline_{int(time.time())}",
                    timestamp=time.time(),
                    insight_type="satisfaction_decline",
                    description="User satisfaction declining in recent feedback",
                    evidence=[{"recent_avg": recent_satisfaction, "historical_avg": historical_satisfaction}],
                    confidence=0.8,
                    actionable_recommendations=["Review recent changes to prompts or validation", "Analyze specific feedback comments"],
                    estimated_impact=0.3
                ))
        
        return insights


def run_improvement_engine_test():
    """Demonstrate improvement engine capabilities."""
    PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id")
    if PROJECT_ID == "your-project-id":
        print("❌ Please set PROJECT_ID environment variable")
        return
    
    print("\n" + "="*60)
    print("  PERFORMANCE ANALYTICS ENGINE")
    print("  Production Monitoring & Continuous Improvement")
    print("="*60)
    
    # Initialize systems
    analytics = PerformanceAnalytics(PROJECT_ID)
    learning_engine = ContinuousLearningEngine(analytics)
    
    # Simulate some operations with metrics
    test_operations = [
        {
            "execution_time": 2.5,
            "content": "High quality business analysis with comprehensive insights and strategic recommendations.",
            "token_usage": {"input_tokens": 150, "output_tokens": 200},
            "quality_score": 0.85,
            "user_satisfaction": 4.2
        },
        {
            "execution_time": 4.1,
            "content": "Brief analysis.",
            "token_usage": {"input_tokens": 100, "output_tokens": 50},
            "quality_score": 0.45,
            "user_satisfaction": 2.1
        },
        {
            "execution_time": 1.8,
            "content": "Comprehensive market analysis with detailed competitive landscape review and strategic planning recommendations.",
            "token_usage": {"input_tokens": 180, "output_tokens": 300},
            "quality_score": 0.92,
            "user_satisfaction": 4.8
        }
    ]
    
    print(f"\n{'='*60}")
    print("  TODO 4: COMPREHENSIVE METRICS COLLECTION")
    print("="*60)
    print("Collecting metrics from 3 operations...")
    for i, operation in enumerate(test_operations, 1):
        metrics = analytics.collect_metrics(f"test_operation_{i}", operation)
        print(f"  Operation {i}: Quality={metrics.quality_metrics.get('overall_confidence', 0):.3f}, Cost=${metrics.cost_metrics.get('estimated_cost', 0):.4f}")

    print(f"\n{'='*60}")
    print("  TODO 5: TREND ANALYSIS & PREDICTIONS")
    print("="*60)
    trend = analytics.analyze_trends("quality_metrics.overall_confidence")
    print(f"  Quality Trend: {trend.trend_direction.value} (strength: {trend.trend_strength:.3f})")
    print(f"  Current: {trend.current_value:.3f}, Average: {trend.historical_average:.3f}")
    
    performance_trend = analytics.analyze_trends("performance_metrics.execution_time")
    print(f"  Performance Trend: {performance_trend.trend_direction.value}")
    
    print(f"\n{'='*60}")
    print("  TODO 6: REAL-TIME MONITORING DASHBOARD")
    print("="*60)
    dashboard = analytics.generate_dashboard()

    status_emoji = {"HEALTHY": "🟢", "WARNING": "🟡", "DEGRADED": "🔴"}.get(dashboard.system_status, "⚪")
    print(f"System Status: {status_emoji} {dashboard.system_status}")
    print(f"Overall Health: {dashboard.health_indicators.get('overall_health', 0):.1%}")
    print(f"Active Alerts: {len(dashboard.active_alerts)}")
    print(f"Operations (last hour): {dashboard.live_metrics.get('operations_last_hour', 0)}")

    if dashboard.recommendations:
        print("\nTop Recommendations:")
        for rec in dashboard.recommendations[:2]:
            print(f"  • {rec}")

    print(f"\n{'='*60}")
    print("  PERFORMANCE STATISTICS")
    print("="*60)
    if hasattr(analytics, 'operation_counters'):
        total_ops = sum(analytics.operation_counters.values())
        print(f"  Total Operations: {total_ops}")
    
    if dashboard.live_metrics:
        avg_confidence = dashboard.live_metrics.get('average_confidence', 0)
        success_rate = dashboard.live_metrics.get('success_rate', 0)
        print(f"  Average Confidence: {avg_confidence:.3f}")
        print(f"  Success Rate: {success_rate:.1%}")
    
    print(f"\n{'='*60}")
    print("  ✅ ALL TODOS IMPLEMENTED")
    print("="*60)
    print("✅ TODO 4: Performance analytics system")
    print("✅ TODO 5: Continuous learning integration")
    print("✅ TODO 6: Production monitoring dashboard")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_improvement_engine_test()