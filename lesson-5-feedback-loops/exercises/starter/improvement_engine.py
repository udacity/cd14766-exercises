"""
Lesson 5: Feedback Loops and Iterative Improvement - Improvement Engine (Starter)

Author: Noble Ackerson (Udacity)
Date: 2025

This module implements performance analytics, continuous learning, and production monitoring
for AI systems with feedback loops and iterative improvement capabilities.

Learning Objectives:
- Implement comprehensive performance analytics with trend detection
- Build continuous learning systems that adapt based on feedback
- Create production monitoring dashboards with real-time metrics
- Integrate cross-lesson components for complete AI system lifecycle
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import defaultdict, deque

from google import genai


class MetricType(Enum):
    """Types of metrics collected by the system."""
    QUALITY = "quality"
    PERFORMANCE = "performance" 
    COST = "cost"
    USER_FEEDBACK = "user_feedback"
    SYSTEM_HEALTH = "system_health"


class AlertSeverity(Enum):
    """Alert severity levels for monitoring system."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MetricsCollection:
    """Comprehensive metrics collection for a single operation."""
    timestamp: datetime
    operation_type: str
    quality_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    cost_metrics: Dict[str, float]
    user_feedback_metrics: Dict[str, float]
    system_health_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Analysis of performance trends over time."""
    metric_name: str
    trend_direction: str  # "improving", "degrading", "stable"
    trend_strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    data_points: int
    time_period: timedelta
    recommendations: List[str] = field(default_factory=list)


@dataclass
class PerformanceAlert:
    """Alert for performance issues or anomalies."""
    alert_id: str
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    suggested_actions: List[str] = field(default_factory=list)


@dataclass
class OptimizationRecommendation:
    """AI-generated optimization recommendation."""
    recommendation_id: str
    category: str  # "prompt", "parameters", "architecture", "workflow"
    priority: str  # "low", "medium", "high", "critical"
    description: str
    expected_impact: str
    implementation_effort: str
    confidence: float
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringDashboard:
    """Real-time monitoring dashboard data."""
    dashboard_id: str
    generated_at: datetime
    system_status: str  # "healthy", "warning", "critical", "offline"
    live_metrics: Dict[str, float]
    performance_summary: Dict[str, Any]
    active_alerts: List[PerformanceAlert]
    trend_insights: List[TrendAnalysis]
    optimization_opportunities: List[OptimizationRecommendation]
    health_score: float  # 0.0 to 1.0


class PerformanceAnalytics:
    """
    Advanced performance analytics system for AI operations.
    
    This class implements sophisticated performance tracking, trend analysis,
    and optimization recommendations for production AI systems.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize the performance analytics system."""
        self.project_id = project_id
        self.location = location
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel("gemini-2.5-flash")
        
        # Metrics storage (in production, use proper database)
        self.metrics_history: List[MetricsCollection] = []
        self.trends_cache: Dict[str, TrendAnalysis] = {}
        self.active_alerts: List[PerformanceAlert] = []
        
        # Configuration
        self.max_history_size = 10000
        self.trend_analysis_window = timedelta(hours=24)
        self.alert_thresholds = self._initialize_alert_thresholds()
        
    def _initialize_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize default alert thresholds for various metrics."""
        return {
            "quality_metrics": {
                "overall_confidence": {"warning": 0.7, "critical": 0.5},
                "content_quality": {"warning": 0.75, "critical": 0.6},
                "validation_passed_rate": {"warning": 0.8, "critical": 0.6}
            },
            "performance_metrics": {
                "response_time": {"warning": 5.0, "critical": 10.0},
                "processing_time": {"warning": 3.0, "critical": 6.0},
                "retry_rate": {"warning": 0.2, "critical": 0.4}
            },
            "cost_metrics": {
                "cost_per_request": {"warning": 0.1, "critical": 0.2},
                "token_usage": {"warning": 5000, "critical": 10000}
            },
            "system_health_metrics": {
                "error_rate": {"warning": 0.05, "critical": 0.1},
                "availability": {"warning": 0.95, "critical": 0.9}
            }
        }
    
    def collect_metrics(self, operation_type: str, execution_result: Dict[str, Any], 
                       validation_result: Optional[Any] = None) -> MetricsCollection:
        """
        TODO 4: Implement comprehensive performance analytics system.
        
        Requirements:
        - Collect metrics across quality, performance, cost, and user feedback dimensions
        - Implement trend analysis to identify performance patterns and degradation
        - Create alerting system with configurable thresholds and severity levels
        - Generate AI-driven optimization recommendations based on collected data
        
        This method should:
        1. Extract quality metrics from validation results (confidence scores, error rates)
        2. Calculate performance metrics (response time, processing time, retry rates)
        3. Compute cost metrics (token usage, API costs, resource utilization)
        4. Process user feedback metrics (satisfaction scores, quality ratings)
        5. Monitor system health metrics (availability, error rates, resource usage)
        6. Store metrics with timestamp and metadata for trend analysis
        7. Trigger real-time alerting if thresholds are exceeded
        8. Update performance trends and generate recommendations
        
        Args:
            operation_type: Type of operation being measured
            execution_result: Results from the operation execution
            validation_result: Optional validation results with quality metrics
            
        Returns:
            MetricsCollection: Comprehensive metrics for this operation
            
        Example Implementation:
            # Extract quality metrics
            quality_metrics = {}
            if validation_result:
                quality_metrics = {
                    "overall_confidence": validation_result.overall_confidence,
                    "content_quality": validation_result.confidence_breakdown.content_quality,
                    "validation_passed": 1.0 if validation_result.validation_passed else 0.0,
                    "error_count": len(validation_result.detected_issues),
                    "improvement_potential": validation_result.improvement_score
                }
            
            # Calculate performance metrics
            performance_metrics = {
                "response_time": execution_result.get("response_time", 0.0),
                "processing_time": execution_result.get("processing_time", 0.0),
                "retry_count": execution_result.get("retry_count", 0),
                "success_rate": 1.0 if execution_result.get("success", False) else 0.0
            }
            
            # Compute cost metrics
            cost_metrics = {
                "input_tokens": execution_result.get("input_tokens", 0),
                "output_tokens": execution_result.get("output_tokens", 0),
                "total_tokens": execution_result.get("total_tokens", 0),
                "estimated_cost": execution_result.get("estimated_cost", 0.0)
            }
            
            # Process user feedback (if available)
            user_feedback_metrics = {
                "satisfaction_score": execution_result.get("user_satisfaction", 0.0),
                "quality_rating": execution_result.get("quality_rating", 0.0),
                "usefulness_score": execution_result.get("usefulness_score", 0.0)
            }
            
            # Monitor system health
            system_health_metrics = {
                "cpu_usage": self._get_system_cpu_usage(),
                "memory_usage": self._get_system_memory_usage(),
                "error_rate": self._calculate_recent_error_rate(),
                "availability": self._calculate_system_availability()
            }
        """
        # TODO 4: Implement comprehensive metrics collection
        # Your implementation here
        pass
    
    def analyze_trends(self, metric_name: str, time_window: Optional[timedelta] = None) -> TrendAnalysis:
        """
        TODO 5: Implement continuous learning integration.
        
        Requirements:
        - Analyze performance trends over configurable time windows
        - Detect anomalies and performance degradation patterns
        - Implement feedback processing for continuous system improvement
        - Generate predictive insights for proactive optimization
        
        This method should:
        1. Filter metrics within the specified time window
        2. Calculate trend direction and strength using statistical analysis
        3. Detect anomalies and outliers in the performance data
        4. Generate confidence scores for trend predictions
        5. Provide actionable recommendations based on trend analysis
        6. Implement learning from historical feedback to improve predictions
        7. Support multiple aggregation methods (mean, median, percentiles)
        8. Handle missing data and irregular sampling intervals
        
        Args:
            metric_name: Name of the metric to analyze trends for
            time_window: Time window for trend analysis (default: 24 hours)
            
        Returns:
            TrendAnalysis: Comprehensive trend analysis with recommendations
            
        Example Implementation:
            # Filter metrics for the time window
            cutoff_time = datetime.now() - (time_window or self.trend_analysis_window)
            recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
            
            # Extract metric values
            metric_values = []
            for metrics in recent_metrics:
                if metric_name in metrics.quality_metrics:
                    metric_values.append(metrics.quality_metrics[metric_name])
                elif metric_name in metrics.performance_metrics:
                    metric_values.append(metrics.performance_metrics[metric_name])
                # ... check other metric categories
            
            # Calculate trend statistics
            if len(metric_values) >= 3:
                # Linear regression for trend direction
                x_values = list(range(len(metric_values)))
                slope = self._calculate_linear_regression_slope(x_values, metric_values)
                trend_direction = "improving" if slope > 0.1 else "degrading" if slope < -0.1 else "stable"
                trend_strength = min(abs(slope), 1.0)
                
                # Calculate confidence based on data consistency
                variance = statistics.variance(metric_values)
                confidence = max(0.0, 1.0 - (variance / max(metric_values)))
            
            # Generate recommendations based on trends
            recommendations = self._generate_trend_recommendations(metric_name, trend_direction, trend_strength)
        """
        # TODO 5: Implement trend analysis and continuous learning
        # Your implementation here
        pass
    
    def generate_optimization_recommendations(self, performance_data: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """
        TODO 5 (continued): Generate AI-driven optimization recommendations.
        
        This method should:
        1. Analyze current performance against historical baselines
        2. Identify bottlenecks and optimization opportunities
        3. Use AI to generate contextual improvement suggestions
        4. Prioritize recommendations based on impact and effort
        5. Provide implementation guidance and expected outcomes
        
        Args:
            performance_data: Current performance metrics and trends
            
        Returns:
            List[OptimizationRecommendation]: Prioritized optimization recommendations
        """
        # TODO 5: Implement AI-driven optimization recommendations
        # Your implementation here
        pass
    
    def generate_dashboard(self, include_predictions: bool = True) -> MonitoringDashboard:
        """
        TODO 6: Implement production monitoring dashboard.
        
        Requirements:
        - Create real-time performance dashboards with key performance indicators
        - Implement system health monitoring with automated diagnostics
        - Build cost analytics with usage optimization insights
        - Validate complete integration across all lesson components
        
        This method should:
        1. Calculate current system health score and status
        2. Generate real-time metrics display with historical context
        3. Compile active alerts and their severity levels
        4. Provide performance summary with trend insights
        5. Generate optimization opportunities and recommendations
        6. Include predictive analytics for proactive monitoring
        7. Format data for dashboard visualization
        8. Ensure responsive updates for real-time monitoring
        
        Args:
            include_predictions: Whether to include predictive analytics
            
        Returns:
            MonitoringDashboard: Complete dashboard data for visualization
            
        Example Implementation:
            # Calculate system health score
            health_score = self._calculate_system_health_score()
            
            # Determine system status
            if health_score >= 0.9:
                system_status = "healthy"
            elif health_score >= 0.7:
                system_status = "warning"
            elif health_score >= 0.5:
                system_status = "critical"
            else:
                system_status = "offline"
            
            # Generate live metrics
            live_metrics = self._get_current_metrics()
            
            # Compile performance summary
            performance_summary = {
                "uptime": self._calculate_uptime(),
                "total_requests": len(self.metrics_history),
                "average_response_time": self._calculate_average_response_time(),
                "success_rate": self._calculate_success_rate(),
                "cost_efficiency": self._calculate_cost_efficiency()
            }
            
            # Get trend insights
            trend_insights = [
                self.analyze_trends("overall_confidence"),
                self.analyze_trends("response_time"),
                self.analyze_trends("cost_per_request")
            ]
            
            # Generate optimization opportunities
            optimization_opportunities = self.generate_optimization_recommendations(live_metrics)
        """
        # TODO 6: Implement production monitoring dashboard
        # Your implementation here
        pass
    
    def process_user_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO 5 (continued): Process user feedback for continuous learning.
        
        This method should:
        1. Structure and validate incoming feedback data
        2. Correlate feedback with performance metrics
        3. Update learning models based on feedback patterns
        4. Generate insights for system improvement
        
        Args:
            feedback_data: User feedback including ratings, comments, and context
            
        Returns:
            Dict[str, Any]: Processed feedback with learning insights
        """
        # TODO 5: Implement feedback processing for continuous learning
        # Your implementation here
        pass
    
    # Helper methods for implementation
    def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score (0.0 to 1.0)."""
        # Implementation hint: Aggregate recent metrics across all dimensions
        pass
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current real-time metrics."""
        # Implementation hint: Return latest values for key metrics
        pass
    
    def _calculate_average_response_time(self, window_hours: int = 24) -> float:
        """Calculate average response time over specified window."""
        # Implementation hint: Filter recent metrics and compute average
        pass
    
    def _calculate_success_rate(self, window_hours: int = 24) -> float:
        """Calculate success rate over specified window."""
        # Implementation hint: Count successful vs failed operations
        pass
    
    def _detect_anomalies(self, metric_values: List[float]) -> List[int]:
        """Detect anomalous values using statistical methods."""
        # Implementation hint: Use z-score or IQR method for outlier detection
        pass
    
    def _trigger_alert_if_needed(self, metric_name: str, value: float, metric_category: str):
        """Check if metric value exceeds thresholds and trigger alerts."""
        # Implementation hint: Compare against configured thresholds
        pass
    
    def _generate_trend_recommendations(self, metric_name: str, trend_direction: str, 
                                      trend_strength: float) -> List[str]:
        """Generate recommendations based on trend analysis."""
        # Implementation hint: Provide specific suggestions based on metric and trend
        pass
    
    def export_performance_report(self, output_path: str = "performance_report.json"):
        """Export comprehensive performance analytics report."""
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "system_overview": {
                "total_operations": len(self.metrics_history),
                "time_period": {
                    "start": min(m.timestamp for m in self.metrics_history).isoformat() if self.metrics_history else None,
                    "end": max(m.timestamp for m in self.metrics_history).isoformat() if self.metrics_history else None
                },
                "health_score": self._calculate_system_health_score()
            },
            "performance_trends": {name: trend.__dict__ for name, trend in self.trends_cache.items()},
            "active_alerts": [alert.__dict__ for alert in self.active_alerts],
            "recommendations": [rec.__dict__ for rec in self.generate_optimization_recommendations({})]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"Performance report exported to {output_path}")


class ContinuousLearningEngine:
    """
    Continuous learning system that adapts based on performance feedback.
    
    This class implements feedback-driven learning mechanisms that improve
    system performance over time through systematic analysis and adjustment.
    """
    
    def __init__(self, project_id: str):
        """Initialize the continuous learning engine."""
        self.project_id = project_id
        self.learning_history: List[Dict[str, Any]] = []
        self.adaptation_strategies: Dict[str, Any] = {}
        
    def learn_from_feedback(self, operation_result: Dict[str, Any], 
                          user_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO 5 (continued): Implement learning from operational feedback.
        
        This method should:
        1. Correlate operation results with user feedback
        2. Identify patterns in successful vs unsuccessful operations  
        3. Update internal models and strategies based on learning
        4. Generate adaptation recommendations for future operations
        
        Args:
            operation_result: Results from a completed operation
            user_feedback: User feedback about the operation
            
        Returns:
            Dict[str, Any]: Learning insights and adaptation strategies
        """
        # TODO 5: Implement learning from feedback
        # Your implementation here
        pass
    
    def adapt_parameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO 5 (continued): Adapt system parameters based on learning.
        
        This method should:
        1. Apply learned optimizations to operation parameters
        2. Adjust strategies based on historical performance
        3. Provide parameter recommendations for current context
        
        Args:
            context: Current operation context
            
        Returns:
            Dict[str, Any]: Adapted parameters and strategies
        """
        # TODO 5: Implement parameter adaptation
        # Your implementation here
        pass


# Integration class that brings together all lesson components
class ProductionAISystem:
    """
    Complete production AI system integrating all lesson components.
    
    This class demonstrates the integration of:
    - Lesson 1: Role-based prompting and personas
    - Lesson 2: Chain-of-Thought and ReACT reasoning
    - Lesson 3: Prompt optimization techniques
    - Lesson 4: Sequential and conditional prompt chaining
    - Lesson 5: Feedback loops and iterative improvement
    """
    
    def __init__(self, project_id: str):
        """Initialize the complete production AI system."""
        self.project_id = project_id
        
        # Initialize lesson components (import paths would be adjusted in real implementation)
        # self.persona_manager = PersonaManager(project_id)
        # self.cot_agent = CoTAgent(project_id) 
        # self.react_agent = ReACTAgent(project_id)
        # self.optimizer = VertexPromptOptimizer(project_id)
        # self.bi_chain = BusinessIntelligenceChain(project_id)
        
        # Initialize Lesson 5 components
        # self.validator = SelfValidator(project_id)
        self.analytics = PerformanceAnalytics(project_id)
        self.learning_engine = ContinuousLearningEngine(project_id)
        
        # System state
        self.system_metrics: Dict[str, Any] = {}
        self.is_production_ready = False
        
    def validate_system_integration(self) -> Dict[str, bool]:
        """
        TODO 6 (continued): Validate integration across all lesson components.
        
        This method should:
        1. Test integration between all lesson components
        2. Validate data flow across the complete pipeline
        3. Ensure proper error handling and recovery
        4. Verify performance meets production requirements
        
        Returns:
            Dict[str, bool]: Validation results for each component integration
        """
        # TODO 6: Implement complete system integration validation
        validation_results = {
            "personas_integration": False,  # Test Lesson 1 integration
            "reasoning_integration": False,  # Test Lesson 2 integration  
            "optimization_integration": False,  # Test Lesson 3 integration
            "chaining_integration": False,  # Test Lesson 4 integration
            "feedback_loops_integration": False,  # Test Lesson 5 integration
            "end_to_end_workflow": False,  # Test complete workflow
            "production_readiness": False   # Overall production readiness
        }
        
        # Your implementation here
        return validation_results
    
    def run_production_workflow(self, business_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute complete production workflow with all feedback loops.
        
        This demonstrates the integration of all lessons in a production environment.
        """
        workflow_start = time.time()
        
        try:
            # TODO 6: Implement complete production workflow
            # This would integrate all lesson components with feedback loops
            
            result = {
                "success": True,
                "execution_time": time.time() - workflow_start,
                "components_used": ["personas", "reasoning", "optimization", "chaining", "feedback"],
                "quality_score": 0.0,  # From validation
                "recommendations": []   # From learning engine
            }
            
            # Record metrics
            metrics = self.analytics.collect_metrics("production_workflow", result)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - workflow_start,
                "recovery_suggestions": ["Check system health", "Validate integrations", "Review error logs"]
            }


if __name__ == "__main__":
    """
    Example usage and testing of the improvement engine.
    
    This demonstrates how to use the performance analytics and continuous learning
    systems in a production environment.
    """
    import os
    
    # Get project ID from environment
    project_id = os.getenv("PROJECT_ID", "your-gcp-project-id")
    
    if project_id == "your-gcp-project-id":
        print("⚠️  Please set PROJECT_ID environment variable")
        print("Example: export PROJECT_ID='your-actual-project-id'")
        exit(1)
    
    print("🚀 Testing Improvement Engine - TODOs 4-30")
    print("=" * 50)
    
    # Initialize systems
    analytics = PerformanceAnalytics(project_id)
    learning_engine = ContinuousLearningEngine(project_id)
    production_system = ProductionAISystem(project_id)
    
    # Test TODO 4: Performance Analytics
    print("\n📊 Testing TODO 4: Performance Analytics")
    print("-" * 40)
    
    # Mock operation result for testing
    mock_operation = {
        "operation_type": "business_report_generation",
        "success": True,
        "response_time": 2.5,
        "processing_time": 2.1,
        "input_tokens": 1500,
        "output_tokens": 3000,
        "total_tokens": 4500,
        "estimated_cost": 0.045,
        "retry_count": 0
    }
    
    print(f"Collecting metrics for: {mock_operation['operation_type']}")
    
    # TODO 4: This should collect comprehensive metrics
    # metrics = analytics.collect_metrics("test_operation", mock_operation)
    # print(f"✅ Metrics collected: {len(metrics.quality_metrics)} quality metrics")
    print("❌ TODO 4: Implement metrics collection")
    
    # Test TODO 5: Continuous Learning
    print("\n🧠 Testing TODO 5: Continuous Learning")
    print("-" * 40)
    
    # TODO 5: This should analyze trends and generate recommendations
    # trend_analysis = analytics.analyze_trends("overall_confidence")
    # print(f"✅ Trend analysis: {trend_analysis.trend_direction} trend detected")
    print("❌ TODO 5: Implement trend analysis and continuous learning")
    
    # Test user feedback processing
    mock_feedback = {
        "satisfaction_score": 4.2,
        "quality_rating": 4.0,
        "usefulness_score": 4.5,
        "comments": "High quality report, could be more concise"
    }
    
    # TODO 5: This should process feedback for learning
    # learning_insights = learning_engine.learn_from_feedback(mock_operation, mock_feedback)
    # print(f"✅ Learning insights generated: {len(learning_insights)} strategies")
    print("❌ TODO 5: Implement feedback processing")
    
    # Test TODO 6: Production Monitoring
    print("\n📈 Testing TODO 6: Production Monitoring Dashboard")
    print("-" * 40)
    
    # TODO 6: This should generate a complete monitoring dashboard
    # dashboard = analytics.generate_dashboard(include_predictions=True)
    # print(f"✅ Dashboard generated - System status: {dashboard.system_status}")
    # print(f"✅ Health score: {dashboard.health_score:.2f}")
    # print(f"✅ Active alerts: {len(dashboard.active_alerts)}")
    print("❌ TODO 6: Implement production monitoring dashboard")
    
    # Test complete system integration
    print("\n🔧 Testing Complete System Integration")
    print("-" * 40)
    
    # TODO 6: This should validate all lesson integrations
    # validation_results = production_system.validate_system_integration()
    # integration_score = sum(validation_results.values()) / len(validation_results)
    # print(f"✅ Integration validation: {integration_score:.1%} components ready")
    print("❌ TODO 6: Implement system integration validation")
    
    print("\n" + "=" * 50)
    print("🎯 Ready to implement TODOs 4-30!")
    print("\nNext steps:")
    print("1. Implement comprehensive performance analytics (TODO 4)")
    print("2. Build continuous learning and trend analysis (TODO 5)")  
    print("3. Create production monitoring dashboard (TODO 6)")
    print("4. Validate complete lesson integration")
    print("\n🚀 Build production-ready AI systems with feedback loops!")