# Lesson 4: Sequential Prompt Chaining

## 🎯 Learning Objective
Master sequential prompt chaining to build sophisticated reasoning systems that break down complex problems into manageable steps with context preservation and quality validation.

## 📋 What You'll Learn
- Design and implement multi-step prompt chaining workflows
- Manage context flow and state preservation across chain steps
- Build quality gates and validation checkpoints
- Create error recovery and retry mechanisms for chain failures

## 🚀 Your Task

Complete **TODOs 1-3** in `sequential_chain.py` to build a Business Intelligence Chain that generates comprehensive market analysis reports through four sequential reasoning steps.

### TODO 1: Chain Step Execution
Implement robust chain step processing with:
- **Context Management**: Seamless data flow between steps
- **State Preservation**: Maintain reasoning history and intermediate results
- **Quality Validation**: Validate each step before proceeding
- **Error Handling**: Graceful failure recovery with meaningful feedback

**Key Implementation Points:**
```python
def execute_chain(self, initial_prompt: str, steps: List[ChainStep]) -> ChainResult:
    # Initialize chain context
    context = ChainContext(initial_input=initial_prompt)

    for step in steps:
        # Execute step with retry logic
        step_result = self._execute_step_with_retry(step, context)

        # Validate step success
        if not step_result.success:
            return failed ChainResult

        # Check quality threshold
        if step_result.quality_score < step.quality_threshold:
            # Attempt quality recovery

        # Update context for next step
        context = self._update_context_flow(context, step_result)

    return successful ChainResult
```

### TODO 2: Context Flow Management
Create sophisticated context passing system with:
- **Context Accumulation**: Build comprehensive reasoning history
- **Selective Context**: Include only relevant information for next step
- **Token Optimization**: Manage context size for efficiency
- **State Tracking**: Maintain chain progress and decision points

**Key Implementation Points:**
```python
def _update_context_flow(self, context: ChainContext, step_result: StepResult) -> ChainContext:
    # Add step to history
    context.step_history.append({
        "step_name": step_result.step_name,
        "quality_score": step_result.quality_score,
        "key_insights": step_result.key_insights
    })

    # Update accumulated insights
    context.accumulated_insights.extend(step_result.key_insights)

    # Track quality scores
    context.quality_scores.append(step_result.quality_score)

    # Update token usage
    context.token_usage["total_input"] += step_result.token_usage["input_tokens"]

    return context
```

### TODO 3: Chain Quality Validation
Build comprehensive validation framework with:
- **Multi-Dimensional Assessment**: Content length, structure, relevance, step-type quality
- **Step Validation**: Quality checks for individual chain steps
- **Chain Coherence**: Ensure logical flow across entire sequence
- **Adaptive Thresholds**: Validation level adjustability (BASIC/STANDARD/STRICT)

**Key Implementation Points:**
```python
def _validate_step_result(self, content: str, step: ChainStep, context: ChainContext) -> float:
    quality_score = 0.0

    # Content length validation (25%)
    quality_score += 0.25 if len(content.split()) >= 100 else 0.15

    # Structure and organization (25%)
    quality_score += self._assess_content_structure(content, step.step_type) * 0.25

    # Step-type specific validation (25%)
    quality_score += self._assess_step_type_quality(content, step.step_type) * 0.25

    # Context relevance (25%)
    quality_score += self._assess_context_relevance(content, context) * 0.25

    return min(quality_score, 1.0)
```

## 📝 Implementation Guidelines

### Sequential Chain Architecture

The chain executes four sequential steps, each building on previous context:

1. **Market Overview** → Business Analyst persona analyzes market landscape
2. **Competitive Analysis** → Market Researcher assesses competitive dynamics
3. **Risk Assessment** → Strategic Consultant evaluates potential risks
4. **Strategic Recommendations** → Strategic Consultant provides actionable guidance

### Chain Context Structure

```python
@dataclass
class ChainContext:
    initial_input: str
    step_history: Optional[List[Dict]] = None
    accumulated_insights: Optional[List[str]] = None
    quality_scores: Optional[List[float]] = None
    token_usage: Optional[Dict[str, int]] = None
```

### Chain Step Configuration

```python
ChainStep(
    name="Market Overview",
    step_type=ChainStepType.ANALYSIS,
    prompt_template="You are a senior business analyst...",
    validation_level=ValidationLevel.BASIC,
    quality_threshold=0.55,
    max_retries=3
)
```

## 🧪 Testing Your Work

### Environment Setup
```bash
export PROJECT_ID="your-gcp-project-id"
export CHAIN_DEBUG=false  # Set to true for detailed logging

# Verify integration with previous lessons
python -c "
from sequential_chain import SequentialChain
print('✅ Sequential chain ready')
"
```

### Run Your Implementation
```bash
python sequential_chain.py
```

**Expected Output:**
```
🔗 Testing Sequential Chain Implementation
============================================================
✅ Sequential chain executed successfully!
📊 Overall Quality: 0.80+
⏱️  Total Time: ~50s
🎯 Steps Completed: 4
💰 Token Usage: ~3500-4000

🧪 TODO Validation:
✅ TODO 1: Chain step execution - IMPLEMENTED
✅ TODO 2: Context flow management - IMPLEMENTED
✅ TODO 3: Chain quality validation - IMPLEMENTED
```

### Debug Chain Execution
```bash
# Enable detailed logging
export CHAIN_DEBUG=true
python sequential_chain.py
```

This will show step-by-step execution:
```
🔗 Starting chain execution with 4 steps
⚡ Executing step 1/4: Market Overview
✅ Step 'Market Overview' completed - Quality: 0.850
⚡ Executing step 2/4: Competitive Analysis
✅ Step 'Competitive Analysis' completed - Quality: 0.825
...
```

## 📊 Success Criteria

Your implementation must achieve:
- ✅ **Chain execution success rate ≥ 90%** across test scenarios
- ✅ **Overall quality score ≥ 0.75** for complete chain
- ✅ **Context preservation** maintains reasoning coherence across steps
- ✅ **Quality validation** prevents low-quality propagation
- ✅ **Error recovery** handles failures gracefully with retries

### Quality Score Breakdown
- **Content Length** (25%): ≥100 words for full score
- **Structure** (25%): Logical flow indicators, paragraphs, lists
- **Step-Type Quality** (25%): Relevant terminology and frameworks
- **Context Relevance** (25%): Integration with previous insights

## 💡 Tips for Success

### 1. Chain Design Best Practices

**Good Chain Design:**
```python
# Execute step with current context
step_result = self._execute_step_with_retry(step, context)

# Always validate before continuing
if not step_result.success or step_result.quality_score < threshold:
    # Handle failure or attempt recovery

# Update context for next step
context = self._update_context_flow(context, step_result)
```

**Avoid:**
- ❌ Executing steps without context updates
- ❌ Skipping quality validation
- ❌ Ignoring failed steps
- ❌ Not tracking token usage

### 2. Context Management Strategies

**Effective Context Flow:**
- Include initial business scenario in every step
- Extract and pass forward key insights (limit to last 3)
- Track quality scores to inform subsequent steps
- Monitor token usage to prevent context overflow

**Context Summary Example:**
```
Business Scenario: TechFlow Solutions strategic expansion...
Previous Key Insights:
- Analysis: Market size estimated at $2.5B with 15% CAGR
- Analysis: Enterprise segment shows strong adoption
- Analysis: Competition increasing in SMB space
```

### 3. Quality Validation Framework

**Multi-Dimensional Assessment:**
```python
# Each dimension contributes equally
content_length_score = 0.25  # Based on word count
structure_score = self._assess_content_structure(content) * 0.25
type_quality_score = self._assess_step_type_quality(content) * 0.25
relevance_score = self._assess_context_relevance(content) * 0.25

total_quality = sum([content_length, structure, type_quality, relevance])
```

**Validation Levels:**
- `BASIC`: Multiplies score by 1.1 (more lenient)
- `STANDARD`: No adjustment (balanced)
- `STRICT`: Multiplies score by 0.9 (higher standards)

### 4. Error Recovery Patterns

**Retry Strategy:**
1. **First attempt**: Standard prompt with temperature=0.1
2. **Retry attempts**: Enhanced prompt with temperature=0.3
3. **Quality recovery**: Add detailed instructions + lower threshold slightly
4. **Final fallback**: Return error with comprehensive diagnostics

## 🔍 Example Chain Execution Flow

### Complete Business Analysis Chain

```
Input: "TechFlow Solutions - Should we expand into small business markets?"

Step 1: Market Overview (Business Analyst)
├─ Context: Initial business scenario
├─ Output: Market sizing, growth trends, opportunity assessment
├─ Quality: 0.85 ✅
└─ Key Insights: ["Market size $2.5B", "15% CAGR", "Strong enterprise demand"]

Step 2: Competitive Analysis (Market Researcher)
├─ Context: Business scenario + Market overview insights
├─ Output: Competitor analysis, positioning, differentiation strategies
├─ Quality: 0.82 ✅
└─ Key Insights: ["3 major competitors", "Price pressure in SMB", "Enterprise loyalty high"]

Step 3: Risk Assessment (Strategic Consultant)
├─ Context: Market + Competitive insights + Risk focus
├─ Output: Risk identification, impact analysis, mitigation strategies
├─ Quality: 0.84 ✅
└─ Key Insights: ["Platform complexity risk", "Resource constraint", "Brand dilution"]

Step 4: Strategic Recommendations (Strategic Consultant)
├─ Context: Complete analysis + Recommendation focus
├─ Output: Strategic options, implementation roadmap, success metrics
├─ Quality: 0.88 ✅
└─ Key Insights: ["Phased approach recommended", "6-month pilot", "Focus on vertical SMBs"]

Final Report: Complete 4-section BI analysis
├─ Overall Quality: 0.85 (Target: ≥0.75) ✅
├─ Token Usage: 3,794 tokens (Efficient)
├─ Generation Time: 48.9 seconds
└─ Success: All quality gates passed ✅
```

## 🎉 What You've Built

After completing this lesson successfully:
1. **Sequential reasoning system** that breaks complex problems into steps
2. **Context management** that preserves reasoning flow across chain
3. **Quality validation** that ensures consistent output standards
4. **Production-ready chain** with error recovery and retry logic

This pattern applies to any multi-step reasoning task:
- Research analysis workflows
- Strategic planning processes
- Technical documentation generation
- Complex decision-making systems

## 🆘 Common Issues & Solutions

### "Chain breaks at step X"
**Cause**: Context not properly updated or quality threshold too high
**Solution**:
- Verify `_update_context_flow` returns updated context
- Check quality threshold is realistic (0.55-0.6 recommended)
- Enable `CHAIN_DEBUG=true` to see detailed execution

### "Quality degradation across steps"
**Cause**: Context not preserving key insights from previous steps
**Solution**:
- Ensure `accumulated_insights.extend()` is called
- Verify context summary includes previous insights
- Check `_build_context_summary` method implementation

### "High token usage"
**Cause**: Passing full content history instead of summaries
**Solution**:
- Limit insights to last 3: `accumulated_insights[-3:]`
- Use content summaries: `content[:200] + "..."`
- Monitor `token_usage` tracking

### "Inconsistent quality scores"
**Cause**: LLM output variation affecting validation
**Solution**:
- Use `BASIC` validation level for intermediate steps
- Set realistic thresholds (0.55 recommended)
- Implement retry logic with enhanced prompts

## 📚 Code Structure Reference

### Main Classes
- **`SequentialChain`**: Main orchestrator for chain execution
- **`ChainContext`**: State management across chain steps
- **`ChainStep`**: Configuration for individual reasoning steps
- **`StepResult`**: Output from executing a single step
- **`ChainResult`**: Final result from complete chain execution

### Key Methods to Implement
1. **`execute_chain()`**: Main execution loop (TODO 1)
2. **`_update_context_flow()`**: Context management (TODO 2)
3. **`_validate_step_result()`**: Quality assessment (TODO 3)

### Helper Methods (Provided)
- `_execute_step_with_retry()`: Retry logic for failed steps
- `_execute_single_step()`: Single step execution with Gemini
- `_create_step_prompt()`: Build context-aware prompts
- `_build_context_summary()`: Intelligent context selection
- `_assess_content_structure()`: Structure quality scoring
- `_assess_step_type_quality()`: Type-specific validation
- `_assess_context_relevance()`: Context integration scoring

---

## 🚀 **Master Sequential Reasoning**

**Before**: Single-prompt solutions with limited reasoning depth
**After**: Sophisticated multi-step reasoning with intelligent context flow

**Key Benefits**:
- Break complex problems into manageable steps
- Build production-ready agentic workflows
- Implement robust error recovery systems
- Create validated, high-quality reasoning chains

**Real-World Impact**: Transform hours of manual analysis into automated, consistent, validated reports! 🎯
