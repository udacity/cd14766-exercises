# Lesson 1: Role-Based Prompting with Vertex AI Integration

## Learning Objectives
Master the art of designing effective role-based personas for AI agents and test them with Vertex AI Gemini to generate consistent, high-quality business analysis across different expertise domains.

## What You'll Learn
- Design professional business personas with specific expertise
- Create effective communication styles for different roles  
- Define analytical frameworks for each business function
- Test personas with Vertex AI Gemini API
- Compare persona effectiveness using real AI responses
- Measure quality and performance metrics

## Lesson Structure
This lesson now consists of two main exercises that build practical AI skills:

1. **Classic Persona Creation** (`personas.py`)
   - Text-based persona design and validation
   - Foundation concepts and frameworks

2. **AI-Integrated Persona Testing** (`personas_with_ai.py`) ⭐ **ENHANCED**
   - Real Vertex AI Gemini integration
   - Live persona testing and comparison
   - Performance metrics and quality analysis

## Your Tasks

### Exercise 1: Foundation Personas (TODOs 1-3)
Complete the original persona creation in `personas.py`:

#### TODO 1: Business Analyst Persona
Design a senior business analyst focused on:
- **Quantitative market analysis** with specific metrics
- **Data-driven insights** with clear reasoning chains  
- **Professional communication** using business terminology
- **Analytical frameworks** like TAM/SAM/SOM, market sizing, trend analysis

#### TODO 2: Market Researcher Persona  
Create a market research specialist expert in:
- **Competitive intelligence** and industry analysis
- **Strategic positioning** assessment and market dynamics
- **Comprehensive analysis** using frameworks like Porter's Five Forces
- **Competitive landscape** mapping and barrier assessment

#### TODO 3: Strategic Consultant Persona
Build a strategic consultant specializing in:
- **Risk assessment** and strategic planning
- **Implementable recommendations** with clear business rationale
- **Action-oriented approach** with ROI considerations
- **Strategic frameworks** and implementation roadmaps

### Exercise 2: AI Integration (TODOs 4-5) ⭐ **NEW**
Complete the enhanced AI testing in `personas_with_ai.py`:

#### TODO 4: Test Persona with Vertex AI
Implement real AI testing functionality:
- **Vertex AI integration** using Gemini 2.5 Flash
- **Prompt combination** of persona + business scenario
- **Quality analysis** of AI responses
- **Performance metrics** tracking (time, tokens, quality scores)

Requirements:
```python
def test_persona_with_scenario(self, persona_key: str, scenario: Dict) -> Dict:
    # 1. Validate persona exists
    # 2. Combine persona with scenario into comprehensive prompt
    # 3. Call Vertex AI Gemini API
    # 4. Analyze response quality
    # 5. Return structured results with metrics
```

#### TODO 5: Compare Persona Effectiveness
Build comprehensive comparison system:
- **Multi-persona testing** on same business scenario
- **Side-by-side comparison** of AI responses
- **Quality scoring** and performance ranking
- **Best persona identification** based on metrics

Requirements:
```python
def compare_personas(self, scenario: Dict) -> Dict:
    # 1. Test all three personas on same scenario
    # 2. Display progress and immediate feedback
    # 3. Identify best performing persona
    # 4. Show detailed comparison results
```

## Implementation Guidelines

### Persona Structure (TODOs 1-3)
Each persona should include:
```
Role: [Professional background and experience level]
Expertise: [Specific areas of specialization]  
Communication Style: [How they communicate and present information]
Analytical Approach: [Frameworks and methodologies they use]
Task Context: [How they should approach their analysis]
```

### AI Integration Requirements (TODOs 4-5)
```python
# Setup Vertex AI client
vertexai.init(project=project_id, location=location)
self.client = genai.Client(vertexai=vertexai)

# Create comprehensive prompt
full_prompt = f"""
{persona}

Business Scenario:
Company: {scenario['company_name']}
Industry: {scenario['industry']}
Strategic Question: {scenario['strategic_question']}

Based on your expertise, provide your analysis...
"""

# Call Gemini API
response = self.client.models.generate_content(
    model="gemini-2.5-flash",
    contents=full_prompt,
    config=GenerateContentConfig(temperature=0.7, max_output_tokens=800)
)
```

## 🧪 Testing Your Work

### Environment Setup
```bash
# Set your GCP project ID
export PROJECT_ID=your-gcp-project-id

# Ensure Vertex AI is enabled and credentials are configured
gcloud auth application-default login
```

### Quick Test (Classic Personas)
```bash
python personas.py
python test_personas.py --verbose
```

### AI Integration Test ⭐ **NEW**
```bash
# Test enhanced AI integration
python personas_with_ai.py

# Run comprehensive AI tests
python test_ai_personas.py --verbose
```

### Full Test Suite
```bash
# Test both classic and AI integration
python test_personas.py --component all
python test_ai_personas.py --component comparison
```

## Success Criteria

### Classic Personas (TODOs 1-3)
- ✅ **Score ≥ 0.8** on validation tests
- ✅ **All quality checks** passing
- ✅ **Professional depth** with business frameworks
- ✅ **Clear communication style** defined

### AI Integration (TODOs 4-5) ⭐ **NEW**
- ✅ **Successful Vertex AI connection** and API calls
- ✅ **Quality scores ≥ 0.7** for AI responses
- ✅ **Persona differentiation** visible in AI outputs
- ✅ **Performance metrics** correctly calculated
- ✅ **Comparison system** working across all personas

## 💡 Tips for Success

### 1. Persona Design (Classic)
- **Be Specific**: Include years of experience and exact frameworks
- **Use Terminology**: Professional business language throughout
- **Define Style**: Clear communication guidelines
- **Add Context**: Specific task-oriented instructions

### 2. AI Integration (New) ⭐
- **Environment Setup**: Ensure PROJECT_ID is set correctly
- **Error Handling**: Check for API connection issues
- **Prompt Design**: Combine persona + scenario effectively
- **Quality Analysis**: Understand the scoring metrics

### 3. Testing Strategy
```python
# Good persona structure
"Role: You are a Senior Business Analyst with 15+ years of experience..."
"Expertise: Your specialization includes TAM/SAM/SOM analysis..."
"Communication Style: Professional, data-driven, objective..."

# Good AI prompt combination
full_prompt = f"{persona}\n\nBusiness Scenario:\n{scenario_details}\n\nAnalyze this scenario using your expertise..."
```

## 🔍 Example AI Output

### Business Analyst Response
```
"Based on my analysis using TAM/SAM/SOM methodology:

Step 1: Total Addressable Market Analysis
The residential solar market represents $45B globally...

Step 2: Market Sizing Calculations  
With 15% current market share in CA/TX, this equals $2.3B...

Therefore, I recommend focusing on residential expansion 
because the data shows 40% growth in this segment..."
```

### Quality Metrics
```
Quality Score: 0.85/1.0
Keyword Coverage: 75% (market, analysis, data, growth, etc.)
Framework Usage: 3 mentions (TAM/SAM/SOM, market share analysis)
Persona Alignment: High
Response Length: 145 words
Generation Time: 1.2s
```

## 🎉 What's Next?

After completing this enhanced lesson:
1. **Verify AI integration** - All personas work with Gemini
2. **Test multiple scenarios** - Try different business cases
3. **Analyze persona strengths** - Understand which works best when
4. **Move to Lesson 2** - Chain-of-Thought and ReACT Prompting

## 🆘 Getting Help

### Common Issues:
- **"PROJECT_ID not set"** → Set environment variable correctly
- **"Vertex AI connection failed"** → Check authentication and API access
- **"Persona not implemented"** → Complete TODOs 1-8 first
- **"Quality scores low"** → Review persona design for business depth
- **"AI responses generic"** → Strengthen persona specificity and frameworks

### Debug AI Integration:
```python
# Test connection
print(f"Testing connection to project: {PROJECT_ID}")

# Check persona content
print(f"Persona length: {len(BUSINESS_ANALYST_PERSONA)} characters")

# Monitor API calls
print(f"Calling Gemini with prompt length: {len(full_prompt)}")
```

---

## 🚀 **New Value Proposition**

**Before**: Text-only persona creation with basic validation  
**After**: Real AI testing with Vertex AI, quality metrics, and performance comparison

**Key Benefits**:
- Students experience actual AI integration from Lesson 1
- Immediate feedback on persona effectiveness
- Smooth progression to advanced techniques in Lesson 2
- Real-world skills with Vertex AI Gemini API

**Pedagogical Excellence**: Hands-on learning with immediate, measurable results! 🎯