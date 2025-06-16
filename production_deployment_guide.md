# 🚀 GAIA Agent Production Deployment Guide

## System Architecture: Qwen Models + LangGraph Workflow

### **🎯 Updated System Requirements**

**GAIA Agent now uses ONLY:**
- ✅ **Qwen 2.5 Models**: 7B/32B/72B via HuggingFace Inference API  
- ✅ **LangGraph Workflow**: Multi-agent orchestration with synthesis
- ✅ **Specialized Agents**: Router, web research, file processing, reasoning
- ✅ **Professional Tools**: Wikipedia, web search, calculator, file processor
- ❌ **No Fallbacks**: Requires proper authentication - no simplified responses

### **🚨 Authentication Requirements - CRITICAL**

**The system now REQUIRES proper authentication:**

```python
# REQUIRED: HuggingFace token with inference permissions
HF_TOKEN=hf_your_token_here

# The system will FAIL without proper authentication
# No SimpleClient fallback available
```

### **🎯 Expected Results**

With proper authentication and Qwen model access:

- **✅ GAIA Benchmark Score**: 30%+ (full LangGraph workflow with Qwen models)
- **✅ Multi-Agent Processing**: Router → Specialized Agents → Tools → Synthesis
- **✅ Intelligent Model Selection**: 7B (fast) → 32B (balanced) → 72B (complex)
- **✅ Professional Tools**: Wikipedia API, DuckDuckGo search, calculator, file processor
- **✅ Detailed Analysis**: Processing details, confidence scores, cost tracking

**Without proper authentication:**
- **❌ System Initialization Fails**: No fallback options available
- **❌ Clear Error Messages**: Guides users to proper authentication setup

## 🔧 Technical Implementation

### OAuth Authentication (Production)

```python
class GAIAAgentApp:
    def __init__(self, hf_token: Optional[str] = None):
        if not hf_token:
            raise ValueError("HuggingFace token with inference permissions is required")
        
        # Initialize QwenClient with token
        self.llm_client = QwenClient(hf_token=hf_token)
        
        # Initialize LangGraph workflow with tools
        self.workflow = SimpleGAIAWorkflow(self.llm_client)

# OAuth token extraction in production
def run_and_submit_all(profile: gr.OAuthProfile | None):
    oauth_token = getattr(profile, 'oauth_token', None)
    agent = GAIAAgentApp.create_with_oauth_token(oauth_token)
```

### Qwen Model Configuration

```python
# QwenClient now uses ONLY Qwen models
self.models = {
    ModelTier.ROUTER: ModelConfig(
        name="Qwen/Qwen2.5-7B-Instruct",      # Fast classification
        cost_per_token=0.0003
    ),
    ModelTier.MAIN: ModelConfig(
        name="Qwen/Qwen2.5-32B-Instruct",     # Balanced performance  
        cost_per_token=0.0008
    ),
    ModelTier.COMPLEX: ModelConfig(
        name="Qwen/Qwen2.5-72B-Instruct",     # Best performance
        cost_per_token=0.0015
    )
}
```

### Error Handling

```python
# Clear error messages guide users to proper authentication
if not oauth_token:
    return "Authentication Required: Valid token with inference permissions needed for Qwen model access."

try:
    agent = GAIAAgentApp.create_with_oauth_token(oauth_token)
except ValueError as ve:
    return f"Authentication Error: {ve}"
except RuntimeError as re:
    return f"System Error: {re}"
```

## 🎯 Deployment Steps

### 1. Pre-Deployment Checklist

- [ ] **Code Ready**: All Qwen-only changes committed
- [ ] **Dependencies**: `requirements.txt` updated with all packages  
- [ ] **Testing**: QwenClient initialization test passes locally
- [ ] **Environment**: No hardcoded tokens in code
- [ ] **Authentication**: HF_TOKEN available with inference permissions

### 2. HuggingFace Space Configuration

Create a new HuggingFace Space with these settings:

```yaml
# Space Configuration
title: "GAIA Agent System"
emoji: "🤖"
colorFrom: "blue"
colorTo: "green"
sdk: gradio
sdk_version: "4.44.0"
app_file: "src/app.py"
pinned: false
license: "mit"
suggested_hardware: "cpu-basic"
suggested_storage: "small"
```

### 3. Required Files Structure

```
/
├── src/
│   ├── app.py                 # Main application (Qwen + LangGraph)
│   ├── models/
│   │   └── qwen_client.py     # Qwen-only client  
│   ├── agents/               # All agent files
│   ├── tools/                # All tool files
│   ├── workflow/             # LangGraph workflow
│   └── requirements.txt      # All dependencies
├── README.md                 # Space documentation
└── .gitignore               # Exclude sensitive files
```

### 4. Environment Variables (Space Secrets)

**🎯 CRITICAL: Set HF_TOKEN for Qwen Model Access**

To get **real GAIA Agent performance** with Qwen models and LangGraph workflow:

```bash
# REQUIRED for Qwen model access and LangGraph functionality
HF_TOKEN=hf_your_token_here                # REQUIRED: Your HuggingFace token
```

**How to set HF_TOKEN:**
1. Go to your Space settings in HuggingFace
2. Navigate to "Repository secrets" 
3. Add new secret:
   - **Name**: `HF_TOKEN`
   - **Value**: Your HuggingFace token (from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

⚠️ **IMPORTANT**: Do NOT set `HF_TOKEN` as a regular environment variable - use Space secrets for security.

**Token Requirements:**
- Token must have **`read`** and **`inference`** scopes
- Generate token at: https://huggingface.co/settings/tokens
- Select "Fine-grained" token type
- Enable both scopes for Qwen model functionality

**Optional environment variables:**

```bash
# Optional: LangSmith tracing (if you want observability)
LANGCHAIN_TRACING_V2=true           # Optional: LangSmith tracing
LANGCHAIN_API_KEY=your_key_here     # Optional: LangSmith API key
LANGCHAIN_PROJECT=gaia-agent        # Optional: LangSmith project
```

### 5. Authentication Flow in Production

```python
# Production OAuth Flow:
1. User clicks "Login with HuggingFace" button
2. OAuth flow provides profile with token
3. System validates OAuth token for Qwen model access
4. If sufficient scopes: Initialize QwenClient with LangGraph workflow
5. If insufficient scopes: Show clear error message with guidance
6. System either works fully or fails clearly - no degraded modes
```

#### OAuth Requirements ⚠️

**CRITICAL**: Gradio OAuth tokens often have **limited scopes** by default:
- ✅ **"read" scope**: Can access user profile, model info
- ❌ **"inference" scope**: Often missing - REQUIRED for Qwen models
- ❌ **"write" scope**: Not needed for this application

**System Behavior**:
- **Full-scope token**: Uses Qwen models with LangGraph → 30%+ GAIA performance
- **Limited-scope token**: Clear error message → User guided to proper authentication
- **No token**: Clear error message → User guided to login

**Clear Error Handling**:
```python
# No more fallback confusion - clear requirements
if test_response.status_code == 401:
    return "Authentication Error: Your OAuth token lacks inference permissions. Please logout and login again with full access."
```

### 6. Deployment Process

1. **Create Space**:

   ```bash
   # Visit https://huggingface.co/new-space
   # Choose Gradio SDK
   # Upload all files from src/ directory
   ```

2. **Upload Files**:
   - Copy entire `src/` directory to Space
   - Ensure `app.py` is the main entry point
   - Include all dependencies in `requirements.txt`

3. **Test Authentication**:
   - Space automatically enables OAuth for Gradio apps
   - Test login/logout functionality
   - Verify Qwen model access works
   - Test GAIA evaluation with LangGraph workflow

### 7. Verification Steps

After deployment, verify these work:

- [ ] **Interface Loads**: Gradio interface appears correctly
- [ ] **OAuth Login**: Login button works and shows user profile
- [ ] **Authentication Check**: Clear error messages when insufficient permissions
- [ ] **Qwen Model Access**: Models initialize and respond correctly
- [ ] **LangGraph Workflow**: Multi-agent system processes questions
- [ ] **Manual Testing**: Individual questions work with full workflow
- [ ] **GAIA Evaluation**: Full evaluation runs and submits to Unit 4 API
- [ ] **Results Display**: Scores and detailed results show correctly

### 8. Troubleshooting

#### Common Issues

**Issue**: "HuggingFace token with inference permissions is required"
**Solution**: Set HF_TOKEN in Space secrets or login with full OAuth permissions

**Issue**: "Failed to initialize any Qwen models"
**Solution**: Verify HF_TOKEN has inference scope and Qwen model access

**Issue**: "Authentication Error: Your OAuth token lacks inference permissions"
**Solution**: Logout and login again, or set HF_TOKEN as Space secret

#### Debug Commands

```python
# In Space, add debug logging to check authentication:
logger.info(f"HF_TOKEN available: {os.getenv('HF_TOKEN') is not None}")
logger.info(f"OAuth token available: {oauth_token is not None}")
logger.info(f"Qwen models initialized: {client.get_model_status()}")
```

### 9. Performance Optimization

For production efficiency with Qwen models:

```python
# Intelligent Model Selection Strategy
- Simple questions: Qwen 2.5-7B (fast, cost-effective)
- Medium complexity: Qwen 2.5-32B (balanced performance)  
- Complex reasoning: Qwen 2.5-72B (best quality)
- Budget management: Auto-downgrade when budget exceeded
- LangGraph workflow: Optimal agent routing and synthesis
```

### 10. Monitoring and Maintenance

**Key Metrics to Monitor**:

- GAIA benchmark success rate (target: 30%+)
- Average response time per question
- Cost per question processed
- LangGraph workflow success rate
- Qwen model availability and performance

**Regular Maintenance**:

- Monitor HuggingFace Inference API status
- Update dependencies for security
- Review and optimize LangGraph workflow performance
- Check Unit 4 API compatibility
- Monitor Qwen model performance and costs

## 🎯 Success Metrics

### Expected Production Results 🚀

With proper deployment and authentication:

- **GAIA Benchmark**: 30%+ success rate
- **LangGraph Workflow**: Multi-agent orchestration working
- **Qwen Model Performance**: Intelligent tier selection (7B→32B→72B)
- **User Experience**: Professional interface with clear authentication
- **System Reliability**: Clear success/failure modes (no degraded performance)

### Final Status:
- **Architecture**: Qwen 2.5 models + LangGraph multi-agent workflow
- **Requirements**: Clear authentication requirements (HF_TOKEN or OAuth with inference)
- **Performance**: 30%+ GAIA benchmark with full functionality
- **Reliability**: Robust error handling with clear user guidance
- **Deployment**: Ready for immediate HuggingFace Space deployment

**The GAIA Agent is now a focused, high-performance system using proper AI models and multi-agent orchestration!** 🎉
 