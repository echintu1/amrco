# Live API evaluation dependencies
# Install only the SDKs for the models you plan to evaluate.

numpy>=1.24.0

# Model SDKs (install as needed):
openai>=1.30.0          # for GPT-4o AND Llama-3 via Together.ai (OpenAI-compatible)
anthropic>=0.40.0       # for Claude 3.5 Sonnet
google-generativeai>=0.8.0   # for Gemini 1.5 Pro

# MMLU dataset loading (recommended):
datasets>=2.18.0        # HuggingFace datasets — loads cais/mmlu automatically
