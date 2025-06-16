---
title: GAIA Agent System
emoji: 🤖
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 5.25.2
app_file: ./src/app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
# Required scopes for Qwen model access via Inference API
hf_oauth_scopes:
  - inference-api
short_description: Multi-Agent AI System for GAIA Benchmark Questions
suggested_hardware: cpu-upgrade
models:
  - Qwen/Qwen2.5-7B-Instruct
  - Qwen/Qwen2.5-32B-Instruct 
  - Qwen/Qwen2.5-72B-Instruct
tags:
  - GAIA
  - multi-agent
  - LangGraph
  - benchmark
  - reasoning
  - web-search
  - file-processing
  - question-answering
---

# 🤖 GAIA Agent System

Advanced Multi-Agent AI System for GAIA Benchmark Questions using LangGraph orchestration.

## Features

- **Multi-Agent Architecture**: Router, Web Research, File Processing, Reasoning, and Synthesizer agents
- **LangGraph Orchestration**: Intelligent workflow management with state tracking
- **Unit 4 API Integration**: Official GAIA benchmark submission and scoring
- **Smart Model Selection**: Tiered Qwen 2.5 models (7B/32B/72B) for optimal cost/performance
- **Comprehensive Tools**: Wikipedia search, web scraping, mathematical calculations, file analysis

## Usage

1. **Official GAIA Evaluation**: Login with HuggingFace and run complete benchmark
2. **Manual Testing**: Test individual questions with detailed reasoning analysis
3. **File Processing**: Upload and analyze CSV, images, code, and audio files

Check out the configuration reference at <https://huggingface.co/docs/hub/spaces-config-reference>

