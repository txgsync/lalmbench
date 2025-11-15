# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a benchmarking tool for testing context window limits of local LLMs. It runs progressively escalating technical conversations about Large Audio Language Models (LALMs) and music production to stress-test context windows and measure performance degradation.

## Environment Setup

```bash
# Create Python 3.11 virtual environment
uv venv --python 3.11

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install openai tiktoken
```

## Running the Benchmark

The benchmark connects to a local LLM server (default: LM Studio at http://localhost:1234/v1) and runs a conversation using prompts from `lalm_prompts.json`.

### Quick Test (1 round)
```bash
python lalm_context_benchmark.py --num-rounds 1
```

### Full Benchmark (231 rounds)
```bash
python lalm_context_benchmark.py
```

### Custom Configuration
```bash
python lalm_context_benchmark.py --num-rounds 10 --model gpt-oss-120b --base-url http://localhost:1234/v1
```

## Architecture

### Core Components

**`lalm_context_benchmark.py`** - Single-file benchmark implementation with:

1. **Data Classes**
   - `TurnMetrics`: Captures per-turn performance metrics (TTFT, tokens/sec, token counts, reasoning metrics)
   - `BenchmarkMetrics`: Aggregates metrics across all turns and generates summary statistics

2. **LALMBenchmark Class**
   - Uses OpenAI-compatible API client to communicate with local LLM server
   - Maintains conversation state across turns
   - Executes streaming completions and tracks real-time metrics

3. **Reasoning Tracking**
   - Automatically detects and measures content within `<think>` tags
   - Tracks reasoning token count, character length, and generation time
   - Uses regex pattern: `<think>(.*?)</think>` to extract reasoning content

### Key Implementation Details

**Streaming Logic** (`execute_turn` method):
- Handles chunks without `choices` array (usage-only chunks) to prevent IndexError
- Tracks first token timing separately from reasoning token timing
- Accumulates text incrementally to detect `<think>` tag boundaries during streaming
- Records timestamps when entering/exiting reasoning mode

**Metrics Calculated**:
- Time to First Token (TTFT): Latency from request to first content token
- Streaming tokens/second: Generation speed after first token
- Reasoning metrics: Tokens (using tiktoken), time, and percentage of turns with reasoning
- Conversation token accumulation: Running total across all turns

**Token Counting**:
- Uses tiktoken (`cl100k_base` encoding, same as GPT-4) for accurate token counting
- Falls back to API-provided token counts when available
- Reasoning tokens are counted using tiktoken to extract and measure `<think>` tag content

**API Configuration**:
- Default parameter: `reasoning_effort="low"` for models supporting reasoning
- `temperature=0.7`, `max_tokens=32768`
- Streaming enabled with usage info: `stream_options={"include_usage": True}`

### Prompt Structure

`lalm_prompts.json` contains 231 prompts organized in progressive difficulty tiers:
- Foundations (1-40): Basic concepts
- Intermediate (41-80): Architectures and training
- Advanced (81-120): Mathematical details and DSP
- Expert (121-160): Implementation and optimization
- Obscure (161-200): Psychoacoustics and edge cases
- Ultra-deep (201-231): Analog modeling and advanced math

### Output Files

Benchmark results are saved in `logs/lalm_benchmark_run_[timestamp].json` with:
- Summary statistics (averages, totals, reasoning stats)
- Per-turn detailed metrics (all TurnMetrics fields)
- Reasoning statistics when applicable

The `logs/` directory is automatically created on first run and is excluded from git.

## Important Constraints

- Requires LM Studio or OpenAI-compatible server running locally
- Model must support streaming API
- The `reasoning_effort` parameter is optional; unsupported models will ignore it
- Empty `choices` arrays in stream chunks are handled gracefully
