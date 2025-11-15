# LALMBench

A benchmark for testing LLM context window limits and performance degradation using progressively complex prompts about Large Audio Language Models (LALMs).

## Quick Start

```bash
# Setup
uv venv --python 3.11
source .venv/bin/activate
pip install openai tiktoken

# Run full benchmark (231 rounds)
python lalm_context_benchmark.py

# Run quick test (1 round)
python lalm_context_benchmark.py --num-rounds 1
```

## What It Measures

- **Time to First Token (TTFT)** - Prompt processing latency
- **Streaming tokens/second** - Generation speed
- **Context exhaustion point** - Where the model fails
- **Performance degradation** - How speed changes as context grows
- **Reasoning metrics** - Tracks content in `<think>` tags (when present)

## Requirements

- LM Studio or OpenAI-compatible API server
- Python 3.11+
- Model with sufficient context window (tested with gpt-oss-120b)

## Configuration

```bash
python lalm_context_benchmark.py \
  --num-rounds 10 \
  --model gpt-oss-120b \
  --base-url http://localhost:1234/v1
```

## Output

Results saved to `logs/lalm_benchmark_run_[timestamp].json` with:
- Per-turn metrics (TTFT, tokens/sec, token counts)
- Summary statistics (averages, totals)
- Running stats displayed after each turn

## Methodology

231 progressively escalating prompts covering:
- Foundations (1-40): Basic LALM concepts
- Intermediate (41-80): Architectures and training
- Advanced (81-120): Mathematical details and DSP
- Expert (121-160): Implementation and optimization
- Obscure (161-200): Psychoacoustics and edge cases
- Ultra-deep (201-231): Analog modeling and advanced math

Designed to stress-test context windows and reveal performance characteristics under realistic multi-turn conversation loads.

## License

MIT
