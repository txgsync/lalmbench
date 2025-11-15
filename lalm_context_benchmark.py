#!/usr/bin/env python3
"""
LALM Context Exhaustion Benchmark

A comprehensive 200+ turn benchmark testing context window limits of local LLMs
through progressively escalating technical discussions about Large Audio Language
Models (LALMs) and music audio production.

Tracks:
- Time to first token (TTFT) - prompt processing time
- Streaming tokens/second - generation speed after first token
- Turn counts (user turns, assistant turns, total)
- Token counts (prompt, completion, total conversation)
- Reasoning metrics (tokens, time, character length in <think> tags)
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from openai import OpenAI
import tiktoken


@dataclass
class TurnMetrics:
    """Metrics for a single conversation turn"""
    turn_number: int
    user_turn_number: int
    assistant_turn_number: int
    user_prompt_length: int
    assistant_response_length: int

    # Timing metrics
    time_to_first_token_ms: float
    streaming_tokens_per_sec: float
    total_turn_time_sec: float

    # Token metrics
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    conversation_total_tokens: Optional[int] = None

    # Reasoning metrics
    reasoning_tokens: Optional[int] = None
    reasoning_char_length: Optional[int] = None
    reasoning_time_sec: Optional[float] = None

    # Response metadata
    finish_reason: Optional[str] = None
    timestamp: str = ""


class BenchmarkMetrics:
    """Tracks comprehensive metrics across the benchmark run"""

    def __init__(self):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.turn_metrics: List[TurnMetrics] = []
        self.total_user_turns = 0
        self.total_assistant_turns = 0
        self.cumulative_conversation_tokens = 0

    def add_turn_metrics(self, metrics: TurnMetrics):
        """Add metrics for a completed turn"""
        self.turn_metrics.append(metrics)
        self.total_user_turns = metrics.user_turn_number
        self.total_assistant_turns = metrics.assistant_turn_number
        if metrics.total_tokens:
            self.cumulative_conversation_tokens = metrics.conversation_total_tokens or 0

    def get_summary(self) -> Dict:
        """Generate summary statistics"""
        if not self.turn_metrics:
            return {}

        avg_ttft = sum(m.time_to_first_token_ms for m in self.turn_metrics) / len(self.turn_metrics)
        avg_streaming_speed = sum(m.streaming_tokens_per_sec for m in self.turn_metrics if m.streaming_tokens_per_sec > 0) / len([m for m in self.turn_metrics if m.streaming_tokens_per_sec > 0])
        total_time = time.time() - self.start_time

        # Calculate reasoning statistics
        reasoning_turns = [m for m in self.turn_metrics if m.reasoning_tokens and m.reasoning_tokens > 0]
        total_reasoning_tokens = sum(m.reasoning_tokens for m in reasoning_turns)
        total_reasoning_time = sum(m.reasoning_time_sec for m in reasoning_turns if m.reasoning_time_sec)
        avg_reasoning_time = total_reasoning_time / len(reasoning_turns) if reasoning_turns else 0
        reasoning_percentage = (len(reasoning_turns) / len(self.turn_metrics) * 100) if self.turn_metrics else 0

        return {
            "run_id": self.run_id,
            "total_turns": len(self.turn_metrics),
            "total_user_turns": self.total_user_turns,
            "total_assistant_turns": self.total_assistant_turns,
            "total_conversation_tokens": self.cumulative_conversation_tokens,
            "total_runtime_sec": total_time,
            "avg_time_to_first_token_ms": avg_ttft,
            "avg_streaming_tokens_per_sec": avg_streaming_speed,
            "last_turn_completed": len(self.turn_metrics),
            "reasoning_stats": {
                "total_reasoning_tokens": total_reasoning_tokens,
                "total_reasoning_time_sec": total_reasoning_time,
                "avg_reasoning_time_sec": avg_reasoning_time,
                "turns_with_reasoning": len(reasoning_turns),
                "reasoning_percentage": reasoning_percentage
            }
        }

    def save_to_json(self, filepath: Optional[Path] = None):
        """Save complete metrics to JSON file"""
        if filepath is None:
            # Create logs directory if it doesn't exist
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            filepath = logs_dir / f"lalm_benchmark_run_{self.run_id}.json"

        data = {
            "summary": self.get_summary(),
            "turns": [asdict(m) for m in self.turn_metrics]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath


def extract_reasoning_content(text: str, encoding) -> Tuple[str, int, int]:
    """
    Extract content within <think> tags and calculate metrics using tiktoken.

    Args:
        text: The full response text
        encoding: tiktoken encoding to use for token counting

    Returns:
        Tuple of (reasoning_content, char_length, actual_tokens)
    """
    think_pattern = r'<think>(.*?)</think>'
    matches = re.findall(think_pattern, text, re.DOTALL)

    if not matches:
        return "", 0, 0

    reasoning_content = ''.join(matches)
    char_length = len(reasoning_content)

    # Use actual tokenizer for accurate count
    actual_tokens = len(encoding.encode(reasoning_content))

    return reasoning_content, char_length, actual_tokens


class LALMBenchmark:
    """Main benchmark orchestrator"""

    def __init__(self, base_url: str = "http://localhost:1234/v1", model: Optional[str] = None):
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self.model = model
        self.messages: List[Dict[str, str]] = []
        self.metrics = BenchmarkMetrics()
        self.conversation_total_tokens = 0

        # Initialize tiktoken encoding for accurate token counting
        # Use cl100k_base which is used by GPT-4 and GPT-3.5-turbo
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            # Fail explicitly if encoding is unavailable - don't silently use gpt2
            raise RuntimeError(
                f"Failed to load cl100k_base tiktoken encoding: {e}\n"
                "This encoding is required for accurate token counting.\n"
                "Please verify tiktoken is properly installed: pip install --upgrade tiktoken\n"
                "Refusing to use gpt2 fallback as it may produce different token counts."
            ) from e

    def get_model_name(self) -> str:
        """Get the currently loaded model from LM Studio"""
        if self.model:
            return self.model

        models = self.client.models.list()
        if not models.data:
            raise RuntimeError("No models loaded in LM Studio")

        self.model = models.data[0].id
        return self.model

    def execute_turn(self, user_prompt: str, turn_number: int) -> TurnMetrics:
        """Execute a single conversation turn with streaming"""

        # Add user message
        self.messages.append({"role": "user", "content": user_prompt})
        user_turn_num = (len(self.messages) + 1) // 2

        # Start timing
        request_start = time.time()

        # Make streaming request
        stream = self.client.chat.completions.create(
            model=self.get_model_name(),
            messages=self.messages,
            stream=True,
            stream_options={"include_usage": True},
            temperature=0.7,
            max_tokens=32768,
            reasoning_effort="low"
        )

        # Track streaming
        first_token_time = None
        first_reasoning_token_time = None
        last_reasoning_token_time = None
        collected_chunks = []
        usage_info = None
        finish_reason = None
        in_reasoning = False
        current_text = ""

        for chunk in stream:
            # Skip chunks without choices (e.g., usage-only chunks)
            if not chunk.choices:
                # Collect usage info from chunks without choices
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_info = chunk.usage
                continue

            # Track time to first token
            if first_token_time is None and chunk.choices[0].delta.content:
                first_token_time = time.time()

            # Collect content
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                collected_chunks.append(content)
                current_text += content

                # Track reasoning timing based on <think> tags
                # Check if we're entering reasoning mode
                if '<think>' in current_text and not in_reasoning:
                    in_reasoning = True
                    if first_reasoning_token_time is None:
                        first_reasoning_token_time = time.time()

                # Update last reasoning token time for any token while in reasoning mode
                if in_reasoning:
                    last_reasoning_token_time = time.time()

                # Check if we're exiting reasoning mode
                if '</think>' in current_text and in_reasoning:
                    in_reasoning = False

            # Track finish reason
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            # Collect usage info
            if hasattr(chunk, 'usage') and chunk.usage:
                usage_info = chunk.usage

        streaming_end = time.time()

        # Assemble response
        full_response = ''.join(collected_chunks)
        self.messages.append({"role": "assistant", "content": full_response})
        assistant_turn_num = len(self.messages) // 2

        # Extract reasoning metrics using tiktoken
        reasoning_content, reasoning_char_len, reasoning_tokens = extract_reasoning_content(
            full_response, self.encoding
        )
        reasoning_time = None
        if first_reasoning_token_time and last_reasoning_token_time:
            reasoning_time = last_reasoning_token_time - first_reasoning_token_time

        # Calculate metrics
        ttft_ms = (first_token_time - request_start) * 1000 if first_token_time else 0

        # Calculate streaming duration (time from first token to end)
        if first_token_time:
            streaming_duration = streaming_end - first_token_time
        else:
            streaming_duration = 0

        # Calculate tokens per second for streaming phase
        # Use actual API token count if available, otherwise use tiktoken
        if usage_info and usage_info.completion_tokens:
            completion_tokens = usage_info.completion_tokens
        else:
            # Use tiktoken for accurate token counting
            completion_tokens = len(self.encoding.encode(full_response))

        tokens_per_sec = completion_tokens / streaming_duration if streaming_duration > 0 else 0

        total_turn_time = streaming_end - request_start

        # Update cumulative conversation token count
        if usage_info and usage_info.total_tokens:
            # Use the total from the API which includes full conversation context
            self.conversation_total_tokens = usage_info.total_tokens
        elif usage_info and (usage_info.prompt_tokens or usage_info.completion_tokens):
            # Manual accumulation: add both prompt and completion tokens
            # This happens when API provides individual token counts but not total
            prompt_tokens = usage_info.prompt_tokens or 0
            completion_tokens = usage_info.completion_tokens or 0
            self.conversation_total_tokens += (prompt_tokens + completion_tokens)
        else:
            # No token information available from API
            print(f"WARNING (Turn {turn_number}): No token count information from API. "
                  "Conversation token accumulation may be inaccurate.")

        # Create metrics object
        metrics = TurnMetrics(
            turn_number=turn_number,
            user_turn_number=user_turn_num,
            assistant_turn_number=assistant_turn_num,
            user_prompt_length=len(user_prompt),
            assistant_response_length=len(full_response),
            time_to_first_token_ms=ttft_ms,
            streaming_tokens_per_sec=tokens_per_sec,
            total_turn_time_sec=total_turn_time,
            prompt_tokens=usage_info.prompt_tokens if usage_info else None,
            completion_tokens=usage_info.completion_tokens if usage_info else None,
            total_tokens=usage_info.total_tokens if usage_info else None,
            conversation_total_tokens=self.conversation_total_tokens,
            reasoning_tokens=reasoning_tokens if reasoning_tokens > 0 else None,
            reasoning_char_length=reasoning_char_len if reasoning_char_len > 0 else None,
            reasoning_time_sec=reasoning_time,
            finish_reason=finish_reason,
            timestamp=datetime.now().isoformat()
        )

        return metrics

    def run_benchmark(self, prompts: List[str], num_rounds: Optional[int] = None):
        """Run the full benchmark with all prompts

        Args:
            prompts: List of prompts to use
            num_rounds: Optional limit on number of rounds to run (for testing)
        """
        # Limit prompts if num_rounds is specified
        if num_rounds is not None:
            prompts = prompts[:num_rounds]

        print(f"{'='*80}")
        print(f"LALM Context Exhaustion Benchmark")
        print(f"{'='*80}")
        print(f"Model: {self.get_model_name()}")
        print(f"Total prompts: {len(prompts)}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

        try:
            for i, prompt in enumerate(prompts, 1):
                print(f"\n--- Turn {i}/{len(prompts)} ---")

                metrics = self.execute_turn(prompt, i)
                self.metrics.add_turn_metrics(metrics)

                # Display metrics
                print(f"User turn #{metrics.user_turn_number} | Assistant turn #{metrics.assistant_turn_number}")
                print(f"TTFT: {metrics.time_to_first_token_ms:.1f}ms | Streaming: {metrics.streaming_tokens_per_sec:.1f} tok/s")
                print(f"Tokens - Prompt: {metrics.prompt_tokens or 'N/A'} | Completion: {metrics.completion_tokens or 'N/A'} | Total in conversation: {metrics.conversation_total_tokens or 'N/A'}")

                # Display reasoning metrics if present
                if metrics.reasoning_tokens:
                    print(f"Reasoning - Tokens: {metrics.reasoning_tokens} | Chars: {metrics.reasoning_char_length} | Time: {metrics.reasoning_time_sec:.2f}s")

                print(f"Turn time: {metrics.total_turn_time_sec:.2f}s | Finish: {metrics.finish_reason}")

                # Display running statistics after each turn
                self.print_running_stats()

        except Exception as e:
            print(f"\n{'='*80}")
            print(f"BENCHMARK STOPPED - Error encountered")
            print(f"{'='*80}")
            print(f"Error: {type(e).__name__}: {e}")
            print(f"Completed {len(self.metrics.turn_metrics)} turns before failure")

            # Print traceback for debugging
            import traceback
            print(f"\nTraceback:")
            traceback.print_exc()

        finally:
            # Print summary and save
            self.print_summary()
            log_file = self.metrics.save_to_json()
            print(f"\nDetailed metrics saved to: {log_file}")

    def print_running_stats(self):
        """Print running statistics after each turn"""
        summary = self.metrics.get_summary()

        print(f"\n{'─'*80}")
        print(f"Running Stats: {summary.get('total_turns', 0)} turns | "
              f"Tokens: {summary.get('total_conversation_tokens', 0):,} | "
              f"Avg TTFT: {summary.get('avg_time_to_first_token_ms', 0):.1f}ms | "
              f"Avg Speed: {summary.get('avg_streaming_tokens_per_sec', 0):.1f} tok/s")

        # Display reasoning statistics if any
        reasoning_stats = summary.get('reasoning_stats', {})
        if reasoning_stats.get('turns_with_reasoning', 0) > 0:
            print(f"Reasoning: {reasoning_stats.get('turns_with_reasoning', 0)} turns "
                  f"({reasoning_stats.get('reasoning_percentage', 0):.1f}%) | "
                  f"{reasoning_stats.get('total_reasoning_tokens', 0):,} tokens | "
                  f"{reasoning_stats.get('total_reasoning_time_sec', 0):.2f}s total")
        print(f"{'─'*80}")

    def print_summary(self):
        """Print benchmark summary"""
        summary = self.metrics.get_summary()

        print(f"\n{'='*80}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print(f"Total turns completed: {summary.get('total_turns', 0)}")
        print(f"User turns: {summary.get('total_user_turns', 0)}")
        print(f"Assistant turns: {summary.get('total_assistant_turns', 0)}")
        print(f"Total conversation tokens: {summary.get('total_conversation_tokens', 0):,}")
        print(f"Total runtime: {summary.get('total_runtime_sec', 0):.2f}s")
        print(f"Avg TTFT: {summary.get('avg_time_to_first_token_ms', 0):.1f}ms")
        print(f"Avg streaming speed: {summary.get('avg_streaming_tokens_per_sec', 0):.1f} tok/s")

        # Display reasoning statistics
        reasoning_stats = summary.get('reasoning_stats', {})
        if reasoning_stats.get('turns_with_reasoning', 0) > 0:
            print(f"\nReasoning Statistics:")
            print(f"  Turns with reasoning: {reasoning_stats.get('turns_with_reasoning', 0)} ({reasoning_stats.get('reasoning_percentage', 0):.1f}%)")
            print(f"  Total reasoning tokens: {reasoning_stats.get('total_reasoning_tokens', 0):,}")
            print(f"  Total reasoning time: {reasoning_stats.get('total_reasoning_time_sec', 0):.2f}s")
            print(f"  Avg reasoning time: {reasoning_stats.get('avg_reasoning_time_sec', 0):.2f}s")

        print(f"{'='*80}")


def load_lalm_prompts(prompts_file: str = "lalm_prompts.json") -> List[str]:
    """Load prompts from JSON file"""
    with open(prompts_file, 'r') as f:
        data = json.load(f)
    return data['prompts']


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Run LALM Context Exhaustion Benchmark')
    parser.add_argument('--num-rounds', type=int, default=None,
                        help='Number of rounds to run (default: all prompts)')
    parser.add_argument('--model', type=str, default="gpt-oss-120b",
                        help='Model name (default: gpt-oss-120b)')
    parser.add_argument('--base-url', type=str, default="http://localhost:1234/v1",
                        help='Base URL for API (default: http://localhost:1234/v1)')

    args = parser.parse_args()

    # Load prompts from JSON file
    prompts = load_lalm_prompts()

    # Create benchmark
    benchmark = LALMBenchmark(base_url=args.base_url, model=args.model)

    # Run benchmark
    benchmark.run_benchmark(prompts, num_rounds=args.num_rounds)


if __name__ == "__main__":
    main()
