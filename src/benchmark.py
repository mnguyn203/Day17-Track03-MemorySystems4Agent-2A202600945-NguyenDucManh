from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    import json
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = sum(1 for e in expected if e.lower() in ans_lower)
    if matches == len(expected):
        return 1.0
    if matches > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Add a lightweight quality score for offline mode."""
    # Simple length + recall check for offline proxy
    if recall_points(answer, expected) > 0:
        return 0.9 if len(answer) > 10 else 0.5
    return 0.2


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_recall = 0.0
    total_quality = 0.0
    total_compactions = 0
    total_memory_growth = 0
    eval_count = 0
    
    for conv in conversations:
        user_id = conv.get("user_id", "u_test")
        thread_id = conv.get("thread_id", "t_test")
        
        # 1. Feed turns
        for turn_msg in conv.get("turns", []):
            agent.reply(user_id, thread_id, turn_msg)
            
        # 2. Track tokens from thread
        total_agent_tokens += agent.token_usage(thread_id)
        total_prompt_tokens += agent.prompt_token_usage(thread_id)
        
        # 3. Ask recall questions in a fresh thread
        evals = conv.get("recall_questions", [])
        for ev in evals:
            fresh_thread = thread_id + "_eval"
            reply_dict = agent.reply(user_id, fresh_thread, ev["question"])
            ans = reply_dict["reply"]
            total_recall += recall_points(ans, ev.get("expected_contains", []))
            total_quality += heuristic_quality(ans, ev.get("expected_contains", []))
            eval_count += 1
            
        # 4. Compactions & Memory
        if hasattr(agent, 'compaction_count'):
            total_compactions += agent.compaction_count(thread_id)
        if hasattr(agent, 'memory_file_size'):
            total_memory_growth += agent.memory_file_size(user_id)

    avg_recall = total_recall / eval_count if eval_count > 0 else 0.0
    avg_quality = total_quality / eval_count if eval_count > 0 else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=total_memory_growth,
        compactions=total_compactions
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a markdown table or tabulated output."""
    try:
        from tabulate import tabulate
    except ImportError:
        return str(rows)
    
    headers = [
        "Agent", "Agent Tokens", "Prompt Tokens", 
        "Recall", "Quality", "Memory Bytes", "Compactions"
    ]
    data = [
        [
            r.agent_name, r.agent_tokens_only, r.prompt_tokens_processed,
            f"{r.recall_score:.2f}", f"{r.response_quality:.2f}",
            r.memory_growth_bytes, r.compactions
        ]
        for r in rows
    ]
    return tabulate(data, headers=headers, tablefmt="github")


def main() -> None:
    """Student TODO: run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    config = load_config(Path(__file__).resolve().parent.parent)

    # load both datasets from root/data
    data_dir = config.data_dir
    standard_data = load_conversations(data_dir / "conversations.json")
    stress_data = load_conversations(data_dir / "advanced_long_context.json")

    # Run Standard
    print("=== Standard Benchmark ===")
    if standard_data:
        b_agent = BaselineAgent(config, force_offline=False)
        a_agent = AdvancedAgent(config, force_offline=False)
        r1 = run_agent_benchmark("Baseline", b_agent, standard_data, config)
        r2 = run_agent_benchmark("Advanced", a_agent, standard_data, config)
        print(format_rows([r1, r2]))
    else:
        print("No standard data found.")

    print("\\n=== Long-Context Stress Benchmark ===")
    if stress_data:
        # Re-init for clean slate
        b_agent_stress = BaselineAgent(config, force_offline=False)
        a_agent_stress = AdvancedAgent(config, force_offline=False)
        r3 = run_agent_benchmark("Baseline", b_agent_stress, stress_data, config)
        r4 = run_agent_benchmark("Advanced", a_agent_stress, stress_data, config)
        print(format_rows([r3, r4]))
    else:
        print("No stress data found.")


if __name__ == "__main__":
    main()
