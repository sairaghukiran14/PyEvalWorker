import json
from pathlib import Path
from typing import List, Dict, Any, Callable
import pytest

from app.schemas import RagEvalRequest
from app.rag_metrics import evaluate_rag_pipeline

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Established CI Gate Quality Thresholds
BASELINE_THRESHOLDS = {
    "min_avg_faithfulness": 0.85,
    "min_avg_answer_relevance": 4.0,
    "min_avg_context_precision": 0.80,
    "min_avg_context_recall": 0.80,
}


def load_golden_set() -> List[Dict[str, Any]]:
    path = DATA_DIR / "golden_qa.jsonl"
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line.strip()))
    return cases


def evaluate_candidate_suite(answer_provider: Callable[[Dict[str, Any]], str]) -> Dict[str, float]:
    """Runs evaluation over the 30 golden QA cases with the candidate's answers."""
    cases = load_golden_set()
    assert len(cases) == 30

    faith_scores = []
    rel_scores = []
    prec_scores = []
    rec_scores = []

    for case in cases:
        candidate_ans = answer_provider(case)
        req = RagEvalRequest(
            question=case["question"],
            answer=candidate_ans,
            chunks=case["chunks"],
            reference=case.get("reference")
        )
        res = evaluate_rag_pipeline(req)
        faith_scores.append(res.faithfulness)
        rel_scores.append(res.answer_relevance)
        prec_scores.append(res.context_precision)
        rec_scores.append(res.context_recall)

    metrics = {
        "avg_faithfulness": sum(faith_scores) / len(faith_scores),
        "avg_answer_relevance": sum(rel_scores) / len(rel_scores),
        "avg_context_precision": sum(prec_scores) / len(prec_scores),
        "avg_context_recall": sum(rec_scores) / len(rec_scores),
    }
    return metrics


def check_ci_gate_thresholds(metrics: Dict[str, float]) -> List[str]:
    """Checks metrics against minimum acceptable thresholds."""
    violations = []
    if metrics["avg_faithfulness"] < BASELINE_THRESHOLDS["min_avg_faithfulness"]:
        violations.append(
            f"Faithfulness {metrics['avg_faithfulness']:.3f} fell below threshold {BASELINE_THRESHOLDS['min_avg_faithfulness']}"
        )
    if metrics["avg_answer_relevance"] < BASELINE_THRESHOLDS["min_avg_answer_relevance"]:
        violations.append(
            f"Answer Relevance {metrics['avg_answer_relevance']:.3f} fell below threshold {BASELINE_THRESHOLDS['min_avg_answer_relevance']}"
        )
    if metrics["avg_context_precision"] < BASELINE_THRESHOLDS["min_avg_context_precision"]:
        violations.append(
            f"Context Precision {metrics['avg_context_precision']:.3f} fell below threshold {BASELINE_THRESHOLDS['min_avg_context_precision']}"
        )
    if metrics["avg_context_recall"] < BASELINE_THRESHOLDS["min_avg_context_recall"]:
        violations.append(
            f"Context Recall {metrics['avg_context_recall']:.3f} fell below threshold {BASELINE_THRESHOLDS['min_avg_context_recall']}"
        )
    return violations


def test_ci_eval_gate_baseline_passes():
    """
    Evaluates the baseline candidate model on all 30 questions.
    Verifies that all metrics comfortably clear the quality gate.
    """
    # Baseline candidate answers from golden dataset
    metrics = evaluate_candidate_suite(lambda case: case["candidate_answer"])

    print("\n--- BASELINE CANDIDATE EVALUATION RESULTS ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    violations = check_ci_gate_thresholds(metrics)
    assert not violations, f"Baseline candidate failed CI gate: {violations}"
    assert metrics["avg_faithfulness"] >= 0.85
    assert metrics["avg_answer_relevance"] >= 4.0


def test_ci_eval_gate_deliberately_worse_prompt_fails_build():
    """
    Feature #7 requirement: 'A deliberately worse prompt turns the build red.'
    Simulates a degraded prompt regression that produces hallucinations & evasive answers.
    Asserts that the CI gate catches the drop and raises an AssertionError.
    """
    def degraded_candidate_generator(case: Dict[str, Any]) -> str:
        # Deliberately degraded prompt that introduces hallucinations and off-topic fillers
        return "FastAPI is a modern web framework that deletes all chunks and requires fine-tuning Llama-3."

    metrics = evaluate_candidate_suite(degraded_candidate_generator)

    print("\n--- DEGRADED CANDIDATE EVALUATION RESULTS ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    violations = check_ci_gate_thresholds(metrics)

    # The gate MUST trigger violations and fail
    assert len(violations) > 0, "Gate should have caught degraded prompt regression!"
    assert metrics["avg_faithfulness"] < BASELINE_THRESHOLDS["min_avg_faithfulness"]

    # Demonstrate that an assertion on the gate threshold fails as expected
    with pytest.raises(AssertionError) as exc_info:
        assert metrics["avg_faithfulness"] >= BASELINE_THRESHOLDS["min_avg_faithfulness"], (
            f"CI GATE FAILURE: Model faithfulness dropped to {metrics['avg_faithfulness']:.3f}!"
        )

    assert "CI GATE FAILURE" in str(exc_info.value)
