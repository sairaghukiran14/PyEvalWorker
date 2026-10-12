import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from app.schemas import JudgeRequest
from app.judge import evaluate_with_judge

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def validate_judge_agreement():
    labels_file = DATA_DIR / "human_labels.jsonl"
    if not labels_file.exists():
        raise FileNotFoundError(f"Human labels file not found: {labels_file}")

    cases: List[Dict[str, Any]] = []
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    print(f"\n=======================================================")
    print(f"  RUNNING LLM JUDGE VALIDATION ACROSS {len(cases)} CASES")
    print(f"=======================================================\n")

    results = []
    agree_within_one = 0
    exact_match = 0
    total = len(cases)

    print(f"{'ID':<6} | {'Rubric':<20} | {'Human':<6} | {'Judge':<6} | {'Diff':<5} | {'Agrees (±1)?'}")
    print("-" * 65)

    for case in cases:
        case_id = case.get("id", "N/A")
        rubric = case.get("rubric", "faithfulness")
        human_score = float(case.get("human_score", 5))

        req = JudgeRequest(
            output=case.get("output", ""),
            source=case.get("source"),
            question=case.get("question"),
            context=case.get("context", []),
            rubric=rubric
        )

        resp = evaluate_with_judge(req)
        # Get score for rubric
        judge_score = resp.scores.get(rubric, list(resp.scores.values())[0] if resp.scores else 5.0)
        diff = abs(judge_score - human_score)

        is_agreed = diff <= 1.0
        if is_agreed:
            agree_within_one += 1
        if diff == 0.0:
            exact_match += 1

        results.append({
            "id": case_id,
            "rubric": rubric,
            "human_score": human_score,
            "judge_score": judge_score,
            "diff": round(diff, 2),
            "agreed": is_agreed,
            "judge_reasoning": resp.reasoning,
            "human_reasoning": case.get("human_reasoning", "")
        })

        status_str = "✓ YES" if is_agreed else "✗ NO"
        print(f"{case_id:<6} | {rubric:<20} | {human_score:<6.1f} | {judge_score:<6.1f} | {diff:<5.1f} | {status_str}")

    agreement_rate = (agree_within_one / total) * 100.0
    exact_rate = (exact_match / total) * 100.0
    mae = sum(r["diff"] for r in results) / total

    print("-" * 65)
    print(f"\nValidation Summary:")
    print(f"Total Hand-Labeled Cases : {total}")
    print(f"Agreement within ±1 point: {agree_within_one}/{total} ({agreement_rate:.1f}%)")
    print(f"Exact Score Matches      : {exact_match}/{total} ({exact_rate:.1f}%)")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    # Save to validation_results.json
    output_file = DATA_DIR / "validation_results.json"
    summary = {
        "total_cases": total,
        "agreement_count": agree_within_one,
        "agreement_percentage": round(agreement_rate, 2),
        "exact_matches": exact_match,
        "exact_percentage": round(exact_rate, 2),
        "mean_absolute_error": round(mae, 2),
        "cases": results
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved detailed validation results to: {output_file}\n")
    return summary


if __name__ == "__main__":
    validate_judge_agreement()
