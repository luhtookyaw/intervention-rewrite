#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE_RE = re.compile(
    r"^eval_(beginner|intermediate|advanced)_([a-z0-9_]+)\.json$",
    re.IGNORECASE,
)


def parse_level(path: Path) -> str | None:
    m = EVAL_FILE_RE.match(path.name)
    if not m:
        return None
    return m.group(1).lower()


def to_float_score(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def mean(sum_value: float, count_value: int) -> float | None:
    if count_value == 0:
        return None
    return round(sum_value / count_value, 4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Average evaluation scores across levels (beginner/intermediate/advanced) "
            "from outputs/evaluations/eval_*.json"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "outputs" / "evaluations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "evaluations" / "averages_by_level.json",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    files = sorted(args.input_dir.glob("eval_*.json"))
    if not files:
        raise ValueError(f"No eval files found in: {args.input_dir}")

    level_counts: dict[str, int] = defaultdict(int)
    alliance_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    alliance_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    skill_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    skill_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for file_path in files:
        level = parse_level(file_path)
        if level is None:
            continue

        data = json.loads(file_path.read_text(encoding="utf-8"))
        level_counts[level] += 1

        alliance_parsed = data.get("alliance_evaluation", {}).get("parsed", {})
        if isinstance(alliance_parsed, dict):
            for q_key, q_obj in alliance_parsed.items():
                if not isinstance(q_obj, dict):
                    continue
                score = to_float_score(q_obj.get("score"))
                if score is None:
                    continue
                alliance_sums[level][q_key] += score
                alliance_counts[level][q_key] += 1

        therapist_scores = data.get("therapist_skill_evaluations", {})
        if isinstance(therapist_scores, dict):
            for metric, result in therapist_scores.items():
                if not isinstance(result, dict):
                    continue
                score = to_float_score(result.get("score"))
                if score is None:
                    continue
                skill_sums[level][metric] += score
                skill_counts[level][metric] += 1

    levels = sorted(level_counts.keys())
    summary: dict[str, Any] = {"levels": {}}

    for level in levels:
        alliance_avg: dict[str, float] = {}
        for key in sorted(alliance_sums[level].keys(), key=lambda x: (len(x), x)):
            avg = mean(alliance_sums[level][key], alliance_counts[level][key])
            if avg is not None:
                alliance_avg[key] = avg

        skill_avg: dict[str, float] = {}
        for key in sorted(skill_sums[level].keys()):
            avg = mean(skill_sums[level][key], skill_counts[level][key])
            if avg is not None:
                skill_avg[key] = avg

        alliance_overall = mean(sum(alliance_sums[level].values()), sum(alliance_counts[level].values()))
        skills_overall = mean(sum(skill_sums[level].values()), sum(skill_counts[level].values()))

        combined_sum = sum(alliance_sums[level].values()) + sum(skill_sums[level].values())
        combined_count = sum(alliance_counts[level].values()) + sum(skill_counts[level].values())
        overall = mean(combined_sum, combined_count)

        summary["levels"][level] = {
            "num_files": level_counts[level],
            "alliance_question_averages": alliance_avg,
            "therapist_skill_averages": skill_avg,
            "alliance_overall_average": alliance_overall,
            "therapist_skills_overall_average": skills_overall,
            "overall_average": overall,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Processed {len(files)} evaluation files from: {args.input_dir}")
    print(f"Wrote averages to: {args.output}")


if __name__ == "__main__":
    main()
