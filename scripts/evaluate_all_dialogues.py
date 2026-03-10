#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_phase_dialogues import (  # noqa: E402
    combine_dialogues,
    evaluate_alliance,
    evaluate_therapist_skills,
    load_json,
)

FILENAME_RE = re.compile(r"^dialogues_([a-z0-9_]+)_([a-z0-9_]+)\.json$", re.IGNORECASE)


def iter_dialogue_files(dialogues_dir: Path) -> Iterable[Path]:
    yield from sorted(dialogues_dir.glob("dialogues_*.json"))


def parse_dialogue_filename(path: Path) -> tuple[str, str]:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected dialogue filename format: {path.name}")
    resistance_level, case_id = match.group(1).lower(), match.group(2).lower()
    return resistance_level, case_id


def build_eval_output_path(output_dir: Path, resistance_level: str, case_id: str) -> Path:
    return output_dir / f"eval_{resistance_level}_{case_id}.json"


def evaluate_one_file(
    input_path: Path,
    output_path: Path,
    model: str,
    temperature: float,
    skip_llm: bool,
) -> None:
    data = load_json(input_path)
    transcript = combine_dialogues(data)

    payload: dict = {
        "input": str(input_path),
        "model": model,
        "temperature": temperature,
        "transcript": transcript,
    }

    if not skip_llm:
        payload["alliance_evaluation"] = evaluate_alliance(
            transcript=transcript,
            model=model,
            temperature=temperature,
        )
        payload["therapist_skill_evaluations"] = evaluate_therapist_skills(
            transcript=transcript,
            model=model,
            temperature=temperature,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all dialogue files in outputs/dialogues and write outputs to "
            "outputs/evaluations/eval_<resistance_level>_<case_id>.json"
        )
    )
    parser.add_argument(
        "--dialogues-dir",
        type=Path,
        default=ROOT / "outputs" / "dialogues",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "evaluations",
    )
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only combine transcript and save it, do not call LLM.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any file evaluation fails.",
    )
    args = parser.parse_args()

    if not args.dialogues_dir.exists():
        raise FileNotFoundError(f"Dialogues directory not found: {args.dialogues_dir}")

    files = list(iter_dialogue_files(args.dialogues_dir))
    if not files:
        raise ValueError(f"No dialogue files found in: {args.dialogues_dir}")

    failures: list[str] = []
    for i, file_path in enumerate(files, start=1):
        try:
            resistance_level, case_id = parse_dialogue_filename(file_path)
            output_path = build_eval_output_path(args.output_dir, resistance_level, case_id)
            print(f"[{i}/{len(files)}] Evaluating {file_path.name} -> {output_path.name}")
            evaluate_one_file(
                input_path=file_path,
                output_path=output_path,
                model=args.model,
                temperature=args.temperature,
                skip_llm=args.skip_llm,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(file_path.name)
            print(f"Failed {file_path.name}: {exc}")
            if args.stop_on_error:
                break

    if failures:
        print(f"Done with failures: {len(failures)} file(s): {', '.join(failures)}")
        sys.exit(1)

    print(f"Done. Evaluated {len(files)} file(s).")


if __name__ == "__main__":
    main()
