#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate stage dialogues for all cases in the dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "Patient_PSi_CM_Dataset_Planning_Resistance.json",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any case generation fails.",
    )
    args = parser.parse_args()

    cases = load_cases(args.data)
    case_ids = [str(case.get("id", "")).strip() for case in cases if str(case.get("id", "")).strip()]
    if not case_ids:
        raise ValueError("No valid case ids found in dataset.")

    script_path = ROOT / "scripts" / "generate_stage_dialogues.py"
    failures: list[str] = []

    for i, case_id in enumerate(case_ids, start=1):
        cmd = [
            sys.executable,
            str(script_path),
            "--data",
            str(args.data),
            "--case-id",
            case_id,
            "--model",
            args.model,
            "--temperature",
            str(args.temperature),
        ]

        print(f"[{i}/{len(case_ids)}] Generating case_id={case_id}...")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            failures.append(case_id)
            print(f"Failed case_id={case_id} (exit={result.returncode})")
            if args.stop_on_error:
                break

    if failures:
        print(f"Done with failures: {len(failures)} case(s): {', '.join(failures)}")
        sys.exit(1)

    print(f"Done. Generated dialogues for {len(case_ids)} case(s).")


if __name__ == "__main__":
    main()
