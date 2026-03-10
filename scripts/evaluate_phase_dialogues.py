#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alliance import C_ALLIANCE_SYSTEM_PROMPT, EXAMPLE_C_ALLIANCE
from src.llm import call_llm
from src.therapist_skills import (
    CBT_SPECIFIC_FOCUS,
    CBT_SPECIFIC_GUIDED_DISCOVERY_SKILL,
    CBT_SPECIFIC_STRATEGY,
    GEN_COLLABORATION,
    GEN_INTERPERSONAL,
    GEN_UNDERSTANDING,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Expected top-level JSON object in phase dialogues file.")
    return data


def combine_dialogues(data: dict[str, Any]) -> str:
    results = data.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Expected 'results' to be a list.")

    lines: list[str] = []
    for phase_block in results:
        if not isinstance(phase_block, dict):
            continue

        dialogue = phase_block.get("dialogue", [])
        if not isinstance(dialogue, list):
            continue

        for turn in dialogue:
            if not isinstance(turn, dict):
                continue
            client = str(turn.get("client_utterance", "")).strip()
            therapist = str(turn.get("therapist_utterance", "")).strip()
            if client:
                lines.append(f"Client: {client}")
            if therapist:
                lines.append(f"Therapist: {therapist}")

    if not lines:
        raise ValueError("No dialogue turns found to combine.")

    return "\n".join(lines)


def _clean_response_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.strip().startswith("```")
        ).strip()
    return cleaned


def extract_json(text: str) -> Any:
    cleaned = _clean_response_text(text)

    for opening, closing in (("{", "}"), ("[", "]")):
        pos = cleaned.find(opening)
        while pos != -1:
            depth = 0
            for i in range(pos, len(cleaned)):
                ch = cleaned[i]
                if ch == opening:
                    depth += 1
                elif ch == closing:
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[pos : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
            pos = cleaned.find(opening, pos + 1)

    return {"raw": text}


def extract_all_json_objects(text: str) -> list[Any]:
    cleaned = _clean_response_text(text)
    objects: list[Any] = []

    pos = 0
    while pos < len(cleaned):
        start_obj = cleaned.find("{", pos)
        start_arr = cleaned.find("[", pos)

        starts = [s for s in (start_obj, start_arr) if s != -1]
        if not starts:
            break
        start = min(starts)
        opening = cleaned[start]
        closing = "}" if opening == "{" else "]"

        depth = 0
        parsed_any = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    try:
                        objects.append(json.loads(candidate))
                        pos = i + 1
                        parsed_any = True
                        break
                    except json.JSONDecodeError:
                        pass
        if not parsed_any:
            pos = start + 1

    return objects


def parse_alliance_output(raw: str) -> dict[str, Any]:
    blocks = extract_all_json_objects(raw)
    if not blocks:
        return {"raw": raw}

    merged: dict[str, Any] = {}
    for block in blocks:
        if isinstance(block, list):
            for item in block:
                if isinstance(item, dict):
                    merged.update(item)
        elif isinstance(block, dict):
            merged.update(block)

    question_objects = [
        b
        for b in blocks
        if isinstance(b, dict) and any(k.startswith("Q") for k in b.keys())
    ]
    if question_objects:
        merged = {}
        for item in question_objects:
            q_key = next((k for k in item.keys() if re.fullmatch(r"Q\d+", k)), None)
            if q_key:
                merged[q_key] = item
            else:
                merged.update(item)

    return merged if merged else {"raw": raw}


def parse_score_and_reason(raw: str) -> dict[str, Any]:
    text = raw.strip()
    m = re.match(r"^\s*([0-6])\s*,\s*(.+)$", text, flags=re.DOTALL)
    if m:
        return {"score": int(m.group(1)), "reason": m.group(2).strip(), "raw": raw}

    m2 = re.search(r"\b([0-6])\b", text)
    if m2:
        return {"score": int(m2.group(1)), "reason": text, "raw": raw}

    return {"score": None, "reason": text, "raw": raw}


def evaluate_alliance(transcript: str, model: str, temperature: float) -> dict[str, Any]:
    prompt = (
        C_ALLIANCE_SYSTEM_PROMPT.replace(
            "{example}", json.dumps(EXAMPLE_C_ALLIANCE, ensure_ascii=False)
        ).replace("{conversation}", transcript)
    )

    raw = call_llm(
        system_prompt="You are a strict counseling evaluator. Return only what the prompt asks.",
        user_prompt=prompt,
        model=model,
        temperature=temperature,
    )
    parsed = parse_alliance_output(raw)
    return {"raw": raw, "parsed": parsed}


def evaluate_therapist_skills(
    transcript: str, model: str, temperature: float
) -> dict[str, dict[str, Any]]:
    prompts = {
        "guided_discovery": CBT_SPECIFIC_GUIDED_DISCOVERY_SKILL,
        "focus": CBT_SPECIFIC_FOCUS,
        "strategy": CBT_SPECIFIC_STRATEGY,
        "understanding": GEN_UNDERSTANDING,
        "interpersonal": GEN_INTERPERSONAL,
        "collaboration": GEN_COLLABORATION,
    }

    evaluations: dict[str, dict[str, Any]] = {}
    for name, template in prompts.items():
        user_prompt = template.replace("{conversation}", transcript)

        raw = call_llm(
            system_prompt="You are a strict CBT evaluator. Follow output format exactly.",
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
        )
        evaluations[name] = parse_score_and_reason(raw)
    return evaluations


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine phase dialogues into one transcript and evaluate using "
            "alliance + therapist skill prompts."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs" / "phase_dialogues.json",
        help="Path to phase dialogues JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "phase_dialogue_evaluation.json",
        help="Where to write evaluation result JSON.",
    )
    parser.add_argument(
        "--transcript-output",
        type=Path,
        default=ROOT / "outputs" / "combined_dialogue.txt",
        help="Where to write the combined Client/Therapist transcript.",
    )
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only combine transcript and save it, do not call LLM.",
    )
    args = parser.parse_args()

    data = load_json(args.input)
    transcript = combine_dialogues(data)

    args.transcript_output.parent.mkdir(parents=True, exist_ok=True)
    args.transcript_output.write_text(transcript, encoding="utf-8")

    output: dict[str, Any] = {
        "input": str(args.input),
        "transcript_output": str(args.transcript_output),
        "model": args.model,
        "temperature": args.temperature,
        "transcript": transcript,
    }

    if not args.skip_llm:
        output["alliance_evaluation"] = evaluate_alliance(
            transcript=transcript,
            model=args.model,
            temperature=args.temperature,
        )
        output["therapist_skill_evaluations"] = evaluate_therapist_skills(
            transcript=transcript,
            model=args.model,
            temperature=args.temperature,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Combined transcript written to: {args.transcript_output}")
    print(f"Evaluation written to: {args.output}")


if __name__ == "__main__":
    main()
