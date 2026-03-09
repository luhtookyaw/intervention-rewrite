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

from src.llm import call_llm

PHASES = ["precontemplation", "contemplation", "preparation"]
PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array.")
    return data


def get_case(cases: list[dict[str, Any]], case_id: str | None, index: int | None) -> dict[str, Any]:
    if case_id:
        for case in cases:
            if str(case.get("id")) == case_id:
                return case
        raise ValueError(f"Case id '{case_id}' not found.")
    idx = 0 if index is None else index
    if idx < 0 or idx >= len(cases):
        raise ValueError(f"Index {idx} out of range for {len(cases)} cases.")
    return cases[idx]


def format_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value)


def build_core_beliefs(case: dict[str, Any]) -> str:
    parts: list[str] = []
    for label, key in [
        ("Helpless", "helpless_belief"),
        ("Unlovable", "unlovable_belief"),
        ("Worthless", "worthless_belief"),
    ]:
        raw = case.get(key, [])
        if isinstance(raw, list):
            joined = ", ".join(str(x) for x in raw if str(x).strip())
        else:
            joined = str(raw).strip()
        if joined:
            parts.append(f"{label}: {joined}")
    return " | ".join(parts)


def build_template_values(case: dict[str, Any]) -> dict[str, str]:
    resistance_emotion = case.get("resistance_emotion", "")
    return {
        "name": format_field(case.get("name")),
        "history": format_field(case.get("history")),
        "core_beliefs": build_core_beliefs(case),
        "intermediate_beliefs": format_field(case.get("intermediate_belief")),
        "coping_strategies": format_field(case.get("coping_strategies")),
        "situation": format_field(case.get("situation")),
        "auto_thought": format_field(case.get("auto_thought")),
        "emotion": format_field(case.get("emotion")),
        "behavior": format_field(case.get("behavior")),
        "resistance_type": format_field(case.get("resistance_type")),
        "resistance_emotion": format_field(resistance_emotion),
        "resistance_emotions": format_field(resistance_emotion),
        "resistance_monologue": format_field(case.get("resistance_internal_monologue")),
        "resistance_level": format_field(case.get("resistance_level")),
    }


def safe_format(template: str, values: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        return values.get(match.group(1), "")

    return PLACEHOLDER_PATTERN.sub(repl, template)


def to_dashed_text(value: Any, indent: int = 0) -> str:
    space = "  " * indent
    lines: list[str] = []

    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{space}- {k}:")
                lines.append(to_dashed_text(v, indent + 1))
            else:
                lines.append(f"{space}- {k}: {v}")
        return "\n".join(lines)

    if isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                if isinstance(item, dict) and item:
                    keys = list(item.keys())
                    first_key = keys[0]
                    first_val = item[first_key]
                    if isinstance(first_val, (dict, list)):
                        lines.append(f"{space}- {first_key}:")
                        lines.append(to_dashed_text(first_val, indent + 1))
                    else:
                        lines.append(f"{space}- {first_key}: {first_val}")

                    for k in keys[1:]:
                        v = item[k]
                        if isinstance(v, (dict, list)):
                            lines.append(f"{space}  - {k}:")
                            lines.append(to_dashed_text(v, indent + 2))
                        else:
                            lines.append(f"{space}  - {k}: {v}")
                else:
                    lines.append(f"{space}-")
                    lines.append(to_dashed_text(item, indent + 1))
            else:
                lines.append(f"{space}- {item}")
        return "\n".join(lines)

    return f"{space}- {value}"


def _strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        return "\n".join(lines).strip()
    return text


def _extract_balanced(text: str, opening: str, closing: str, start: int) -> str | None:
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_text(raw: str, expect_type: type) -> str:
    text = _strip_code_fences(raw.strip())

    preferred = [("{", "}"), ("[", "]")] if expect_type is dict else [("[", "]"), ("{", "}")]
    for opening, closing in preferred:
        pos = text.find(opening)
        while pos != -1:
            candidate = _extract_balanced(text, opening, closing, pos)
            if candidate is not None:
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, expect_type):
                        return candidate
                except json.JSONDecodeError:
                    pass
            pos = text.find(opening, pos + 1)

    for opening, closing in [("{", "}"), ("[", "]")]:
        pos = text.find(opening)
        if pos != -1:
            candidate = _extract_balanced(text, opening, closing, pos)
            if candidate is not None:
                return candidate
    return text


def generate_json(prompt: str, model: str, temperature: float, expect_type: type) -> Any:
    raw = call_llm(
        system_prompt="Return only valid JSON.",
        user_prompt=prompt,
        model=model,
        temperature=temperature,
    )
    candidate = extract_json_text(raw, expect_type=expect_type)
    parsed = json.loads(candidate)

    if expect_type is dict and isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        parsed = parsed[0]
    if expect_type is list and isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, expect_type):
        preview = candidate[:300].replace("\n", "\\n")
        raise ValueError(
            f"Expected {expect_type.__name__}, got {type(parsed).__name__}. JSON preview: {preview}"
        )

    return parsed


def generate_text(prompt: str, model: str, temperature: float) -> str:
    raw = call_llm(
        system_prompt="Return only the requested paragraph text.",
        user_prompt=prompt,
        model=model,
        temperature=temperature,
    )
    return _strip_code_fences(raw.strip())


def format_previous_dialogues(previous_phase_dialogues: list[dict[str, Any]]) -> str:
    if not previous_phase_dialogues:
        return "None"

    lines: list[str] = []
    for phase_block in previous_phase_dialogues:
        phase = str(phase_block.get("phase", "")).strip()
        dialogue = phase_block.get("dialogue", [])
        if phase:
            lines.append(f"[Phase: {phase}]")
        if isinstance(dialogue, list):
            for turn in dialogue:
                if not isinstance(turn, dict):
                    continue
                therapist = str(turn.get("therapist_utterance", "")).strip()
                client = str(turn.get("client_utterance", "")).strip()
                if therapist:
                    lines.append(f"Therapist: {therapist}")
                if client:
                    lines.append(f"Client: {client}")

    return "\n".join(lines) if lines else "None"


def parse_dialogue_history(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []

    payload = load_json(path)

    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        blocks = []
        for item in payload["results"]:
            if not isinstance(item, dict):
                continue
            blocks.append(
                {
                    "phase": item.get("phase", ""),
                    "dialogue": item.get("dialogue", []),
                }
            )
        return blocks

    if isinstance(payload, list):
        blocks = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            blocks.append(
                {
                    "phase": item.get("phase", ""),
                    "dialogue": item.get("dialogue", []),
                }
            )
        return blocks

    raise ValueError("Unsupported history format. Use output JSON with `results` or list of phase blocks.")


def normalize_phase(raw: str) -> str:
    clean = raw.strip().lower().replace("_", "").replace("-", "")
    if clean == "precontemplation":
        return "precontemplation"
    if clean == "contemplation":
        return "contemplation"
    if clean == "preparation":
        return "preparation"
    raise ValueError(f"Unsupported phase: {raw}")


def phase_index_or_none(raw: str) -> int | None:
    try:
        phase = normalize_phase(raw)
    except ValueError:
        return None
    return PHASES.index(phase)


def parse_phases(value: str) -> list[str]:
    tokens = [t.strip() for t in value.split(",") if t.strip()]
    if not tokens:
        raise ValueError("At least one phase must be provided.")

    phases = [normalize_phase(t) for t in tokens]
    for phase in phases:
        if phase not in PHASES:
            raise ValueError(f"Invalid phase order member: {phase}")

    phase_indexes = [PHASES.index(p) for p in phases]
    if phase_indexes != sorted(phase_indexes):
        raise ValueError("Phases must be sequentially ordered based on precontemplation -> contemplation -> preparation.")

    return phases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate therapist/client plans and dialogue-plan narrative by phase."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "Patient_PSi_CM_Dataset_Planning_Resistance.json",
    )
    parser.add_argument("--case-id", type=str, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--phases",
        type=str,
        default="precontemplation",
        help="Comma-separated phases in order. Example: precontemplation,contemplation",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="Optional JSON file with previous phase dialogues for continuity in later phases.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "phase_dialogue_plans.json",
    )
    args = parser.parse_args()

    selected_phases = parse_phases(args.phases)
    history_blocks = parse_dialogue_history(args.history)

    cases = load_cases(args.data)
    case = get_case(cases, args.case_id, args.index)
    values = build_template_values(case)

    dialogue_plan_template = load_text(ROOT / "prompts" / "dialogue_plan.txt")

    results: list[dict[str, Any]] = []
    for phase in selected_phases:
        therapist_tpl = load_text(ROOT / "prompts" / "therapist_plans" / f"{phase}_plan.txt")
        client_tpl = load_text(ROOT / "prompts" / "client_plans" / f"{phase}_plan.txt")

        therapist_prompt = safe_format(therapist_tpl, values)
        client_prompt = safe_format(client_tpl, values)

        therapist_plan = generate_json(
            prompt=therapist_prompt,
            model=args.model,
            temperature=args.temperature,
            expect_type=dict,
        )
        client_plan = generate_json(
            prompt=client_prompt,
            model=args.model,
            temperature=args.temperature,
            expect_type=dict,
        )

        current_phase_idx = PHASES.index(phase)
        prior_blocks = []
        for block in history_blocks:
            if not isinstance(block, dict):
                continue
            block_phase_idx = phase_index_or_none(str(block.get("phase", "")))
            if block_phase_idx is not None and block_phase_idx < current_phase_idx:
                prior_blocks.append(block)
        previous_dialogue = format_previous_dialogues(prior_blocks)

        plan_values = {
            "name": values.get("name", ""),
            "history": values.get("history", ""),
            "situation": values.get("situation", ""),
            "therapist_plan_json": to_dashed_text(therapist_plan),
            "client_plan_json": to_dashed_text(client_plan),
            "previous_dialogue": previous_dialogue,
        }

        dialogue_plan_prompt = safe_format(dialogue_plan_template, plan_values)
        dialogue_plan = generate_text(
            prompt=dialogue_plan_prompt,
            model=args.model,
            temperature=args.temperature,
        )

        results.append(
            {
                "phase": phase,
                "therapist_plan": therapist_plan,
                "client_plan": client_plan,
                "dialogue_plan": dialogue_plan,
                "dialogue_history_used": previous_dialogue,
            }
        )

    payload = {
        "case_id": case.get("id"),
        "name": case.get("name"),
        "phase_order": selected_phases,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(results)} dialogue plans for case_id={case.get('id')} -> {args.output}")


if __name__ == "__main__":
    main()
