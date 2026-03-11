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


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array.")
    return data


def format_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value)


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
                if client:
                    lines.append(f"Client: {client}")
                if therapist:
                    lines.append(f"Therapist: {therapist}")
    return "\n".join(lines) if lines else "None"


def build_core_beliefs(case: dict[str, Any]) -> str:
    parts = []
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
    values = {
        "name": format_field(case.get("name")),
        "history": format_field(case.get("history")),
        "core_beliefs": build_core_beliefs(case),
        "intermediate_beliefs": format_field(case.get("intermediate_belief")),
        "coping_strategies": format_field(case.get("coping_strategies")),
        "situation": format_field(case.get("situation")),
        "auto_thought": format_field(case.get("auto_thought")),
        "emotion": format_field(case.get("emotion")),
        "behavior": format_field(case.get("behavior")),
        "cbt_technique": format_field(case.get("cbt_technique")),
        "counseling_plan": format_field(case.get("counseling_plan")),
        "resistance_type": format_field(case.get("resistance_type")),
        "resistance_emotion": format_field(resistance_emotion),
        "resistance_emotions": format_field(resistance_emotion),
        "resistance_monologue": format_field(case.get("resistance_internal_monologue")),
        "resistance_level": format_field(case.get("resistance_level")),
    }
    return values


PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def safe_format(template: str, values: dict[str, str]) -> str:
    # Replace only simple {placeholder_name} tokens and keep all other braces
    # untouched (e.g., JSON examples in prompt text).
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        return values.get(key, "")

    return PLACEHOLDER_PATTERN.sub(repl, template)


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

    # Last resort: first balanced structure of either type.
    for opening, closing in [("{", "}"), ("[", "]")]:
        pos = text.find(opening)
        if pos != -1:
            candidate = _extract_balanced(text, opening, closing, pos)
            if candidate is not None:
                return candidate
    return text


def generate_json(
    prompt: str, model: str, temperature: float, expect_type: type
) -> Any:
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
            f"Expected {expect_type.__name__}, got {type(parsed).__name__}. "
            f"JSON preview: {preview}"
        )
    return parsed


def get_case(cases: list[dict[str, Any]], case_id: str | None, index: int | None) -> dict[str, Any]:
    if case_id:
        for case in cases:
            if str(case.get("id")) == case_id:
                return case
        raise ValueError(f"Case id '{case_id}' not found.")
    if index is None:
        index = 0
    if index < 0 or index >= len(cases):
        raise ValueError(f"Index {index} out of range for {len(cases)} cases.")
    return cases[index]


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def build_default_output_path(case_id: str, resistance_level: str) -> Path:
    safe_level = _slugify(resistance_level) or "unknown"
    safe_case_id = case_id.replace("-", "_")
    return ROOT / "outputs" / "dialogues" / f"dialogues_{safe_level}_{safe_case_id}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate therapist/client plans and dialogue sequentially by phase."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "Patient_PSi_CM_Dataset_Planning_Resistance.json",
    )
    parser.add_argument("--case-id", type=str, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    cases = load_cases(args.data)
    case = get_case(cases, args.case_id, args.index)
    case_id = str(case.get("id"))
    resistance_level = str(case.get("resistance_level", "unknown"))
    output_path = (
        args.output
        if args.output is not None
        else build_default_output_path(case_id, resistance_level)
    )
    values = build_template_values(case)

    dialogue_template = load_text(ROOT / "prompts" / "dialogue_system.txt")
    previous_phase_dialogues: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for phase in PHASES:
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

        # Keep dialogue input minimal; situation was removed from dialogue_system.txt.
        dialogue_values = {
            "name": values.get("name", ""),
            "history": values.get("history", ""),
            "cbt_technique": values.get("cbt_technique", ""),
            "counseling_plan": values.get("counseling_plan", ""),
        }
        dialogue_values["therapist_plan_json"] = to_dashed_text(therapist_plan)
        dialogue_values["client_plan_json"] = to_dashed_text(client_plan)
        previous_dialogues_text = format_previous_dialogues(previous_phase_dialogues)
        dialogue_values["previous_dialogues"] = previous_dialogues_text
        # Backward compatibility if prompt uses old placeholder name.
        dialogue_values["previous_phase_dialogues"] = previous_dialogues_text
        dialogue_prompt = safe_format(dialogue_template, dialogue_values)

        dialogue = generate_json(
            prompt=dialogue_prompt,
            model=args.model,
            temperature=args.temperature,
            expect_type=list,
        )

        phase_result = {
            "phase": phase,
            "therapist_plan": therapist_plan,
            "client_plan": client_plan,
            "dialogue": dialogue,
        }
        results.append(phase_result)
        previous_phase_dialogues.append({"phase": phase, "dialogue": dialogue})

    payload = {
        "case_id": case_id,
        "name": case.get("name"),
        "phase_order": PHASES,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(results)} phases for case_id={case_id} -> {output_path}")


if __name__ == "__main__":
    main()
