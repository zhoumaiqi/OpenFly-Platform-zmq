import json
import re

import requests


STAGE_SPLIT_RE = re.compile(
    r"\b("
    r"then|finally|next|after that|afterward|afterwards|lastly|last|at last|in the end|"
    r"head straight|go straight|proceed straight|slightly turn left|slightly turn right|"
    r"turn left|turn right|ascend|descend|drop near|stop near"
    r")\b",
    flags=re.IGNORECASE,
)


def parse_stage_plan(
    instruction,
    base_url="",
    model_name="",
    parser_type="rule",
    timeout=180,
    max_tokens=900,
):
    if parser_type == "rule":
        stages = extract_stage_plan_rule(instruction)
        meta = {
            "parser_source": "rule",
            "parser_error": "",
            "parser_raw_response": "",
            "parser_timeout_sec": 0,
            "parser_max_tokens": 0,
        }
        print(f"[STAGE_PARSER] source=rule stages={len(stages)}")
        return stages, meta
    if parser_type != "qwen":
        raise ValueError(f"unsupported stage parser: {parser_type}")

    try:
        stages, raw_response = _parse_with_qwen(
            instruction,
            base_url,
            model_name,
            timeout,
            max_tokens,
        )
        if _is_multi_sentence(instruction) and len(stages) <= 1:
            print("[STAGE_PARSER] warning=qwen_single_stage_for_multi_sentence_instruction")
            fallback = extract_stage_plan_rule(instruction)
            meta = {
                "parser_source": "rule_fallback",
                "parser_error": "qwen_single_stage_for_multi_sentence_instruction",
                "parser_raw_response": raw_response[:4000],
                "parser_timeout_sec": timeout,
                "parser_max_tokens": max_tokens,
            }
            print("[STAGE_PARSER] source=rule_fallback reason=qwen_single_stage_for_multi_sentence_instruction")
            return fallback, meta
        meta = {
            "parser_source": "qwen",
            "parser_error": "",
            "parser_raw_response": raw_response[:4000],
            "parser_timeout_sec": timeout,
            "parser_max_tokens": max_tokens,
        }
        print(f"[STAGE_PARSER] source=qwen stages={len(stages)}")
        return stages, meta
    except Exception as exc:
        fallback = extract_stage_plan_rule(instruction)
        meta = {
            "parser_source": "rule_fallback",
            "parser_error": f"{type(exc).__name__}: {exc}",
            "parser_raw_response": "",
            "parser_timeout_sec": timeout,
            "parser_max_tokens": max_tokens,
        }
        print(f"[STAGE_PARSER] source=rule_fallback reason={meta['parser_error']}")
        return fallback, meta


def extract_stage_plan_rule(instruction):
    parts = []
    for segment in _sentence_segments(instruction):
        segment_parts = _split_segment_by_markers(segment)
        parts.extend(segment_parts)

    if not parts:
        parts = [instruction.strip().strip('"')]

    stages = []
    previous_landmark = ""
    for idx, stage_text in enumerate(parts):
        lower = stage_text.lower()
        motion_hints = _motion_hints(lower)
        landmark_phrase = _extract_landmark(stage_text)
        inherits_from = None
        is_motion_only = _is_pronoun_landmark(landmark_phrase)
        resolved_landmark = landmark_phrase
        if is_motion_only and previous_landmark:
            resolved_landmark = previous_landmark
            inherits_from = idx - 1
        elif landmark_phrase:
            previous_landmark = landmark_phrase

        stages.append(
            {
                "stage_id": idx,
                "stage_text": stage_text,
                "motion_hint": ", ".join(motion_hints),
                "landmark_phrase": resolved_landmark or landmark_phrase or stage_text,
                "resolved_landmark_phrase": resolved_landmark or landmark_phrase or stage_text,
                "parser_landmark_phrase": landmark_phrase,
                "is_motion_only": is_motion_only,
                "inherits_landmark_from": inherits_from,
                "status": "pending",
            }
        )
    return stages


def _parse_with_qwen(instruction, base_url, model_name, timeout, max_tokens):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON parser for VLN navigation instructions. "
                    "Return only valid JSON. No markdown. No explanation."
                ),
            },
            {
                "role": "user",
                "content": _stage_parser_prompt(instruction),
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    raw_response = data["choices"][0]["message"]["content"]
    parsed = json.loads(_strip_json_fence(raw_response))
    stages = parsed.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Qwen parser returned empty or invalid stages")
    return _normalize_qwen_stages(stages), raw_response


def _stage_parser_prompt(instruction):
    return (
        "Split this VLN instruction into semantic navigation stages. "
        "Use sentence boundaries, direction changes, and action phrases such as Ascend, Descend, "
        "Head straight, Proceed straight, Slightly turn left/right, Turn left/right, Lastly, Stop, Drop near. "
        "Resolve it/this/that/the building/the tower to the nearest explicit landmark. "
        "Motion-only stages inherit the previous landmark. "
        "Return JSON only, no markdown, no explanation. Schema: "
        "{\"stages\":[{\"stage_id\":0,\"stage_text\":\"...\",\"motion_hint\":\"...\","
        "\"landmark_phrase\":\"...\",\"resolved_landmark_phrase\":\"...\","
        "\"is_motion_only\":false,\"inherits_landmark_from\":null}]}.\n"
        f"Instruction:\n{instruction}"
    )


def _normalize_qwen_stages(raw_stages):
    stages = []
    for idx, raw_stage in enumerate(raw_stages):
        if not isinstance(raw_stage, dict):
            raise ValueError("stage item is not an object")
        stage_text = str(raw_stage.get("stage_text", "")).strip()
        if not stage_text:
            raise ValueError("stage_text is empty")
        parser_landmark = str(raw_stage.get("landmark_phrase", "")).strip()
        resolved = str(raw_stage.get("resolved_landmark_phrase", "")).strip()
        landmark_for_downstream = resolved or parser_landmark or stage_text
        inherits = raw_stage.get("inherits_landmark_from", None)
        if inherits == "":
            inherits = None
        stages.append(
            {
                "stage_id": idx,
                "stage_text": stage_text,
                "motion_hint": str(raw_stage.get("motion_hint", "")).strip(),
                "landmark_phrase": landmark_for_downstream,
                "resolved_landmark_phrase": landmark_for_downstream,
                "parser_landmark_phrase": parser_landmark,
                "is_motion_only": bool(raw_stage.get("is_motion_only", False)),
                "inherits_landmark_from": inherits,
                "status": "pending",
            }
        )
    return stages


def _sentence_segments(instruction):
    cleaned = instruction.strip().strip('"').strip()
    return [
        part.strip(" ,.;")
        for part in re.split(r"\s*[.;]\s*", cleaned)
        if part.strip(" ,.;")
    ]


def _split_segment_by_markers(segment):
    matches = list(STAGE_SPLIT_RE.finditer(segment))
    if not matches:
        return [segment.strip(" ,.;")]
    if matches[0].start() == 0:
        return [segment.strip(" ,.;")]
    parts = []
    first = segment[:matches[0].start()].strip(" ,.;")
    if first:
        parts.append(first)
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(segment)
        part = segment[match.start():end].strip(" ,.;")
        if part:
            parts.append(part)
    return parts


def _motion_hints(lower):
    hints = []
    for word in (
        "ascend",
        "descend",
        "straight",
        "left",
        "right",
        "forward",
        "turn",
        "stop",
        "drop",
        "near",
    ):
        if word in lower:
            hints.append(word)
    return hints


def _extract_landmark(stage_text):
    lower = stage_text.lower()
    for token in ("towards", "toward", "to", "until", "near"):
        marker = f" {token} "
        if marker in lower:
            start = lower.find(marker) + len(marker)
            return stage_text[start:].strip(" ,.;")
    return ""


def _is_pronoun_landmark(landmark):
    normalized = landmark.strip().lower()
    return normalized == "" or normalized in {
        "it",
        "this",
        "that",
        "the building",
        "the tower",
    }


def _is_multi_sentence(instruction):
    return len(_sentence_segments(instruction)) > 1


def _strip_json_fence(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()
