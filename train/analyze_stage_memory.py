import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze recorded stage memory for one sample.")
    parser.add_argument("--graph_dir", required=True)
    return parser.parse_args()


def load_json(path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def to_float(value):
    try:
        if value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def to_int(value, default=-1):
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def reliable(node):
    return not to_bool(node.get("fallback_used", "")) and to_float(node.get("confidence", "")) >= 0.8


def longest_true_run(values):
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def bool_series(nodes, field):
    return [to_bool(node.get(field, "")) for node in nodes]


def true_node_refs(nodes, field):
    refs = []
    for node in nodes:
        if not to_bool(node.get(field, "")):
            continue
        refs.append(
            {
                "node_id": to_int(node.get("node_id", "")),
                "step_id": to_int(node.get("step_id", "")),
                "active_stage": to_int(node.get("active_stage", "")),
                "observed_stage_id": to_int(node.get("observed_stage_id", "")),
                "confidence": to_float(node.get("confidence", "")),
                "progress": node.get("stage_progress", ""),
                "stage_memory_note": node.get("stage_memory_note", ""),
                "weak_stage_completion_reason": node.get("weak_stage_completion_reason", ""),
                "tracker_completion_reason": node.get("tracker_completion_reason", ""),
                "long_approach_reason": node.get("long_approach_reason", ""),
                "long_approach_score": to_int(node.get("long_approach_score", ""), default=0),
                "long_approach_visible_count": to_int(node.get("long_approach_visible_count", ""), default=0),
                "long_approach_centered_count": to_int(node.get("long_approach_centered_count", ""), default=0),
                "long_approach_weak_count": to_int(node.get("long_approach_weak_count", ""), default=0),
                "long_approach_step_span": to_int(node.get("long_approach_step_span", ""), default=0),
                "long_approach_confirmed_reason": node.get("long_approach_confirmed_reason", ""),
                "long_approach_confirmed_count": to_int(node.get("long_approach_confirmed_count", ""), default=0),
                "consecutive_tracker_completion_count": to_int(
                    node.get("consecutive_tracker_completion_count", ""),
                    default=0,
                ),
                "consecutive_long_approach_confirmed_count": to_int(
                    node.get("consecutive_long_approach_confirmed_count", ""),
                    default=0,
                ),
                "fallback_used": to_bool(node.get("fallback_used", "")),
            }
        )
    return refs


def analyze(graph_dir):
    stage_plan = load_json(graph_dir / "stage_plan.json", [])
    stage_memory = load_json(graph_dir / "stage_memory.json", {})
    nodes = load_csv(graph_dir / "memory_nodes.csv")
    graph_json = load_json(graph_dir / "memory_graph.json", {})
    if not nodes and graph_json:
        nodes = graph_json.get("nodes", [])

    transitions = stage_memory.get("transitions", [])
    observed_ids = [to_int(node.get("observed_stage_id", "")) for node in nodes]
    observed_distribution = Counter(str(stage_id) for stage_id in observed_ids)
    observed_jumps = 0
    for prev, cur in zip(observed_ids, observed_ids[1:]):
        if prev != cur:
            observed_jumps += 1

    reliable_nodes = [node for node in nodes if reliable(node)]
    active_visible_values = [
        to_bool(node.get("active_stage_target_visible", node.get("stage_target_visible", "")))
        for node in nodes
    ]
    centered_values = bool_series(nodes, "active_stage_target_centered")
    close_values = bool_series(nodes, "active_stage_target_close")
    passed_values = bool_series(nodes, "active_stage_target_passed")
    weak_completion_values = bool_series(nodes, "weak_stage_completion_candidate")
    tracker_completion_values = bool_series(nodes, "stage_completion_candidate_by_tracker")
    transition_candidate_values = bool_series(nodes, "stage_transition_candidate")
    close_memory_values = bool_series(nodes, "close_memory_active")
    long_approach_values = bool_series(nodes, "long_approach_candidate")
    long_approach_confirmed_values = bool_series(nodes, "long_approach_confirmed")
    reliable_visible_by_stage = defaultdict(list)
    for node in reliable_nodes:
        active_stage = to_int(node.get("active_stage", ""))
        if to_bool(node.get("active_stage_target_visible", node.get("stage_target_visible", ""))):
            reliable_visible_by_stage[str(active_stage)].append(
                {
                    "node_id": to_int(node.get("node_id", "")),
                    "step_id": to_int(node.get("step_id", "")),
                    "choice": node.get("choice", ""),
                    "progress": node.get("stage_progress", ""),
                    "confidence": to_float(node.get("confidence", "")),
                }
            )

    complete_nodes = [
        {
            "node_id": to_int(node.get("node_id", "")),
            "step_id": to_int(node.get("step_id", "")),
            "active_stage": to_int(node.get("active_stage", "")),
            "observed_stage_id": to_int(node.get("observed_stage_id", "")),
            "confidence": to_float(node.get("confidence", "")),
            "progress": node.get("stage_progress", ""),
            "fallback_used": to_bool(node.get("fallback_used", "")),
        }
        for node in nodes
        if to_bool(node.get("stage_complete_candidate", ""))
    ]

    fallback_nodes = [node for node in nodes if to_bool(node.get("fallback_used", ""))]
    transition_from_fallback = []
    fallback_step_ids = {to_int(node.get("step_id", "")) for node in fallback_nodes}
    for transition in transitions:
        if to_int(transition.get("step_id", "")) in fallback_step_ids:
            transition_from_fallback.append(transition)

    conflicts = []
    for node in reliable_nodes:
        active_stage = to_int(node.get("active_stage", ""))
        observed_stage_id = to_int(node.get("observed_stage_id", ""))
        if observed_stage_id >= 0 and observed_stage_id != active_stage:
            conflicts.append(
                {
                    "node_id": to_int(node.get("node_id", "")),
                    "step_id": to_int(node.get("step_id", "")),
                    "active_stage": active_stage,
                    "observed_stage_id": observed_stage_id,
                    "choice": node.get("choice", ""),
                    "progress": node.get("stage_progress", ""),
                    "confidence": to_float(node.get("confidence", "")),
                }
            )

    reliable_stage_evidence = defaultdict(list)
    for node in reliable_nodes:
        active_stage = str(to_int(node.get("active_stage", "")))
        reliable_stage_evidence[active_stage].append(
            {
                "node_id": to_int(node.get("node_id", "")),
                "step_id": to_int(node.get("step_id", "")),
                "active_stage": to_int(node.get("active_stage", "")),
                "observed_stage_id": to_int(node.get("observed_stage_id", "")),
                "choice": node.get("choice", ""),
                "active_stage_target_visible": to_bool(
                    node.get("active_stage_target_visible", node.get("stage_target_visible", ""))
                ),
                "active_stage_target_centered": to_bool(node.get("active_stage_target_centered", "")),
                "active_stage_target_close": to_bool(node.get("active_stage_target_close", "")),
                "active_stage_target_passed": to_bool(node.get("active_stage_target_passed", "")),
                "stage_progress": node.get("stage_progress", ""),
                "stage_complete_candidate": to_bool(node.get("stage_complete_candidate", "")),
                "weak_stage_completion_candidate": to_bool(node.get("weak_stage_completion_candidate", "")),
                "weak_stage_completion_reason": node.get("weak_stage_completion_reason", ""),
                "stage_completion_candidate_by_tracker": to_bool(node.get("stage_completion_candidate_by_tracker", "")),
                "tracker_completion_reason": node.get("tracker_completion_reason", ""),
                "stage_transition_candidate": to_bool(node.get("stage_transition_candidate", "")),
                "long_approach_candidate": to_bool(node.get("long_approach_candidate", "")),
                "long_approach_confirmed": to_bool(node.get("long_approach_confirmed", "")),
                "long_approach_confirmed_reason": node.get("long_approach_confirmed_reason", ""),
                "reason": node.get("reason", ""),
            }
        )

    weak_nodes = true_node_refs(nodes, "weak_stage_completion_candidate")
    strong_nodes = true_node_refs(nodes, "stage_completion_candidate_by_tracker")
    weak_centered_count = sum(
        1 for node in weak_nodes if node.get("weak_stage_completion_reason") == "centered_visible_count"
    )
    strong_reason_distribution = Counter(
        node.get("tracker_completion_reason", "") for node in strong_nodes
    )
    close_memory_source_distribution = Counter(
        node.get("close_memory_source", "") for node in nodes if to_bool(node.get("close_memory_active", ""))
    )
    smoothed_nodes = [
        node for node in strong_nodes
        if node.get("tracker_completion_reason") == "smoothed_close_memory"
    ]
    smoothed_fallback_nodes = [
        node for node in smoothed_nodes
        if node.get("fallback_used")
    ]
    close_memory_expired_nodes = [
        {
            "node_id": to_int(node.get("node_id", "")),
            "step_id": to_int(node.get("step_id", "")),
            "stage_memory_note": node.get("stage_memory_note", ""),
        }
        for node in nodes
        if "close_memory_expired" in node.get("stage_memory_note", "")
    ]
    close_memory_cleared_nodes = [
        {
            "node_id": to_int(node.get("node_id", "")),
            "step_id": to_int(node.get("step_id", "")),
            "stage_memory_note": node.get("stage_memory_note", ""),
        }
        for node in nodes
        if "close_memory_cleared" in node.get("stage_memory_note", "")
    ]
    fallback_preserved_close_memory = [
        {
            "node_id": to_int(node.get("node_id", "")),
            "step_id": to_int(node.get("step_id", "")),
            "close_memory_age": to_int(node.get("close_memory_age", ""), default=0),
            "close_memory_source": node.get("close_memory_source", ""),
        }
        for node in nodes
        if to_bool(node.get("fallback_used", "")) and to_bool(node.get("close_memory_active", ""))
    ]
    long_approach_nodes = true_node_refs(nodes, "long_approach_candidate")
    long_approach_confirmed_nodes = true_node_refs(nodes, "long_approach_confirmed")
    long_approach_reason_distribution = Counter(
        node.get("long_approach_reason", "") for node in long_approach_nodes
    )
    long_approach_confirmed_reason_distribution = Counter(
        node.get("long_approach_confirmed_reason", "") for node in long_approach_confirmed_nodes
    )
    long_approach_max_score = max(
        [node.get("long_approach_score", 0) for node in long_approach_nodes] or [0]
    )
    long_approach_confirmed_strong_nodes = [
        node for node in long_approach_confirmed_nodes
        if node.get("tracker_completion_reason") == "long_approach_confirmed"
    ]
    long_approach_confirmed_steps = [
        node.get("step_id") for node in long_approach_confirmed_nodes
    ]
    long_approach_confirmed_strong_steps = [
        node.get("step_id") for node in long_approach_confirmed_strong_nodes
    ]
    long_approach_confirmed_fallback_nodes = [
        node for node in long_approach_confirmed_nodes
        if node.get("fallback_used")
    ]
    long_approach_transition_details = []
    for transition in transitions:
        transition_reason = transition.get("tracker_completion_reason") or transition.get("reason", "")
        if transition_reason != "long_approach_confirmed":
            continue
        step_id = to_int(transition.get("step_id", ""))
        transition_node = next(
            (node for node in long_approach_confirmed_nodes if node.get("step_id") == step_id),
            {},
        )
        long_approach_transition_details.append(
            {
                "from_stage": to_int(transition.get("from_stage", "")),
                "to_stage": to_int(transition.get("to_stage", "")),
                "step": step_id,
                "reason": transition_reason,
                "long_approach_score": to_int(
                    transition.get("long_approach_score", transition_node.get("long_approach_score", "")),
                    default=transition_node.get("long_approach_score", 0),
                ),
                "visible_count": to_int(
                    transition.get("visible_count", transition_node.get("long_approach_visible_count", "")),
                    default=transition_node.get("long_approach_visible_count", 0),
                ),
                "centered_count": to_int(
                    transition.get("centered_count", transition_node.get("long_approach_centered_count", "")),
                    default=transition_node.get("long_approach_centered_count", 0),
                ),
                "weak_count": to_int(
                    transition.get("weak_count", transition_node.get("long_approach_weak_count", "")),
                    default=transition_node.get("long_approach_weak_count", 0),
                ),
                "step_span": to_int(
                    transition.get("step_span", transition_node.get("long_approach_step_span", "")),
                    default=transition_node.get("long_approach_step_span", 0),
                ),
                "consecutive_tracker_completion_count": to_int(
                    transition.get(
                        "consecutive_tracker_completion_count",
                        transition_node.get("consecutive_tracker_completion_count", ""),
                    ),
                    default=transition_node.get("consecutive_tracker_completion_count", 0),
                ),
            }
        )
    stage_stats = stage_memory.get("stage_stats", {})
    long_approach_count_by_stage = {}
    for stage_id, stats in stage_stats.items():
        long_approach_count_by_stage[str(stage_id)] = {
            "candidate_count": stats.get("long_approach_candidate_count", 0),
            "confirmed_count": stats.get("long_approach_confirmed_count", 0),
            "consecutive_confirmed_count": stats.get("consecutive_long_approach_confirmed_count", 0),
            "tracker_completion_candidate_count": stats.get("tracker_completion_candidate_count", 0),
            "consecutive_tracker_completion_count": stats.get("consecutive_tracker_completion_count", 0),
        }
    long_approach_strong_without_transition = bool(
        long_approach_confirmed_strong_nodes and not long_approach_transition_details
    )
    weak_transition_warnings = [
        transition
        for transition in transitions
        if transition.get("tracker_completion_reason") == "centered_visible_count"
        or transition.get("reason") == "centered_visible_count"
    ]
    no_transition_reasons = []
    if len(transitions) == 0:
        if any(weak_completion_values) and not any(tracker_completion_values):
            no_transition_reasons.append("only weak centered_visible evidence observed")
        if any(long_approach_values) and not any(long_approach_confirmed_values):
            no_transition_reasons.append("long approach detected but not confirmed")
        if any(long_approach_confirmed_values) and longest_true_run(long_approach_confirmed_values) < 2:
            no_transition_reasons.append("long approach confirmed but not consecutive")
        if long_approach_strong_without_transition:
            no_transition_reasons.append("long approach confirmed strong candidates but no transition occurred")
        if any(
            to_int(node.get("active_stage", "")) >= len(stage_plan) - 1
            for node in long_approach_confirmed_nodes
        ):
            no_transition_reasons.append("long approach confirmed but active_stage is last stage")
        if long_approach_confirmed_fallback_nodes:
            no_transition_reasons.append("long approach confirmed but transition blocked by fallback")
        if not any(long_approach_values) and not any(close_values) and not any(passed_values) and not any(close_memory_values):
            no_transition_reasons.append("no long approach and no close/near/passed")
        if sum(close_values) == 1 and not smoothed_nodes:
            no_transition_reasons.append("close appeared only once but was not smoothed")
        if close_memory_expired_nodes:
            no_transition_reasons.append("close memory expired before reliable confirmation")
        if close_memory_cleared_nodes:
            no_transition_reasons.append("close memory was interrupted by lost/mismatch")
        if not any(close_values):
            no_transition_reasons.append("close never triggered")
        if longest_true_run(centered_values) < 3 or longest_true_run(active_visible_values) < 3:
            no_transition_reasons.append("centered not stable enough")
        if not any(passed_values):
            no_transition_reasons.append("passed never triggered")
        fallback_rate = len(fallback_nodes) / len(nodes) if nodes else 0.0
        if fallback_rate >= 0.2:
            no_transition_reasons.append("fallback rate high")
        if not any(bool_series(nodes, "stage_complete_candidate")):
            no_transition_reasons.append("vlm_complete never triggered")
        if not any(tracker_completion_values):
            no_transition_reasons.append("strong completion candidate never triggered")

    analysis = {
        "graph_dir": str(graph_dir),
        "stage_plan_check": {
            "stage_count": len(stage_plan),
            "stages": [
                {
                    "stage_id": stage.get("stage_id"),
                    "stage_text": stage.get("stage_text", ""),
                    "landmark_phrase": stage.get("landmark_phrase", ""),
                    "motion_hint": stage.get("motion_hint", ""),
                }
                for stage in stage_plan
            ],
        },
        "active_stage_check": {
            "final_active_stage": stage_memory.get("active_stage"),
            "transitions": transitions,
            "no_stage_transition_detected": len(transitions) == 0,
        },
        "observed_stage_check": {
            "distribution": dict(observed_distribution),
            "jump_count": observed_jumps,
            "observed_stage_unstable": observed_jumps >= max(2, len(nodes) // 5),
        },
        "active_stage_target_visible_check": {
            "true_count": sum(active_visible_values),
            "longest_true_run": longest_true_run(active_visible_values),
            "reliable_visible_by_stage": dict(reliable_visible_by_stage),
        },
        "active_stage_target_centered_check": {
            "true_count": sum(centered_values),
            "longest_true_run": longest_true_run(centered_values),
        },
        "active_stage_target_close_check": {
            "true_count": sum(close_values),
            "longest_true_run": longest_true_run(close_values),
        },
        "active_stage_target_passed_check": {
            "true_count": sum(passed_values),
            "longest_true_run": longest_true_run(passed_values),
            "nodes": true_node_refs(nodes, "active_stage_target_passed"),
        },
        "complete_candidate_check": {
            "true_count": len(complete_nodes),
            "complete_never_triggered": len(complete_nodes) == 0,
            "nodes": complete_nodes,
        },
        "tracker_completion_candidate_check": {
            "true_count": sum(tracker_completion_values),
            "reason_distribution": dict(strong_reason_distribution),
            "nodes": strong_nodes,
        },
        "close_memory_check": {
            "active_count": sum(close_memory_values),
            "source_distribution": dict(close_memory_source_distribution),
            "last_close_step_values": sorted(
                set(
                    to_int(node.get("last_close_step", ""), default=-1)
                    for node in nodes
                    if node.get("last_close_step", "") != ""
                )
            ),
            "last_near_step_values": sorted(
                set(
                    to_int(node.get("last_near_step", ""), default=-1)
                    for node in nodes
                    if node.get("last_near_step", "") != ""
                )
            ),
            "fallback_preserved_close_memory": fallback_preserved_close_memory,
            "expired_nodes": close_memory_expired_nodes,
            "cleared_nodes": close_memory_cleared_nodes,
        },
        "smoothed_close_memory_check": {
            "true_count": len(smoothed_nodes),
            "nodes": smoothed_nodes,
            "triggered_on_fallback_nodes": smoothed_fallback_nodes,
            "warning_triggered_on_fallback": bool(smoothed_fallback_nodes),
            "participates_as_strong_candidate": len(smoothed_nodes) > 0,
        },
        "long_approach_check": {
            "true_count": sum(long_approach_values),
            "nodes": long_approach_nodes,
            "max_score": long_approach_max_score,
            "reason_distribution": dict(long_approach_reason_distribution),
            "stable_visible_centered_without_close": bool(long_approach_nodes),
            "confirmed_true_count": sum(long_approach_confirmed_values),
            "confirmed_steps": long_approach_confirmed_steps,
            "confirmed_nodes": long_approach_confirmed_nodes,
            "confirmed_reason_distribution": dict(long_approach_confirmed_reason_distribution),
            "confirmed_became_strong_candidate_count": len(long_approach_confirmed_strong_nodes),
            "confirmed_became_strong_candidate": bool(long_approach_confirmed_strong_nodes),
            "confirmed_strong_steps": long_approach_confirmed_strong_steps,
            "confirmed_strong_nodes": long_approach_confirmed_strong_nodes,
            "confirmed_strong_consecutive_counts": [
                {
                    "step_id": node.get("step_id"),
                    "consecutive_tracker_completion_count": node.get("consecutive_tracker_completion_count", 0),
                    "consecutive_long_approach_confirmed_count": node.get(
                        "consecutive_long_approach_confirmed_count",
                        0,
                    ),
                }
                for node in long_approach_confirmed_strong_nodes
            ],
            "candidate_vs_confirmed_by_stage": long_approach_count_by_stage,
            "transition_caused_by_long_approach_confirmed": bool(long_approach_transition_details),
            "transition_details": long_approach_transition_details,
            "confirmed_on_fallback_nodes": long_approach_confirmed_fallback_nodes,
            "warning_strong_without_transition": long_approach_strong_without_transition,
        },
        "weak_completion_candidate_check": {
            "true_count": sum(weak_completion_values),
            "centered_visible_count": weak_centered_count,
            "nodes": weak_nodes,
        },
        "stage_transition_candidate_check": {
            "true_count": sum(transition_candidate_values),
            "nodes": true_node_refs(nodes, "stage_transition_candidate"),
            "transition_steps": transitions,
            "no_transition_reasons": no_transition_reasons,
            "warning_centered_visible_triggered_transition": bool(weak_transition_warnings),
            "weak_transition_warnings": weak_transition_warnings,
        },
        "fallback_impact_check": {
            "fallback_count": len(fallback_nodes),
            "fallback_rate": len(fallback_nodes) / len(nodes) if nodes else 0.0,
            "transition_from_fallback": transition_from_fallback,
            "warning_transition_from_fallback": bool(transition_from_fallback),
        },
        "observed_active_conflict_check": {
            "future_or_other_stage_attention_count": len(conflicts),
            "examples": conflicts[:10],
        },
        "reliable_stage_evidence": dict(reliable_stage_evidence),
        "stage_stats": stage_memory.get("stage_stats", {}),
    }
    return analysis


def build_summary(analysis):
    lines = []
    lines.append("Stage Memory Analysis")
    lines.append("=====================")
    lines.append("")
    lines.append(f"Graph dir: {analysis['graph_dir']}")
    lines.append("")
    lines.append("Stage plan:")
    for stage in analysis["stage_plan_check"]["stages"]:
        lines.append(
            f"- [{stage['stage_id']}] {stage['stage_text']} "
            f"(landmark={stage['landmark_phrase']}, motion={stage['motion_hint']})"
        )
    lines.append("")
    lines.append("Key findings:")
    if analysis["active_stage_check"]["no_stage_transition_detected"]:
        lines.append("- active_stage never changed; no stage transition was detected.")
    else:
        lines.append(f"- transitions: {analysis['active_stage_check']['transitions']}")
    if analysis["complete_candidate_check"]["complete_never_triggered"]:
        lines.append("- stage_complete_candidate was never true.")
    if analysis["tracker_completion_candidate_check"]["true_count"] == 0:
        lines.append("- tracker did not generate any strong completion candidate.")
    if analysis["weak_completion_candidate_check"]["true_count"] > 0:
        lines.append(
            "- weak completion candidates observed: "
            f"{analysis['weak_completion_candidate_check']['true_count']} "
            f"(centered_visible_count={analysis['weak_completion_candidate_check']['centered_visible_count']})"
        )
    lines.append(
        "- observed_stage_id distribution: "
        f"{analysis['observed_stage_check']['distribution']}"
    )
    if analysis["observed_stage_check"]["observed_stage_unstable"]:
        lines.append("- observed_stage_id is unstable and should not be directly used as active_stage.")
    lines.append(
        "- active_stage_target_visible true_count="
        f"{analysis['active_stage_target_visible_check']['true_count']}, "
        f"longest_run={analysis['active_stage_target_visible_check']['longest_true_run']}"
    )
    lines.append(
        "- centered true_count="
        f"{analysis['active_stage_target_centered_check']['true_count']}, "
        f"longest_run={analysis['active_stage_target_centered_check']['longest_true_run']}"
    )
    lines.append(
        "- close true_count="
        f"{analysis['active_stage_target_close_check']['true_count']}, "
        f"longest_run={analysis['active_stage_target_close_check']['longest_true_run']}"
    )
    lines.append(
        "- passed true_count="
        f"{analysis['active_stage_target_passed_check']['true_count']}"
    )
    if analysis["stage_transition_candidate_check"]["no_transition_reasons"]:
        lines.append(
            "- no transition reasons: "
            f"{analysis['stage_transition_candidate_check']['no_transition_reasons']}"
        )
    if analysis["long_approach_check"]["true_count"] > 0:
        lines.append(
            "- long_approach_candidate detected: "
            f"count={analysis['long_approach_check']['true_count']}, "
            f"max_score={analysis['long_approach_check']['max_score']}, "
            f"reasons={analysis['long_approach_check']['reason_distribution']}"
        )
    if analysis["long_approach_check"]["confirmed_true_count"] > 0:
        lines.append(
            "- long_approach_confirmed detected: "
            f"count={analysis['long_approach_check']['confirmed_true_count']}, "
            f"strong_count={analysis['long_approach_check']['confirmed_became_strong_candidate_count']}, "
            f"reasons={analysis['long_approach_check']['confirmed_reason_distribution']}"
        )
        lines.append(
            "- long_approach_confirmed steps: "
            f"{analysis['long_approach_check']['confirmed_steps']}; "
            "strong steps: "
            f"{analysis['long_approach_check']['confirmed_strong_steps']}"
        )
        lines.append(
            "- long_approach confirmed strong consecutive counts: "
            f"{analysis['long_approach_check']['confirmed_strong_consecutive_counts']}"
        )
    if analysis["long_approach_check"]["warning_strong_without_transition"]:
        lines.append(
            "- WARNING: long_approach_confirmed produced strong candidates, "
            "but no stage transition occurred. Check transition execution logic "
            "or consecutive strong candidate counter."
        )
        lines.append(
            "- long_approach tracker counts by stage: "
            f"{analysis['long_approach_check']['candidate_vs_confirmed_by_stage']}"
        )
    if analysis["long_approach_check"]["transition_caused_by_long_approach_confirmed"]:
        lines.append(
            "- long_approach_confirmed caused transition(s): "
            f"{analysis['long_approach_check']['transition_details']}"
        )
    lines.append(
        "- close_memory active_count="
        f"{analysis['close_memory_check']['active_count']}, "
        f"sources={analysis['close_memory_check']['source_distribution']}"
    )
    if analysis["close_memory_check"]["fallback_preserved_close_memory"]:
        lines.append("- close memory was preserved through fallback nodes.")
    if analysis["close_memory_check"]["expired_nodes"]:
        lines.append("- close memory expired before confirmation.")
    if analysis["close_memory_check"]["cleared_nodes"]:
        lines.append("- close memory was cleared by lost/mismatch evidence.")
    lines.append(
        "- smoothed_close_memory strong candidates="
        f"{analysis['smoothed_close_memory_check']['true_count']}"
    )
    if analysis["smoothed_close_memory_check"]["warning_triggered_on_fallback"]:
        lines.append("- WARNING: smoothed_close_memory triggered on fallback; this should not happen.")
    if analysis["stage_transition_candidate_check"]["warning_centered_visible_triggered_transition"]:
        lines.append("- WARNING: centered_visible_count triggered a transition; this should not happen.")
    lines.append(
        "- fallback_rate="
        f"{analysis['fallback_impact_check']['fallback_rate']:.3f}; "
        "fallback nodes were ignored by stage tracker."
    )
    if analysis["fallback_impact_check"]["warning_transition_from_fallback"]:
        lines.append("- WARNING: a stage transition occurred on a fallback step.")
    lines.append(
        "- future_or_other_stage_attention_count="
        f"{analysis['observed_active_conflict_check']['future_or_other_stage_attention_count']}"
    )
    lines.append("")
    lines.append("Recommendation:")
    lines.append("- Stage memory fields are recorded successfully.")
    lines.append("- active_stage_target_visible is the key field for current-stage progress.")
    lines.append("- close/passed evidence controls tracker-generated completion candidates.")
    lines.append("- close memory can preserve a reliable close/near event through short fallback gaps.")
    lines.append("- long_approach_candidate indicates stable visible/centered tracking without close/near/passed evidence.")
    lines.append("- long_approach_confirmed is a conservative strong candidate and still requires consecutive confirmation.")
    lines.append("- If long approach appears without confirmed transition, inspect confirmed_count and consecutive runs.")
    lines.append("- centered_visible_count is weak evidence only and should not trigger stage transition.")
    lines.append("- If close/passed never trigger, improve visual definitions or add scale/depth cues later.")
    lines.append("- observed_stage_id should remain diagnostic visual evidence, not direct stage state.")
    lines.append("- Current stage evidence should remain out of prompt/action until reliability improves.")
    return "\n".join(lines)


def main():
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    analysis = analyze(graph_dir)
    analysis_path = graph_dir / "stage_memory_analysis.json"
    summary_path = graph_dir / "stage_memory_summary.txt"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(build_summary(analysis))
    print(f"[STAGE ANALYZE] wrote {analysis_path}")
    print(f"[STAGE ANALYZE] wrote {summary_path}")


if __name__ == "__main__":
    main()
