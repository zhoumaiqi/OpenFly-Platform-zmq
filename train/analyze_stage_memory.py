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
                "final_stage_completion_candidate": to_bool(
                    node.get("final_stage_completion_candidate", "")
                ),
                "task_complete_reason": node.get("task_complete_reason", ""),
                "task_complete_step": to_int(node.get("task_complete_step", ""), default=-1),
                "task_complete_confidence": to_float(node.get("task_complete_confidence", "")),
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
    sample_id = graph_json.get("sample_id", "")
    steps_path = graph_dir.parents[2] / "ours_steps.csv" if len(graph_dir.parents) >= 3 else None
    step_rows = []
    if steps_path is not None and steps_path.exists():
        step_rows = [
            row for row in load_csv(steps_path)
            if str(row.get("sample_id", "")) == str(sample_id)
        ]

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
    task_complete_values = bool_series(nodes, "task_complete_candidate")
    stop_ready_values = bool_series(nodes, "stop_ready_candidate")
    finish_ready_values = bool_series(nodes, "finish_ready_candidate")
    finish_mode_values = bool_series(nodes, "finish_mode")
    finish_ready_blocked_reasons = Counter(
        node.get("finish_ready_blocked_reason", "")
        for node in nodes
        if node.get("finish_ready_blocked_reason", "")
    )
    early_finish_blocked_reasons = Counter(
        node.get("early_finish_blocked_reason", "")
        for node in nodes
        if node.get("early_finish_blocked_reason", "")
    )
    lost_recovery_values = bool_series(nodes, "lost_recovery_mode")
    control_override_values = bool_series(nodes, "control_override_used")
    motion_hint_conflict_values = bool_series(nodes, "motion_hint_conflict")
    last_stage_id = len(stage_plan) - 1
    final_stage_nodes = [
        node for node in nodes
        if to_int(node.get("active_stage", ""), default=-1) == last_stage_id
    ]
    final_stage_expected_group_distribution = Counter(
        node.get("stage_motion_hint_expected_group", node.get("motion_hint_expected_group", ""))
        for node in final_stage_nodes
        if node.get("stage_motion_hint_expected_group", node.get("motion_hint_expected_group", ""))
    )
    final_stage_actual_group_distribution = Counter(
        node.get("motion_hint_actual_group", "")
        for node in final_stage_nodes
        if node.get("motion_hint_actual_group", "")
    )
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
    task_complete_candidate = to_bool(stage_memory.get("task_complete_candidate", False))
    stop_ready_candidate = to_bool(stage_memory.get("stop_ready_candidate", False))
    final_completion_reasons = {
        "close_count",
        "smoothed_close_memory",
        "vlm_complete_count",
        "long_approach_confirmed",
    }
    final_stage_strong_nodes = [
        {
            "node_id": to_int(node.get("node_id", "")),
            "step_id": to_int(node.get("step_id", "")),
            "active_stage": to_int(node.get("active_stage", "")),
            "tracker_completion_reason": node.get("tracker_completion_reason", ""),
            "consecutive_tracker_completion_count": to_int(
                node.get("consecutive_tracker_completion_count", ""),
                default=0,
            ),
            "confidence": to_float(node.get("confidence", "")),
        }
        for node in nodes
        if to_int(node.get("active_stage", "")) == last_stage_id
        and to_bool(node.get("stage_completion_candidate_by_tracker", ""))
        and node.get("tracker_completion_reason", "") in final_completion_reasons
        and to_int(node.get("consecutive_tracker_completion_count", ""), default=0) >= 2
    ]
    warning_final_stage_strong_without_task_complete = bool(
        final_stage_strong_nodes and not task_complete_candidate
    )
    post_task_complete_distance_delta = to_float(stage_memory.get("post_task_complete_distance_delta", ""))
    warning_distance_increased_after_task_complete = (
        task_complete_candidate and post_task_complete_distance_delta > 10.0
    )
    lost_forward_nodes = [
        {
            "node_id": to_int(node.get("node_id", "")),
            "step_id": to_int(node.get("step_id", "")),
            "action_name": node.get("action_name", ""),
            "confidence": to_float(node.get("confidence", "")),
            "stage_progress": node.get("stage_progress", ""),
        }
        for node in nodes
        if node.get("action_name") == "forward_3"
        and (
            node.get("stage_progress") == "lost"
            or not to_bool(node.get("active_stage_target_visible", node.get("stage_target_visible", "")))
        )
    ]
    warning_lost_but_forward = bool(lost_forward_nodes)
    stop_ready_forward_steps = [
        to_int(row.get("step_id", ""))
        for row in step_rows
        if to_bool(row.get("stop_ready_candidate", ""))
        and row.get("action_name", "") == "forward_3"
    ]
    first_finish_ready_step_value = to_int(stage_memory.get("first_finish_ready_step", ""), default=-1)
    confirmed_finish_ready_step_value = to_int(stage_memory.get("confirmed_finish_ready_step", ""), default=-1)
    early_finish_step_value = to_int(stage_memory.get("early_finish_step", ""), default=-1)
    finish_ready_to_confirm_steps = (
        confirmed_finish_ready_step_value - first_finish_ready_step_value
        if first_finish_ready_step_value >= 0 and confirmed_finish_ready_step_value >= 0
        else ""
    )
    finish_ready_to_early_finish_steps = (
        early_finish_step_value - first_finish_ready_step_value
        if first_finish_ready_step_value >= 0 and early_finish_step_value >= 0
        else ""
    )
    finish_ready_forward_steps = [
        to_int(row.get("step_id", ""))
        for row in step_rows
        if first_finish_ready_step_value >= 0
        and to_int(row.get("step_id", ""), default=-1) >= first_finish_ready_step_value
        and row.get("action_name", "") == "forward_3"
    ]
    confirmed_finish_forward_steps = [
        to_int(row.get("step_id", ""))
        for row in step_rows
        if confirmed_finish_ready_step_value >= 0
        and to_int(row.get("step_id", ""), default=-1) >= confirmed_finish_ready_step_value
        and row.get("action_name", "") == "forward_3"
    ]
    warning_stop_ready_forward_many = len(stop_ready_forward_steps) > 8
    warning_finish_mode_long_no_early_finish = (
        longest_true_run(finish_mode_values) > 30
        and not to_bool(stage_memory.get("early_finish_by_memory", False))
    )
    warning_early_finish_confirmation_too_conservative = (
        isinstance(finish_ready_to_early_finish_steps, int)
        and finish_ready_to_early_finish_steps >= 4
    )
    warning_early_finish_without_motion_hint = (
        to_bool(stage_memory.get("early_finish_by_memory", False))
        and to_bool(stage_memory.get("stage_motion_hint_required_for_finish", False))
        and not to_bool(stage_memory.get("stage_motion_hint_satisfied", True))
    )
    motion_hint_missing_rows = [
        {
            "step_id": to_int(row.get("step_id", "")),
            "expected": row.get("motion_hint_expected_group", ""),
            "actual": row.get("motion_hint_actual_group", ""),
        }
        for row in step_rows
        if row.get("motion_hint_expected_group", "") and not row.get("motion_hint_actual_group", "")
    ]
    warning_motion_hint_fields_missing = bool(motion_hint_missing_rows)
    action_values = [node.get("action_name", "") for node in nodes]
    action_counts = Counter(action_values)
    action_total = len([action for action in action_values if action])
    forward_action_count = action_counts.get("forward_3", 0)
    turn_left_action_count = action_counts.get("turn_left_30", 0)
    turn_right_action_count = action_counts.get("turn_right_30", 0)
    forward_ratio = forward_action_count / action_total if action_total else 0.0
    motion_hint_override_nodes = [
        {
            "node_id": to_int(node.get("node_id", "")),
            "step_id": to_int(node.get("step_id", "")),
            "direction": node.get("motion_hint_override_direction", ""),
            "original_action": node.get("motion_hint_original_action_name", ""),
            "corrected_action": node.get("motion_hint_corrected_action_name", ""),
        }
        for node in nodes
        if to_bool(node.get("motion_hint_control_applied", ""))
    ]
    motion_hint_override_blocked_reasons = Counter(
        node.get("motion_hint_override_blocked_reason", "")
        for node in nodes
        if node.get("motion_hint_override_blocked_reason", "")
    )
    stage_has_lr_hint = any(
        "left" in str(stage.get("motion_hint", "")).lower()
        or "right" in str(stage.get("motion_hint", "")).lower()
        for stage in stage_plan
    )
    warning_motion_hint_conflict_no_override = (
        sum(motion_hint_conflict_values) > 0 and not motion_hint_override_nodes
    )
    warning_forward_dominated_with_lr_hint = forward_ratio > 0.9 and stage_has_lr_hint
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
        "task_complete_check": {
            "task_complete_candidate": task_complete_candidate,
            "task_complete_step": stage_memory.get("task_complete_step", ""),
            "task_complete_reason": stage_memory.get("task_complete_reason", ""),
            "task_complete_confidence": stage_memory.get("task_complete_confidence", ""),
            "final_stage_completion_count": stage_memory.get("final_stage_completion_count", 0),
            "stop_ready_candidate": stop_ready_candidate,
            "stop_ready_step": stage_memory.get("stop_ready_step", ""),
            "stop_ready_reason": stage_memory.get("stop_ready_reason", ""),
            "stop_ready_confidence": stage_memory.get("stop_ready_confidence", ""),
            "stop_ready_count": stage_memory.get("stop_ready_count", 0),
            "first_task_complete_step": stage_memory.get("first_task_complete_step", ""),
            "first_stop_ready_step": stage_memory.get("first_stop_ready_step", ""),
            "post_task_complete_steps": stage_memory.get("post_task_complete_steps", 0),
            "post_task_complete_lost_count": stage_memory.get("post_task_complete_lost_count", 0),
            "post_task_complete_revisit_count": stage_memory.get("post_task_complete_revisit_count", 0),
            "post_task_complete_action_counts": stage_memory.get("post_task_complete_action_counts", {}),
            "post_task_complete_control_override_count": stage_memory.get("post_task_complete_control_override_count", 0),
            "post_task_complete_distance_min": stage_memory.get("post_task_complete_distance_min", ""),
            "post_task_complete_distance_last": stage_memory.get("post_task_complete_distance_last", ""),
            "post_task_complete_distance_delta": stage_memory.get("post_task_complete_distance_delta", ""),
            "warning_distance_increased_after_task_complete": warning_distance_increased_after_task_complete,
            "node_true_count": sum(task_complete_values),
            "nodes": true_node_refs(nodes, "task_complete_candidate"),
            "stop_ready_node_true_count": sum(stop_ready_values),
            "stop_ready_nodes": true_node_refs(nodes, "stop_ready_candidate"),
            "first_stop_ready_step": stage_memory.get("first_stop_ready_step", ""),
            "finish_mode_enter_step": stage_memory.get("finish_mode_enter_step", ""),
            "finish_mode_exit_step": stage_memory.get("finish_mode_exit_step", ""),
            "finish_mode_exit_reason": stage_memory.get("finish_mode_exit_reason", ""),
            "finish_mode_reset_count": stage_memory.get("finish_mode_reset_count", 0),
            "finish_mode_invalidated": stage_memory.get("finish_mode_invalidated", False),
            "finish_mode_active_steps": sum(finish_mode_values),
            "finish_mode_longest_run": longest_true_run(finish_mode_values),
            "finish_mode_consecutive_not_ready_count": stage_memory.get("finish_mode_consecutive_not_ready_count", ""),
            "finish_mode_consecutive_approaching_count": stage_memory.get("finish_mode_consecutive_approaching_count", ""),
            "finish_mode_consecutive_stop_not_ready_count": stage_memory.get("finish_mode_consecutive_stop_not_ready_count", ""),
            "first_finish_ready_step": stage_memory.get("first_finish_ready_step", ""),
            "finish_ready_reason": stage_memory.get("finish_ready_reason", ""),
            "finish_ready_streak": stage_memory.get("finish_ready_streak", 0),
            "finish_ready_streak_required": stage_memory.get("finish_ready_streak_required", ""),
            "finish_ready_blocked_reason": stage_memory.get("finish_ready_blocked_reason", ""),
            "finish_ready_blocked_reason_distribution": dict(finish_ready_blocked_reasons),
            "finish_ready_motion_hint_not_satisfied_count": finish_ready_blocked_reasons.get(
                "motion_hint_not_satisfied",
                0,
            ),
            "confirmed_finish_ready_step": stage_memory.get("confirmed_finish_ready_step", ""),
            "confirmed_finish_ready_reason": stage_memory.get("confirmed_finish_ready_reason", ""),
            "confirmed_finish_ready": bool(stage_memory.get("confirmed_finish_ready_step", "")),
            "finish_ready_to_confirm_steps": finish_ready_to_confirm_steps,
            "finish_ready_to_early_finish_steps": finish_ready_to_early_finish_steps,
            "finish_ready_node_true_count": sum(finish_ready_values),
            "finish_ready_nodes": true_node_refs(nodes, "finish_ready_candidate"),
            "early_finish_by_memory": stage_memory.get("early_finish_by_memory", False),
            "early_finish_step": stage_memory.get("early_finish_step", ""),
            "early_finish_reason": stage_memory.get("early_finish_reason", ""),
            "early_finish_confirm_policy": stage_memory.get("early_finish_confirm_policy", ""),
            "early_finish_blocked_reason": stage_memory.get("early_finish_blocked_reason", ""),
            "early_finish_blocked_reason_distribution": dict(early_finish_blocked_reasons),
            "stage_motion_hint_expected_group": stage_memory.get("stage_motion_hint_expected_group", ""),
            "stage_motion_hint_required_for_finish": stage_memory.get("stage_motion_hint_required_for_finish", False),
            "stage_motion_hint_satisfied": stage_memory.get("stage_motion_hint_satisfied", True),
            "stage_motion_hint_satisfied_step": stage_memory.get("stage_motion_hint_satisfied_step", ""),
            "stage_motion_hint_satisfied_reason": stage_memory.get("stage_motion_hint_satisfied_reason", ""),
            "stage_motion_hint_satisfied_by_action": stage_memory.get("stage_motion_hint_satisfied_by_action", False),
            "stage_motion_hint_satisfied_by_choice": stage_memory.get("stage_motion_hint_satisfied_by_choice", False),
            "stage_motion_hint_satisfied_by_override": stage_memory.get("stage_motion_hint_satisfied_by_override", False),
            "final_stage_expected_group_distribution": dict(final_stage_expected_group_distribution),
            "final_stage_actual_group_distribution": dict(final_stage_actual_group_distribution),
            "post_stop_ready_steps": stage_memory.get("post_stop_ready_steps", 0),
            "post_stop_ready_action_counts": stage_memory.get("post_stop_ready_action_counts", {}),
            "stop_ready_forward_steps": stop_ready_forward_steps,
            "stop_ready_forward_count": len(stop_ready_forward_steps),
            "finish_ready_forward_steps": finish_ready_forward_steps,
            "finish_ready_forward_count": len(finish_ready_forward_steps),
            "confirmed_finish_forward_steps": confirmed_finish_forward_steps,
            "confirmed_finish_forward_count": len(confirmed_finish_forward_steps),
            "warning_stop_ready_forward_many": warning_stop_ready_forward_many,
            "warning_finish_mode_long_no_early_finish": warning_finish_mode_long_no_early_finish,
            "warning_early_finish_confirmation_too_conservative": warning_early_finish_confirmation_too_conservative,
            "warning_early_finish_without_motion_hint": warning_early_finish_without_motion_hint,
            "final_stage_strong_nodes": final_stage_strong_nodes,
            "warning_final_stage_strong_without_task_complete": warning_final_stage_strong_without_task_complete,
        },
        "memory_control_check": {
            "motion_hint_conflict_count": sum(motion_hint_conflict_values),
            "motion_hint_override_count": len(motion_hint_override_nodes),
            "motion_hint_override_applied_steps": [
                node["step_id"] for node in motion_hint_override_nodes
            ],
            "motion_hint_override_directions": dict(
                Counter(node["direction"] for node in motion_hint_override_nodes)
            ),
            "motion_hint_override_blocked_reasons": dict(motion_hint_override_blocked_reasons),
            "warning_motion_hint_conflict_no_override": warning_motion_hint_conflict_no_override,
            "lost_recovery_count": sum(lost_recovery_values),
            "control_override_count": sum(control_override_values),
            "forward_action_count": forward_action_count,
            "turn_left_action_count": turn_left_action_count,
            "turn_right_action_count": turn_right_action_count,
            "forward_ratio": forward_ratio,
            "warning_forward_dominated_with_lr_hint": warning_forward_dominated_with_lr_hint,
            "lost_forward_count": len(lost_forward_nodes),
            "lost_forward_nodes": lost_forward_nodes[:20],
            "warning_lost_but_forward": warning_lost_but_forward,
            "motion_hint_missing_rows": motion_hint_missing_rows[:20],
            "warning_motion_hint_fields_missing": warning_motion_hint_fields_missing,
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
    if analysis["task_complete_check"]["task_complete_candidate"]:
        lines.append(
            "- TASK COMPLETE CANDIDATE: yes "
            f"step={analysis['task_complete_check']['task_complete_step']} "
            f"reason={analysis['task_complete_check']['task_complete_reason']} "
            f"confidence={analysis['task_complete_check']['task_complete_confidence']}"
        )
        lines.append(
            "- STOP READY CANDIDATE: "
            f"{'yes' if analysis['task_complete_check']['stop_ready_candidate'] else 'no'} "
            f"step={analysis['task_complete_check']['stop_ready_step']} "
            f"reason={analysis['task_complete_check']['stop_ready_reason']} "
            f"count={analysis['task_complete_check']['stop_ready_count']}"
        )
        lines.append(
            "- finish_mode: "
            f"enter_step={analysis['task_complete_check']['finish_mode_enter_step']}, "
            f"exit_step={analysis['task_complete_check']['finish_mode_exit_step']}, "
            f"exit_reason={analysis['task_complete_check']['finish_mode_exit_reason']}, "
            f"reset_count={analysis['task_complete_check']['finish_mode_reset_count']}, "
            f"active_steps={analysis['task_complete_check']['finish_mode_active_steps']}, "
            f"longest_run={analysis['task_complete_check']['finish_mode_longest_run']}, "
            f"first_finish_ready_step={analysis['task_complete_check']['first_finish_ready_step']}, "
            f"confirmed_finish_ready_step={analysis['task_complete_check']['confirmed_finish_ready_step']}, "
            f"finish_ready_reason={analysis['task_complete_check']['finish_ready_reason']}, "
            f"finish_ready_streak={analysis['task_complete_check']['finish_ready_streak']}, "
            f"finish_ready_streak_required={analysis['task_complete_check']['finish_ready_streak_required']}, "
            f"finish_ready_to_confirm_steps={analysis['task_complete_check']['finish_ready_to_confirm_steps']}, "
            f"finish_ready_to_early_finish_steps={analysis['task_complete_check']['finish_ready_to_early_finish_steps']}, "
            f"early_finish_confirm_policy={analysis['task_complete_check']['early_finish_confirm_policy']}, "
            f"early_finish={analysis['task_complete_check']['early_finish_by_memory']} "
            f"early_step={analysis['task_complete_check']['early_finish_step']}"
        )
        if analysis["task_complete_check"]["finish_ready_blocked_reason_distribution"]:
            lines.append(
                "- finish_ready_blocked_reasons: "
                f"{analysis['task_complete_check']['finish_ready_blocked_reason_distribution']}"
            )
        if analysis["task_complete_check"]["early_finish_blocked_reason_distribution"]:
            lines.append(
                "- early_finish_blocked_reasons: "
                f"{analysis['task_complete_check']['early_finish_blocked_reason_distribution']}"
            )
        lines.append(
            "- final_stage_motion_hint: "
            f"expected={analysis['task_complete_check']['stage_motion_hint_expected_group']}, "
            f"required={analysis['task_complete_check']['stage_motion_hint_required_for_finish']}, "
            f"satisfied={analysis['task_complete_check']['stage_motion_hint_satisfied']}, "
            f"step={analysis['task_complete_check']['stage_motion_hint_satisfied_step']}, "
            f"reason={analysis['task_complete_check']['stage_motion_hint_satisfied_reason']}, "
            f"by_action={analysis['task_complete_check']['stage_motion_hint_satisfied_by_action']}, "
            f"by_choice={analysis['task_complete_check']['stage_motion_hint_satisfied_by_choice']}, "
            f"by_override={analysis['task_complete_check']['stage_motion_hint_satisfied_by_override']}"
        )
        lines.append(
            "- final_stage_motion_hint_groups: "
            f"expected={analysis['task_complete_check']['final_stage_expected_group_distribution']}, "
            f"actual={analysis['task_complete_check']['final_stage_actual_group_distribution']}"
        )
        lines.append(
            "- post_stop_ready: "
            f"steps={analysis['task_complete_check']['post_stop_ready_steps']}, "
            f"actions={analysis['task_complete_check']['post_stop_ready_action_counts']}"
        )
        if analysis["task_complete_check"]["warning_stop_ready_forward_many"]:
            lines.append("- WARNING: stop_ready_candidate active but agent continued forward for many steps.")
        lines.append(
            "- forward_after_finish_signals: "
            f"stop_ready_forward={analysis['task_complete_check']['stop_ready_forward_count']}, "
            f"finish_ready_forward={analysis['task_complete_check']['finish_ready_forward_count']}, "
            f"confirmed_finish_forward={analysis['task_complete_check']['confirmed_finish_forward_count']}"
        )
        if analysis["task_complete_check"]["warning_finish_mode_long_no_early_finish"]:
            lines.append("- WARNING: finish_mode stayed active for many steps without early_finish.")
        if analysis["task_complete_check"]["warning_early_finish_confirmation_too_conservative"]:
            lines.append(
                "- WARNING: early_finish confirmation may be too conservative; "
                "agent continued too many steps after first finish_ready."
            )
        if analysis["task_complete_check"]["warning_early_finish_without_motion_hint"]:
            lines.append("- WARNING: early_finish triggered without satisfying final-stage motion hint.")
        lines.append(
            "- post_task_complete: "
            f"steps={analysis['task_complete_check']['post_task_complete_steps']}, "
            f"lost={analysis['task_complete_check']['post_task_complete_lost_count']}, "
            f"actions={analysis['task_complete_check']['post_task_complete_action_counts']}, "
            f"distance_min={analysis['task_complete_check']['post_task_complete_distance_min']}, "
            f"distance_last={analysis['task_complete_check']['post_task_complete_distance_last']}, "
            f"delta={analysis['task_complete_check']['post_task_complete_distance_delta']}"
        )
        if analysis["task_complete_check"]["warning_distance_increased_after_task_complete"]:
            lines.append(
                "- WARNING: distance increased after task_complete_candidate; "
                "completion may be too early or stop control is missing."
            )
    elif analysis["task_complete_check"]["warning_final_stage_strong_without_task_complete"]:
        lines.append(
            "- WARNING: final stage has repeated strong completion evidence "
            "but no task_complete_candidate was recorded."
        )
        lines.append(
            "- final stage strong nodes: "
            f"{analysis['task_complete_check']['final_stage_strong_nodes']}"
        )
    if not analysis["task_complete_check"]["task_complete_candidate"]:
        lines.append(
            "- finish_control: "
            f"enter_step={analysis['task_complete_check']['finish_mode_enter_step']}, "
            f"exit_step={analysis['task_complete_check']['finish_mode_exit_step']}, "
            f"exit_reason={analysis['task_complete_check']['finish_mode_exit_reason']}, "
            f"reset_count={analysis['task_complete_check']['finish_mode_reset_count']}, "
            f"active_steps={analysis['task_complete_check']['finish_mode_active_steps']}, "
            f"longest_run={analysis['task_complete_check']['finish_mode_longest_run']}, "
            f"first_finish_ready_step={analysis['task_complete_check']['first_finish_ready_step']}, "
            f"confirmed_finish_ready_step={analysis['task_complete_check']['confirmed_finish_ready_step']}, "
            f"finish_ready_streak={analysis['task_complete_check']['finish_ready_streak']}, "
            f"finish_ready_streak_required={analysis['task_complete_check']['finish_ready_streak_required']}, "
            f"finish_ready_to_confirm_steps={analysis['task_complete_check']['finish_ready_to_confirm_steps']}, "
            f"finish_ready_to_early_finish_steps={analysis['task_complete_check']['finish_ready_to_early_finish_steps']}, "
            f"early_finish_confirm_policy={analysis['task_complete_check']['early_finish_confirm_policy']}, "
            f"early_finish={analysis['task_complete_check']['early_finish_by_memory']}"
        )
        if analysis["task_complete_check"]["finish_ready_blocked_reason_distribution"]:
            lines.append(
                "- finish_ready_blocked_reasons: "
                f"{analysis['task_complete_check']['finish_ready_blocked_reason_distribution']}"
            )
        if analysis["task_complete_check"]["early_finish_blocked_reason_distribution"]:
            lines.append(
                "- early_finish_blocked_reasons: "
                f"{analysis['task_complete_check']['early_finish_blocked_reason_distribution']}"
            )
        lines.append(
            "- final_stage_motion_hint: "
            f"expected={analysis['task_complete_check']['stage_motion_hint_expected_group']}, "
            f"required={analysis['task_complete_check']['stage_motion_hint_required_for_finish']}, "
            f"satisfied={analysis['task_complete_check']['stage_motion_hint_satisfied']}, "
            f"step={analysis['task_complete_check']['stage_motion_hint_satisfied_step']}, "
            f"reason={analysis['task_complete_check']['stage_motion_hint_satisfied_reason']}, "
            f"expected_groups={analysis['task_complete_check']['final_stage_expected_group_distribution']}, "
            f"actual_groups={analysis['task_complete_check']['final_stage_actual_group_distribution']}"
        )
        if analysis["task_complete_check"]["warning_finish_mode_long_no_early_finish"]:
            lines.append("- WARNING: finish_mode stayed active for many steps without early_finish.")
        if analysis["task_complete_check"]["warning_early_finish_confirmation_too_conservative"]:
            lines.append(
                "- WARNING: early_finish confirmation may be too conservative; "
                "agent continued too many steps after first finish_ready."
            )
        if analysis["task_complete_check"]["warning_early_finish_without_motion_hint"]:
            lines.append("- WARNING: early_finish triggered without satisfying final-stage motion hint.")
    lines.append(
        "- memory_control: "
        f"motion_hint_conflict_count={analysis['memory_control_check']['motion_hint_conflict_count']}, "
        f"motion_hint_override_count={analysis['memory_control_check']['motion_hint_override_count']}, "
        f"lost_recovery_count={analysis['memory_control_check']['lost_recovery_count']}, "
        f"control_override_count={analysis['memory_control_check']['control_override_count']}"
    )
    lines.append(
        "- action_distribution: "
        f"forward_3={analysis['memory_control_check']['forward_action_count']}, "
        f"turn_left_30={analysis['memory_control_check']['turn_left_action_count']}, "
        f"turn_right_30={analysis['memory_control_check']['turn_right_action_count']}, "
        f"forward_ratio={analysis['memory_control_check']['forward_ratio']:.3f}"
    )
    if analysis["memory_control_check"]["motion_hint_override_blocked_reasons"]:
        lines.append(
            "- motion_hint_override_blocked_reasons: "
            f"{analysis['memory_control_check']['motion_hint_override_blocked_reasons']}"
        )
    if analysis["memory_control_check"]["warning_motion_hint_conflict_no_override"]:
        lines.append("- WARNING: motion_hint conflicts detected but no action override was applied.")
    if analysis["memory_control_check"]["warning_forward_dominated_with_lr_hint"]:
        lines.append("- WARNING: actions are dominated by forward_3 despite left/right motion hints.")
    if analysis["memory_control_check"]["warning_lost_but_forward"]:
        lines.append("- WARNING: target lost but forward action continued.")
    if analysis["memory_control_check"]["warning_motion_hint_fields_missing"]:
        lines.append("- WARNING: motion_hint fields missing in step log.")
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
