import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


STAGE_SPLIT_RE = re.compile(
    r"\b(then|finally|next|afterward|afterwards|after that|subsequently)\b",
    flags=re.IGNORECASE,
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "toward",
    "towards",
    "go",
    "make",
    "move",
    "proceed",
    "continue",
    "advance",
    "straight",
    "slightly",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze one online memory graph sample.")
    parser.add_argument("--graph_dir", required=True)
    return parser.parse_args()


def load_json(path):
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
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def to_float(value):
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value):
    try:
        if value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def split_stages(instruction):
    if not instruction:
        return []
    parts = []
    last = 0
    for match in STAGE_SPLIT_RE.finditer(instruction):
        prefix = instruction[last:match.start()].strip(" ,.;")
        if prefix:
            parts.append(prefix)
        last = match.start()
    tail = instruction[last:].strip(" ,.;")
    if tail:
        parts.append(tail)
    return [
        {
            "stage_id": idx,
            "stage_text": stage,
            "keywords": extract_keywords(stage),
        }
        for idx, stage in enumerate(parts)
    ]


def extract_keywords(text):
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text.lower())
    keywords = []
    for word in words:
        if len(word) < 4 or word in STOPWORDS:
            continue
        if word not in keywords:
            keywords.append(word)
    return keywords[:12]


def infer_stage_for_node(node, stages):
    raw = str(node.get("raw_response", "")).lower()
    if not raw or not stages:
        return {"stage_id": None, "score": 0, "matched_keywords": []}

    best = {"stage_id": None, "score": 0, "matched_keywords": []}
    for stage in stages:
        matched = [keyword for keyword in stage["keywords"] if keyword in raw]
        if len(matched) > best["score"]:
            best = {
                "stage_id": stage["stage_id"],
                "score": len(matched),
                "matched_keywords": matched,
            }
    return best


def count_visibility_switches(nodes):
    values = [to_bool(node.get("target_visible", "")) for node in nodes if node.get("target_visible", "") != ""]
    switches = 0
    switch_steps = []
    for prev, cur, node in zip(values, values[1:], nodes[1:]):
        if prev != cur:
            switches += 1
            switch_steps.append(to_int(node.get("step_id")))
    return switches, switch_steps


def analyze_distance_to_goal(nodes):
    series = []
    for node in nodes:
        distance = to_float(node.get("distance_to_goal", ""))
        if distance is None:
            continue
        series.append(
            {
                "node_id": to_int(node.get("node_id")),
                "step_id": to_int(node.get("step_id")),
                "distance_to_goal": distance,
            }
        )
    if not series:
        return {
            "min_distance_to_goal": None,
            "min_step_id": None,
            "min_node_id": None,
            "increases_after_min": False,
            "overshoot_suspected": False,
            "series": [],
        }

    min_item = min(series, key=lambda item: item["distance_to_goal"])
    min_index = series.index(min_item)
    after = series[min_index + 1 :]
    increases_after_min = bool(after) and all(
        item["distance_to_goal"] >= prev["distance_to_goal"]
        for prev, item in zip(series[min_index:], after)
    )
    overshoot_suspected = increases_after_min and bool(after)
    return {
        "min_distance_to_goal": min_item["distance_to_goal"],
        "min_step_id": min_item["step_id"],
        "min_node_id": min_item["node_id"],
        "increases_after_min": increases_after_min,
        "overshoot_suspected": overshoot_suspected,
        "series": series,
    }


def analyze_action_changes(nodes, min_distance_to_goal):
    changes = []
    for node in nodes:
        if node.get("reason") != "action_change":
            continue
        distance = to_float(node.get("distance_to_goal", ""))
        near_goal = False
        if distance is not None and min_distance_to_goal is not None:
            near_goal = distance <= min_distance_to_goal + 20.0
        changes.append(
            {
                "node_id": to_int(node.get("node_id")),
                "step_id": to_int(node.get("step_id")),
                "choice": node.get("choice", ""),
                "action_id": node.get("action_id", ""),
                "action_name": node.get("action_name", ""),
                "distance_to_goal": distance,
                "near_goal": near_goal,
            }
        )
    return changes


def reliable_nodes(nodes):
    selected = []
    for node in nodes:
        confidence = to_float(node.get("confidence", ""))
        if to_bool(node.get("fallback_used", "")):
            continue
        if confidence is not None and confidence < 0.5:
            continue
        selected.append(
            {
                "node_id": to_int(node.get("node_id")),
                "step_id": to_int(node.get("step_id")),
                "reason": node.get("reason", ""),
                "choice": node.get("choice", ""),
                "action_name": node.get("action_name", ""),
                "target_visible": to_bool(node.get("target_visible", "")),
                "confidence": confidence,
                "distance_to_goal": to_float(node.get("distance_to_goal", "")),
            }
        )
    return selected


def make_memory_hint(analysis):
    signals = analysis["failure_signals"]
    hint_parts = []
    if signals["visibility_flicker"]:
        hint_parts.append("Target visibility flickered; avoid overreacting to a single visible/invisible frame.")
    if signals["high_fallback_rate"]:
        hint_parts.append("Several decisions used fallback parsing; prefer stable visual evidence over malformed responses.")
    if signals["overshoot_suspected"]:
        hint_parts.append("Distance to goal reached a minimum and then increased; consider stopping or turning near that point.")
    if analysis["action_changes"]:
        hint_parts.append("Action changes were observed; verify whether they correspond to a real stage transition.")
    if not hint_parts:
        hint_parts.append("No strong failure signal detected; continue using recent reliable nodes as route context.")
    return " ".join(hint_parts)


def build_summary(analysis):
    lines = []
    lines.append("Memory Graph Analysis")
    lines.append("=====================")
    lines.append("")
    lines.append(f"Graph dir: {analysis['graph_dir']}")
    lines.append(f"Nodes: {analysis['counts']['nodes']}")
    lines.append(f"Trajectory edges: {analysis['counts']['trajectory_edges']}")
    lines.append(f"Revisit edges: {analysis['counts']['revisit_edges']}")
    lines.append("")
    lines.append("Task stages:")
    for stage in analysis["stages"]:
        lines.append(f"- [{stage['stage_id']}] {stage['stage_text']}")
    if not analysis["stages"]:
        lines.append("- none parsed")
    lines.append("")
    lines.append("Key failure signals:")
    for key, value in analysis["failure_signals"].items():
        lines.append(f"- {key}: {value}")
    lines.append(f"- reason_distribution: {analysis['reason_distribution']}")
    lines.append(f"- fallback_rate: {analysis['fallback']['fallback_rate']:.3f}")
    lines.append("")
    lines.append("Reliable node summary:")
    for node in analysis["reliable_nodes"][:10]:
        lines.append(
            "- node={node_id} step={step_id} reason={reason} choice={choice} "
            "action={action_name} visible={target_visible} conf={confidence} "
            "dist={distance_to_goal}".format(**node)
        )
    if not analysis["reliable_nodes"]:
        lines.append("- none")
    lines.append("")
    lines.append("Memory hint:")
    lines.append(analysis["memory_hint"])
    lines.append("")
    return "\n".join(lines)


def analyze(graph_dir):
    graph_path = graph_dir / "memory_graph.json"
    nodes_path = graph_dir / "memory_nodes.csv"
    edges_path = graph_dir / "memory_edges.csv"
    graph = load_json(graph_path)
    nodes = graph.get("nodes") or load_csv(nodes_path)
    edges = graph.get("edges") or load_csv(edges_path)

    reason_distribution = Counter(node.get("reason", "") for node in nodes)
    trajectory_edges = [edge for edge in edges if edge.get("edge_type") == "trajectory"]
    revisit_edges = [edge for edge in edges if edge.get("edge_type") == "revisit"]

    visibility_switches, visibility_switch_steps = count_visibility_switches(nodes)
    fallback_nodes = [node for node in nodes if to_bool(node.get("fallback_used", ""))]
    fallback_rate = len(fallback_nodes) / len(nodes) if nodes else 0.0
    distance_analysis = analyze_distance_to_goal(nodes)
    action_changes = analyze_action_changes(nodes, distance_analysis["min_distance_to_goal"])

    instruction = graph.get("instruction", "")
    if not instruction and nodes:
        instruction = nodes[0].get("instruction", "")
    stages = split_stages(instruction)
    node_stage_matches = []
    stage_attention_counts = Counter()
    for node in nodes:
        match = infer_stage_for_node(node, stages)
        node_stage_matches.append(
            {
                "node_id": to_int(node.get("node_id")),
                "step_id": to_int(node.get("step_id")),
                **match,
            }
        )
        if match["stage_id"] is not None:
            stage_attention_counts[str(match["stage_id"])] += 1

    failure_signals = {
        "visibility_flicker": visibility_switches >= 2,
        "high_fallback_rate": fallback_rate >= 0.2,
        "overshoot_suspected": distance_analysis["overshoot_suspected"],
        "action_changes_near_goal": any(item["near_goal"] for item in action_changes),
        "revisit_detected": bool(revisit_edges) or any(to_bool(node.get("is_revisit", "")) for node in nodes),
    }

    analysis = {
        "graph_dir": str(graph_dir),
        "instruction": instruction,
        "counts": {
            "nodes": len(nodes),
            "edges": len(edges),
            "trajectory_edges": len(trajectory_edges),
            "revisit_edges": len(revisit_edges),
        },
        "reason_distribution": dict(reason_distribution),
        "focus_reasons": {
            "target_visible_change": reason_distribution.get("target_visible_change", 0),
            "distance": reason_distribution.get("distance", 0),
            "action_change": reason_distribution.get("action_change", 0),
        },
        "visibility": {
            "switch_count": visibility_switches,
            "switch_steps": visibility_switch_steps,
            "visibility_flicker": failure_signals["visibility_flicker"],
        },
        "fallback": {
            "fallback_count": len(fallback_nodes),
            "fallback_rate": fallback_rate,
            "low_reliability_nodes": [
                {
                    "node_id": to_int(node.get("node_id")),
                    "step_id": to_int(node.get("step_id")),
                    "reason": node.get("reason", ""),
                    "choice": node.get("choice", ""),
                    "confidence": to_float(node.get("confidence", "")),
                }
                for node in fallback_nodes
            ],
        },
        "distance_to_goal": distance_analysis,
        "action_changes": action_changes,
        "stages": stages,
        "node_stage_matches": node_stage_matches,
        "stage_attention_counts": dict(stage_attention_counts),
        "failure_signals": failure_signals,
        "reliable_nodes": reliable_nodes(nodes),
    }
    analysis["memory_hint"] = make_memory_hint(analysis)
    return analysis


def main():
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    analysis = analyze(graph_dir)

    analysis_path = graph_dir / "memory_analysis.json"
    summary_path = graph_dir / "memory_summary.txt"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(build_summary(analysis))

    print(f"[ANALYZE] wrote {analysis_path}")
    print(f"[ANALYZE] wrote {summary_path}")


if __name__ == "__main__":
    main()
