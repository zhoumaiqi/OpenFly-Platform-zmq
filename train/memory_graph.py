import csv
import json
from datetime import datetime


class MemoryGraph:
    def __init__(
        self,
        source="manual",
        env_name="",
        sample_id=None,
        instruction="",
    ):
        self.source = source
        self.env_name = env_name
        self.sample_id = sample_id
        self.instruction = instruction
        self.nodes = []
        self.edges = []

    def add_node(
        self,
        env_name,
        pose_x,
        pose_y,
        pose_z,
        yaw_deg,
        rgb_path,
        source=None,
        sample_id=None,
        step_id=None,
        instruction="",
        selected_image_path="",
        choice="",
        choice_group="",
        action_id="",
        action_name="",
        target_visible="",
        confidence="",
        raw_response="",
        fallback_used="",
        distance_to_goal="",
        manual_tag="",
        auto_tag=None,
        scene_summary="",
        active_stage="",
        observed_stage_id="",
        active_stage_target_visible="",
        active_stage_target_centered="",
        active_stage_target_close="",
        active_stage_target_passed="",
        stage_target_visible="",
        stage_progress="",
        stage_complete_candidate="",
        weak_stage_completion_candidate="",
        weak_stage_completion_reason="",
        stage_completion_candidate_by_tracker="",
        tracker_completion_reason="",
        stage_transition_candidate="",
        stage_transition="",
        long_approach_candidate="",
        long_approach_reason="",
        long_approach_score="",
        long_approach_visible_count="",
        long_approach_centered_count="",
        long_approach_weak_count="",
        long_approach_step_span="",
        long_approach_confirmed="",
        long_approach_confirmed_reason="",
        long_approach_confirmed_count="",
        consecutive_tracker_completion_count="",
        consecutive_long_approach_confirmed_count="",
        close_memory_active="",
        close_memory_age="",
        close_memory_source="",
        last_close_step="",
        last_near_step="",
        stage_memory_note="",
        reason="manual",
        note="",
        is_revisit=False,
    ):
        if auto_tag is None:
            auto_tag = reason

        node = {
            "node_id": len(self.nodes),
            "env_name": env_name,
            "sample_id": sample_id if sample_id is not None else self.sample_id,
            "step_id": step_id,
            "source": source or self.source,
            "trusted": True,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pose_x": pose_x,
            "pose_y": pose_y,
            "pose_z": pose_z,
            "yaw": yaw_deg,
            "yaw_deg": yaw_deg,
            "instruction": instruction or self.instruction,
            "rgb_path": str(rgb_path),
            "input_image_path": str(rgb_path),
            "selected_image_path": str(selected_image_path),
            "choice": choice,
            "choice_group": choice_group,
            "action_id": action_id,
            "action_name": action_name,
            "target_visible": target_visible,
            "confidence": confidence,
            "auto_tag": auto_tag,
            "manual_tag": manual_tag,
            "scene_summary": scene_summary,
            "active_stage": active_stage,
            "observed_stage_id": observed_stage_id,
            "active_stage_target_visible": active_stage_target_visible,
            "active_stage_target_centered": active_stage_target_centered,
            "active_stage_target_close": active_stage_target_close,
            "active_stage_target_passed": active_stage_target_passed,
            "stage_target_visible": stage_target_visible,
            "stage_progress": stage_progress,
            "stage_complete_candidate": stage_complete_candidate,
            "weak_stage_completion_candidate": weak_stage_completion_candidate,
            "weak_stage_completion_reason": weak_stage_completion_reason,
            "stage_completion_candidate_by_tracker": stage_completion_candidate_by_tracker,
            "tracker_completion_reason": tracker_completion_reason,
            "stage_transition_candidate": stage_transition_candidate,
            "stage_transition": stage_transition,
            "long_approach_candidate": long_approach_candidate,
            "long_approach_reason": long_approach_reason,
            "long_approach_score": long_approach_score,
            "long_approach_visible_count": long_approach_visible_count,
            "long_approach_centered_count": long_approach_centered_count,
            "long_approach_weak_count": long_approach_weak_count,
            "long_approach_step_span": long_approach_step_span,
            "long_approach_confirmed": long_approach_confirmed,
            "long_approach_confirmed_reason": long_approach_confirmed_reason,
            "long_approach_confirmed_count": long_approach_confirmed_count,
            "consecutive_tracker_completion_count": consecutive_tracker_completion_count,
            "consecutive_long_approach_confirmed_count": consecutive_long_approach_confirmed_count,
            "close_memory_active": close_memory_active,
            "close_memory_age": close_memory_age,
            "close_memory_source": close_memory_source,
            "last_close_step": last_close_step,
            "last_near_step": last_near_step,
            "stage_memory_note": stage_memory_note,
            "reason": reason,
            "raw_response": raw_response,
            "fallback_used": fallback_used,
            "distance_to_goal": distance_to_goal,
            "is_revisit": is_revisit,
            "note": note,
        }
        self.nodes.append(node)
        return node

    def add_edge(
        self,
        from_node,
        to_node,
        edge_type,
        distance="",
        yaw_delta="",
        step_start="",
        step_end="",
        actions_between=None,
        choices_between=None,
    ):
        edge = {
            "edge_id": len(self.edges),
            "from_node": from_node,
            "to_node": to_node,
            "edge_type": edge_type,
            "distance": distance,
            "yaw_delta": yaw_delta,
            "step_start": step_start,
            "step_end": step_end,
            "actions_between": actions_between or [],
            "choices_between": choices_between or [],
        }
        self.edges.append(edge)
        return edge

    def update_node_tag(self, node_id, tag):
        for node in self.nodes:
            if node["node_id"] == node_id:
                node["manual_tag"] = tag
                return node
        raise ValueError(f"node_id not found: {node_id}")

    def save_json(self, path):
        data = {
            "source": self.source,
            "env_name": self.env_name,
            "sample_id": self.sample_id,
            "instruction": self.instruction,
            "nodes": self.nodes,
            "edges": self.edges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_nodes_csv(self, path):
        fieldnames = [
            "node_id",
            "env_name",
            "sample_id",
            "step_id",
            "source",
            "trusted",
            "timestamp",
            "pose_x",
            "pose_y",
            "pose_z",
            "yaw",
            "yaw_deg",
            "instruction",
            "rgb_path",
            "input_image_path",
            "selected_image_path",
            "choice",
            "choice_group",
            "action_id",
            "action_name",
            "target_visible",
            "confidence",
            "auto_tag",
            "manual_tag",
            "scene_summary",
            "active_stage",
            "observed_stage_id",
            "active_stage_target_visible",
            "active_stage_target_centered",
            "active_stage_target_close",
            "active_stage_target_passed",
            "stage_target_visible",
            "stage_progress",
            "stage_complete_candidate",
            "weak_stage_completion_candidate",
            "weak_stage_completion_reason",
            "stage_completion_candidate_by_tracker",
            "tracker_completion_reason",
            "stage_transition_candidate",
            "stage_transition",
            "long_approach_candidate",
            "long_approach_reason",
            "long_approach_score",
            "long_approach_visible_count",
            "long_approach_centered_count",
            "long_approach_weak_count",
            "long_approach_step_span",
            "long_approach_confirmed",
            "long_approach_confirmed_reason",
            "long_approach_confirmed_count",
            "consecutive_tracker_completion_count",
            "consecutive_long_approach_confirmed_count",
            "close_memory_active",
            "close_memory_age",
            "close_memory_source",
            "last_close_step",
            "last_near_step",
            "stage_memory_note",
            "reason",
            "raw_response",
            "fallback_used",
            "distance_to_goal",
            "is_revisit",
            "note",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.nodes)

    def save_edges_csv(self, path):
        fieldnames = [
            "edge_id",
            "from_node",
            "to_node",
            "edge_type",
            "distance",
            "yaw_delta",
            "step_start",
            "step_end",
            "actions_between",
            "choices_between",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.edges)
