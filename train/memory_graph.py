import csv
import json
from datetime import datetime


class MemoryGraph:
    def __init__(self):
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
        manual_tag="",
        auto_tag=None,
        scene_summary="",
        reason="manual",
        note="",
    ):
        if auto_tag is None:
            auto_tag = reason

        node = {
            "node_id": len(self.nodes),
            "env_name": env_name,
            "source": "manual",
            "trusted": True,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pose_x": pose_x,
            "pose_y": pose_y,
            "pose_z": pose_z,
            "yaw_deg": yaw_deg,
            "rgb_path": str(rgb_path),
            "auto_tag": auto_tag,
            "manual_tag": manual_tag,
            "scene_summary": scene_summary,
            "reason": reason,
            "note": note,
        }
        self.nodes.append(node)
        return node

    def update_node_tag(self, node_id, tag):
        for node in self.nodes:
            if node["node_id"] == node_id:
                node["manual_tag"] = tag
                return node
        raise ValueError(f"node_id not found: {node_id}")

    def save_json(self, path):
        data = {
            "nodes": self.nodes,
            "edges": self.edges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_nodes_csv(self, path):
        fieldnames = [
            "node_id",
            "env_name",
            "source",
            "trusted",
            "timestamp",
            "pose_x",
            "pose_y",
            "pose_z",
            "yaw_deg",
            "rgb_path",
            "auto_tag",
            "manual_tag",
            "scene_summary",
            "reason",
            "note",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.nodes)
