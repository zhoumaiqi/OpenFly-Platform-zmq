import base64
import json
import os
import re

import cv2
import requests


ACTION_NAMES = {
    1: "forward_3",
    2: "turn_left_30",
    3: "turn_right_30",
}

CANDIDATE_RATIOS = [
    (0.10, 0.30), (0.30, 0.30), (0.50, 0.30), (0.70, 0.30), (0.90, 0.30),
    (0.10, 0.50), (0.30, 0.50), (0.50, 0.50), (0.70, 0.50), (0.90, 0.50),
    (0.10, 0.70), (0.30, 0.70), (0.50, 0.70), (0.70, 0.70), (0.90, 0.70),
]

P_TO_ACTION_ID = {
    "P1": 2,
    "P6": 2,
    "P11": 2,
    "P5": 3,
    "P10": 3,
    "P15": 3,
    "P2": 1,
    "P3": 1,
    "P4": 1,
    "P7": 1,
    "P8": 1,
    "P9": 1,
    "P12": 1,
    "P13": 1,
    "P14": 1,
}


class OurSPFAgent:
    def __init__(self, base_url, model_name, output_dir, request_timeout=180):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.output_dir = output_dir
        self.request_timeout = request_timeout
        self.image_dir = os.path.join(output_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

    def act(
        self,
        image,
        instruction,
        sample_id=None,
        step_id=None,
        pose=None,
        history=None,
        stage_plan=None,
        active_stage=0,
    ):
        del pose, history
        candidates = self._generate_candidate_points(image)
        input_overlay = self._draw_candidates(image, candidates)
        prefix = self._make_prefix(sample_id, step_id)
        input_image_path = os.path.join(self.image_dir, f"{prefix}_candidates.jpg")
        cv2.imwrite(input_image_path, input_overlay)

        raw_response = ""
        fallback_used = False
        parse_error = ""
        fallback_reason = ""
        try:
            raw_response = self._call_vlm(
                input_overlay,
                instruction,
                candidates,
                stage_plan=stage_plan,
                active_stage=active_stage,
            )
            parsed = self._parse_response(raw_response)
        except Exception as exc:
            parse_error = str(exc)
            fallback_reason = f"parse_error: {type(exc).__name__}"
            print(f"[ours] JSON parse failed, fallback P8, error={parse_error}")
            parsed = {
                "choice": "P8",
                "target_visible": False,
                "confidence": 0.0,
                "reason": fallback_reason,
                "observed_stage_id": -1,
                "active_stage_target_visible": False,
                "active_stage_target_centered": False,
                "active_stage_target_close": False,
                "active_stage_target_passed": False,
                "stage_progress": "unclear",
                "stage_complete_candidate": False,
            }
            fallback_used = True

        choice = str(parsed.get("choice", "P8")).upper()
        if choice not in P_TO_ACTION_ID:
            fallback_used = True
            fallback_reason = f"invalid_choice: {choice}"
            parse_error = parse_error or fallback_reason
            print(f"[ours] invalid choice, fallback P8, choice={choice}")
            choice = "P8"

        target_visible = self._to_bool(parsed.get("target_visible", False))
        confidence = self._to_float(parsed.get("confidence", 0.0))
        reason = str(parsed.get("reason", ""))[:200]
        observed_stage_id = self._to_int(parsed.get("observed_stage_id", -1), default=-1)
        active_stage_target_visible = self._to_bool(
            parsed.get("active_stage_target_visible", parsed.get("stage_target_visible", False))
        )
        active_stage_target_centered = self._to_bool(parsed.get("active_stage_target_centered", False))
        active_stage_target_close = self._to_bool(parsed.get("active_stage_target_close", False))
        active_stage_target_passed = self._to_bool(parsed.get("active_stage_target_passed", False))
        stage_progress = str(parsed.get("stage_progress", "unclear")).strip().lower()
        if stage_progress not in {"approaching", "near", "passed", "lost", "unclear"}:
            stage_progress = "unclear"
        stage_complete_candidate = self._to_bool(parsed.get("stage_complete_candidate", False))
        action_id = P_TO_ACTION_ID[choice]

        selected_overlay = self._draw_candidates(image, candidates, selected_id=choice)
        selected_image_path = os.path.join(self.image_dir, f"{prefix}_selected.jpg")
        cv2.imwrite(selected_image_path, selected_overlay)

        info = {
            "choice": choice,
            "target_visible": target_visible,
            "confidence": confidence,
            "action_id": action_id,
            "action_name": ACTION_NAMES.get(action_id, str(action_id)),
            "reason": reason,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "parse_error": parse_error,
            "raw_response": raw_response,
            "input_image_path": input_image_path,
            "selected_image_path": selected_image_path,
            "observed_stage_id": observed_stage_id,
            "active_stage_target_visible": active_stage_target_visible,
            "active_stage_target_centered": active_stage_target_centered,
            "active_stage_target_close": active_stage_target_close,
            "active_stage_target_passed": active_stage_target_passed,
            "stage_target_visible": active_stage_target_visible,
            "stage_progress": stage_progress,
            "stage_complete_candidate": stage_complete_candidate,
        }
        if not fallback_used:
            print(
                f"[ours] parsed choice={choice} action={action_id} "
                f"visible={target_visible} confidence={confidence:.2f}"
            )
        else:
            print(
                f"[ours] choice={choice}, action={action_id}, "
                f"visible={target_visible}, confidence={confidence:.2f}, fallback={fallback_used}"
            )
        return action_id, info

    def _make_prefix(self, sample_id, step_id):
        sample = "sample_unknown" if sample_id is None else f"sample_{sample_id:04d}"
        step = "step_unknown" if step_id is None else f"step_{step_id:04d}"
        return f"{sample}_{step}"

    def _generate_candidate_points(self, image):
        height, width = image.shape[:2]
        candidates = []
        for idx, (x_ratio, y_ratio) in enumerate(CANDIDATE_RATIOS, 1):
            candidates.append(
                {
                    "id": f"P{idx}",
                    "point": [int(round(y_ratio * 1000)), int(round(x_ratio * 1000))],
                    "pixel": (int(round(x_ratio * width)), int(round(y_ratio * height))),
                }
            )
        return candidates

    def _draw_candidates(self, image, candidates, selected_id=None):
        # AirSim Scene images are commonly RGB, while cv2 drawing/saving uses BGR.
        # The overlay colors are diagnostic only; the content is left unchanged.
        annotated = image.copy()
        for candidate in candidates:
            x, y = candidate["pixel"]
            is_selected = candidate["id"] == selected_id
            fill_color = (0, 255, 255) if not is_selected else (0, 180, 0)
            border_color = (0, 0, 0) if not is_selected else (0, 0, 255)
            radius = 13 if is_selected else 10
            cv2.circle(annotated, (x, y), radius, fill_color, -1)
            cv2.circle(annotated, (x, y), radius, border_color, 2)
            cv2.putText(annotated, candidate["id"], (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(annotated, candidate["id"], (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return annotated

    def _call_vlm(self, image, instruction, candidates, stage_plan=None, active_stage=0):
        image_url = self._image_to_data_url(image)
        candidate_lines = "\n".join(
            f'- {candidate["id"]}: [y, x] = {candidate["point"]}'
            for candidate in candidates
        )
        stage_lines = self._format_stage_plan(stage_plan)
        active_stage_info = self._format_active_stage(stage_plan, active_stage)
        system_prompt = (
            "You are a waypoint selection module. You must return only valid JSON. "
            "No explanations. No markdown. No reasoning."
        )
        prompt = (
            "Return ONLY one valid JSON object.\n"
            "Do not include Markdown.\n"
            "Do not include analysis paragraphs.\n"
            "Do not include explanation outside JSON.\n"
            "Do not wrap JSON in code fences.\n"
            "Use lowercase true/false.\n"
            "All required keys must be present.\n"
            "Do not include task breakdown. Do not include reasoning.\n"
            "Output final JSON only.\n"
            "The first character must be { and the last character must be }.\n\n"
            "The image contains labeled candidate waypoints P1 to P15.\n"
            f"Task: {instruction}\n\n"
            "You are given a full instruction decomposed into stages.\n"
            f"active_stage: {active_stage}\n"
            f"Stages:\n{stage_lines}\n\n"
            f"Current active stage:\n{active_stage_info}\n\n"
            "The following fields must be judged ONLY relative to the current active stage target:\n"
            "- active_stage_target_visible\n"
            "- active_stage_target_centered\n"
            "- active_stage_target_close\n"
            "- active_stage_target_passed\n"
            "- stage_progress\n"
            "- stage_complete_candidate\n\n"
            "observed_stage_id can be different from active_stage, but it is only used to record which stage the current image visually resembles.\n\n"
            "Choose exactly one candidate that best moves the drone toward the task.\n"
            "Output format, exactly one object:\n"
            "{\"choice\":\"P8\",\"target_visible\":true,\"confidence\":0.9,\"reason\":\"brief\","
            "\"observed_stage_id\":0,\"active_stage_target_visible\":true,"
            "\"active_stage_target_centered\":true,\"active_stage_target_close\":false,"
            "\"active_stage_target_passed\":false,"
            "\"stage_progress\":\"approaching\",\"stage_complete_candidate\":false}\n\n"
            f"Valid choices:\n{candidate_lines}\n\n"
            "Rules:\n"
            "- choice must be one of P1 through P15.\n"
            "- target_visible must be true or false.\n"
            "- confidence must be a number from 0.0 to 1.0.\n"
            "- Keep reason under 12 words.\n"
            "- observed_stage_id is visual evidence only: which stage the current image and selected waypoint best match.\n"
            "- observed_stage_id must be based on visual evidence; do not simply copy active_stage; use -1 if unclear.\n"
            "- observed_stage_id does not mean the task has progressed to that stage.\n"
            "- active_stage_target_visible, active_stage_target_centered, active_stage_target_close, active_stage_target_passed, stage_progress, and stage_complete_candidate must all judge only the current active_stage target.\n"
            "- Do not set active-stage fields true because observed_stage_id is a future stage.\n"
            "- If active_stage=0, active_stage_target_visible is only about stage 0 even if observed_stage_id is 2.\n"
            "- active_stage_target_centered means the active_stage target is near the image center or the selected waypoint clearly aims toward its center.\n"
            "- active_stage_target_close does NOT mean already touched the target or reached the final goal.\n"
            "- active_stage_target_close means the active_stage target has entered a near-before-stage-switch state.\n"
            "- Set active_stage_target_close=true if the active-stage target is the main forward structure, occupies clear image area, or is no longer a tiny distant landmark.\n"
            "- Set active_stage_target_close=true if the selected waypoint lies on the target building, its edge, its lower entrance area, or the street directly in front of it.\n"
            "- Set active_stage_target_close=true if continuing forward will likely enter the target area or pass the target.\n"
            "- Set active_stage_target_close=true if the target changed from distant direction cue into a nearby structure that may require turning, bypassing, or switching stage.\n"
            "- Visible alone is not close, but if the target dominates the forward view or the waypoint lands on/near it, do not keep returning close=false.\n"
            "- active_stage_target_passed means the active_stage target may already have been passed or current motion no longer clearly approaches it.\n"
            "- Set active_stage_target_passed=true if the target moved from front/center to side, partly left the main view, or the waypoint points beyond/beside it or toward the next area.\n"
            "- Set active_stage_target_passed=true if the image suggests the drone has passed the target facade/location or continuing forward moves away from it.\n"
            "- If uncertain, use false for visible/centered/close/passed and unclear for progress.\n"
            "- stage_progress must be relative to active_stage only and one of: approaching, near, passed, lost, unclear.\n"
            "- approaching: active_stage target is still mainly a direction cue, distant, or not clearly near yet.\n"
            "- near: active_stage target is a main visual structure, clearly larger/closer, or the waypoint lands on the target, its edge, lower area, or near-front street.\n"
            "- near also applies when continuing current motion will likely reach or pass the active-stage target.\n"
            "- passed: active_stage target appears already passed or behind/side-behind.\n"
            "- lost: active_stage target was likely visible/clear before but is currently lost.\n"
            "- unclear: not enough evidence.\n"
            "- stage_complete_candidate must refer only to active_stage.\n"
            "- Do not set stage_complete_candidate=true just because the target is visible.\n"
            "- Do not set stage_complete_candidate=true just because the target is centered but still distant.\n"
            "- Set stage_complete_candidate=true if active_stage_target_close=true and the waypoint is on or near the target building/area.\n"
            "- Set stage_complete_candidate=true if stage_progress=near and the active-stage target is the main image structure.\n"
            "- Set stage_complete_candidate=true if active_stage_target_passed=true.\n"
            "- Set stage_complete_candidate=true if the active-stage target is close enough that the next stage should likely begin soon.\n"
            "- If the active-stage target is only distant visible and centered, keep stage_complete_candidate=false."
        )
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
            json=payload,
            timeout=self.request_timeout,
        )
        if response.status_code >= 400 and "response_format" in payload:
            payload.pop("response_format", None)
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
                json=payload,
                timeout=self.request_timeout,
            )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _format_stage_plan(self, stage_plan):
        if not stage_plan:
            return "- no explicit stages"
        lines = []
        for stage in stage_plan:
            lines.append(
                "- {stage_id}: {stage_text} | landmark={landmark_phrase} | motion={motion_hint}".format(
                    stage_id=stage.get("stage_id", ""),
                    stage_text=stage.get("stage_text", ""),
                    landmark_phrase=stage.get("landmark_phrase", ""),
                    motion_hint=stage.get("motion_hint", ""),
                )
            )
        return "\n".join(lines)

    def _format_active_stage(self, stage_plan, active_stage):
        stage = self._get_active_stage(stage_plan, active_stage)
        if not stage:
            return (
                f"- stage_id: {active_stage}\n"
                "- stage_text: unknown\n"
                "- motion_hint: unknown\n"
                "- target_landmark: unknown"
            )
        return (
            f"- stage_id: {stage.get('stage_id', active_stage)}\n"
            f"- stage_text: {stage.get('stage_text', '')}\n"
            f"- motion_hint: {stage.get('motion_hint', '')}\n"
            f"- target_landmark: {stage.get('landmark_phrase', stage.get('target_landmark', ''))}"
        )

    def _get_active_stage(self, stage_plan, active_stage):
        if not stage_plan:
            return None
        for stage in stage_plan:
            if stage.get("stage_id") == active_stage:
                return stage
        if 0 <= active_stage < len(stage_plan):
            return stage_plan[active_stage]
        return None

    def _image_to_data_url(self, image):
        ok, buffer = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("failed to encode image")
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _parse_response(self, text):
        cleaned = text.strip()
        errors = []
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            errors.append(f"direct: {exc}")

        fence_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError as exc:
                errors.append(f"fenced: {exc}")

        json_text = self._extract_first_json_object(cleaned)
        if json_text:
            try:
                return json.loads(json_text)
            except json.JSONDecodeError as exc:
                errors.append(f"object: {exc}")

        raise ValueError("; ".join(errors) or "no JSON object found")

    def _extract_first_json_object(self, text):
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:idx + 1]
        return None

    def _to_bool(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)

    def _to_float(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _to_int(self, value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
