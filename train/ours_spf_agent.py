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
    def __init__(self, base_url, model_name, output_dir):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.output_dir = output_dir
        self.image_dir = os.path.join(output_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

    def act(self, image, instruction, sample_id=None, step_id=None, pose=None, history=None):
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
            raw_response = self._call_vlm(input_overlay, instruction, candidates)
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

    def _call_vlm(self, image, instruction, candidates):
        image_url = self._image_to_data_url(image)
        candidate_lines = "\n".join(
            f'- {candidate["id"]}: [y, x] = {candidate["point"]}'
            for candidate in candidates
        )
        system_prompt = (
            "You are a waypoint selection module. You must return only valid JSON. "
            "No explanations. No markdown. No reasoning."
        )
        prompt = (
            "Return ONLY one valid JSON object.\n"
            "Do not explain. Do not use markdown. Do not include ```json code block.\n"
            "Do not include task breakdown. Do not include reasoning.\n"
            "Do not think step by step. Do not output analysis. Output final JSON only.\n"
            "The first character must be { and the last character must be }.\n\n"
            "The image contains labeled candidate waypoints P1 to P15.\n"
            f"Task: {instruction}\n\n"
            "Choose exactly one candidate that best moves the drone toward the task.\n"
            "Output format, exactly one object:\n"
            "{\"choice\":\"P8\",\"target_visible\":true,\"confidence\":0.9,\"reason\":\"brief\"}\n\n"
            f"Valid choices:\n{candidate_lines}\n\n"
            "Rules:\n"
            "- choice must be one of P1 through P15.\n"
            "- target_visible must be true or false.\n"
            "- confidence must be a number from 0.0 to 1.0.\n"
            "- Keep reason under 12 words."
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
            timeout=60,
        )
        if response.status_code >= 400 and "response_format" in payload:
            payload.pop("response_format", None)
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

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
