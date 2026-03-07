from typing import Dict, Any, List
import math


class MockRoboTracerService:
    """
    Lightweight mock trajectory estimator for development and testing.

    It does NOT perform real visual odometry.
    It fabricates a stable trajectory based on frame count so that
    the SPAgent tool-calling pipeline can be verified end-to-end.
    """

    def infer(
        self,
        image_paths: List[str],
        coordinate_mode: str = "relative_2d",
        return_summary_only: bool = False,
    ) -> Dict[str, Any]:
        if not image_paths:
            return {"success": False, "error": "No image_paths provided."}

        num_frames = len(image_paths)

        trajectory_points = []
        x, y = 0.0, 0.0
        step_x, step_y = 1.0, 0.4

        for i, img_path in enumerate(image_paths):
            point = {
                "frame_index": i,
                "image_path": img_path,
                "x": round(x, 4),
                "y": round(y, 4),
            }
            trajectory_points.append(point)
            x += step_x
            y += step_y

        total_distance = 0.0
        for i in range(1, len(trajectory_points)):
            dx = trajectory_points[i]["x"] - trajectory_points[i - 1]["x"]
            dy = trajectory_points[i]["y"] - trajectory_points[i - 1]["y"]
            total_distance += math.sqrt(dx * dx + dy * dy)

        start = trajectory_points[0]
        end = trajectory_points[-1]

        movement_direction = self._describe_direction(
            dx=end["x"] - start["x"],
            dy=end["y"] - start["y"],
        )

        summary = (
            f"Estimated trajectory across {num_frames} frames. "
            f"Start=({start['x']}, {start['y']}), end=({end['x']}, {end['y']}). "
            f"Approximate movement: {movement_direction}. "
            f"Estimated total path length: {round(total_distance, 3)}."
        )

        result = {
            "success": True,
            "num_frames": num_frames,
            "coordinate_mode": coordinate_mode,
            "trajectory_points": [] if return_summary_only else trajectory_points,
            "start_point": {"x": start["x"], "y": start["y"]},
            "end_point": {"x": end["x"], "y": end["y"]},
            "movement_direction": movement_direction,
            "total_distance": round(total_distance, 4),
            "summary": summary,
            "output_path": None,
        }

        return result

    def trace(
        self,
        image_paths: List[str],
        coordinate_mode: str = "relative_2d",
        return_summary_only: bool = False,
    ) -> Dict[str, Any]:
        return self.infer(
            image_paths=image_paths,
            coordinate_mode=coordinate_mode,
            return_summary_only=return_summary_only,
        )

    def _describe_direction(self, dx: float, dy: float) -> str:
        horizontal = ""
        vertical = ""

        if dx > 0.2:
            horizontal = "right"
        elif dx < -0.2:
            horizontal = "left"

        if dy > 0.2:
            vertical = "forward/up"
        elif dy < -0.2:
            vertical = "backward/down"

        if horizontal and vertical:
            return f"{vertical} and {horizontal}"
        if horizontal:
            return horizontal
        if vertical:
            return vertical
        return "mostly stationary"
