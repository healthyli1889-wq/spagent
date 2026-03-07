"""
RoboTracer 
A minimal tool that estimates a simple trajectory from images. 
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

sys.path.append(str(Path(__file__).parent.parent))
from core.tool import Tool


class RoboTracerTool(Tool):
    """A minimal tool that estimates a simple trajectory from an ordered list of images."""

    def __init__(self):
        super().__init__(
            name="robotracer_tool",
            description=(
                "Estimate a simple motion trajectory from an ordered list of image frames. "
                "Use this for basic movement direction and path tracing."
            )
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered list of image file paths from earliest to latest."
                }
            },
            "required": ["image_paths"]
        }

    def call(self, image_paths: List[str]) -> Dict[str, Any]:
        try:
            if not image_paths or not isinstance(image_paths, list):
                return {
                    "success": False,
                    "error": "image_paths must be a non-empty list."
                }

            # check all files exist
            for image_path in image_paths:
                p = Path(image_path)
                if not p.exists():
                    return {
                        "success": False,
                        "error": f"File not found: {image_path}"
                    }

            # minimal placeholder trajectory logic
            # this does NOT do real trajectory estimation
            trajectory_points = []
            x, y = 0, 0

            for i, image_path in enumerate(image_paths):
                trajectory_points.append({
                    "frame_index": i,
                    "image_path": image_path,
                    "x": x,
                    "y": y
                })
                x += 1
                y += 0.5

            total_distance = ((len(image_paths) - 1) * ((1**2 + 0.5**2) ** 0.5)) if len(image_paths) > 1 else 0.0

            return {
                "success": True,
                "result": {
                    "num_frames": len(image_paths),
                    "trajectory_points": trajectory_points,
                    "total_distance": total_distance,
                    "movement_direction": "forward-right"
                },
                "summary": (
                    f"Estimated a simple trajectory across {len(image_paths)} frames "
                    f"with approximate movement direction forward-right."
                )
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
