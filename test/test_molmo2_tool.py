"""
Mock-based tests for Molmo2Tool.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from spagent.tools import Molmo2Tool


ASSET_IMAGE = str(project_root / "assets" / "dog.jpeg")


def test_molmo2_mock_qa():
    tool = Molmo2Tool(use_mock=True)
    result = tool.call(
        image_path=ASSET_IMAGE,
        task="qa",
        prompt="What is shown in this image?",
    )

    assert result["success"] is True
    assert "generated_text" in result["result"]
    assert result["result"]["task"] == "qa"


def test_molmo2_mock_point(tmp_path):
    tool = Molmo2Tool(use_mock=True, output_dir=str(tmp_path))
    result = tool.call(
        image_path=ASSET_IMAGE,
        task="point",
        prompt="Point to the dog.",
        save_annotated=True,
    )

    assert result["success"] is True
    assert result["result"]["task"] == "point"
    assert result["result"]["num_points"] >= 1
    assert result["output_path"] is not None
