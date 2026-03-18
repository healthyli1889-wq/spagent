"""
Tests for WanTool — Alibaba Wan video generation.

Usage:
  # Mock test (no API key needed)
  python test/wan/test_wan_tool.py --use_mock

  # Real API test (requires DASHSCOPE_API_KEY)
  python test/wan/test_wan_tool.py

  # Image-to-video with real API
  python test/wan/test_wan_tool.py --image_path path/to/image.png

  # Custom duration / aspect ratio
  python test/wan/test_wan_tool.py --use_mock --duration 8 --aspect_ratio 9:16
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "spagent"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test WanTool")
    parser.add_argument("--use_mock", action="store_true", default=False,
                        help="Use mock service instead of real API")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Optional image path for image-to-video")
    parser.add_argument("--duration", type=int, default=5,
                        help="Video duration in seconds (3-10)")
    parser.add_argument("--aspect_ratio", type=str, default="16:9",
                        choices=["16:9", "9:16", "1:1"],
                        help="Video aspect ratio")
    parser.add_argument("--model", type=str, default="wanx2.1-t2v-turbo",
                        help="Wan model name")
    args = parser.parse_args()

    from tools.wan_tool import WanTool

    tool = WanTool(use_mock=args.use_mock, model=args.model)
    logger.info(f"WanTool initialized (mock={args.use_mock})")

    prompt = (
        "A serene aerial shot of a mountain lake at sunrise, "
        "mist rising from the water, golden light on snow-capped peaks."
    )

    logger.info(f"Prompt: {prompt}")
    logger.info(f"Duration: {args.duration}s | Aspect ratio: {args.aspect_ratio}")
    if args.image_path:
        logger.info(f"Image: {args.image_path}")

    result = tool.call(
        prompt=prompt,
        image_path=args.image_path,
        duration=args.duration,
        aspect_ratio=args.aspect_ratio,
    )

    if result["success"]:
        logger.info(f"[PASS] Video generated: {result['output_path']}")
        if result.get("result", {}).get("mock"):
            logger.info("  (mock output)")
    else:
        logger.error(f"[FAIL] {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
