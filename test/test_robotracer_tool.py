# test_tools.py
from spagent.tools.robotracer_tool import RoboTracerTool

# ===== 测试 RoboTracer =====
rt = RoboTracerTool(use_mock=True)      # mock
print("RoboTracer name:", rt.name)
print("RoboTracer parameters:", rt.parameters)

# 创建一个假图片用于测试
from pathlib import Path
Path("test_image.jpg").touch()           # Create empty folder

result = rt.call(
    image_path="test_image.jpg",
    instruction="pick up the red cup and place it on the shelf"
)
print("RoboTracer result:", result)
print("Success?", result["success"])
print()
