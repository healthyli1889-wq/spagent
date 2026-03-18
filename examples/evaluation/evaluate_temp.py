from pathlib import Path

from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import RoboTracerTool, VGGTTool


def prepare_test_images():
    """VGGT: 3D education building. RoboTracer: motion trajectory (desk scene)."""
    img_dir = Path("assets")
    img_dir.mkdir(exist_ok=True)
    vggt_img = img_dir / "vggt_education.jpg"
    robotracer_img = img_dir / "robotracer_desk.jpg"
    for p in (vggt_img, robotracer_img):
        if not p.exists():
            raise FileNotFoundError(f"Download the image first: {p}")
    return [str(vggt_img), str(robotracer_img)]


def main():
    # 1. Define model 
    model = GPTModel(model_name="gpt-4o-mini")
    # 2. Define tools
    tools = [
        RoboTracerTool(use_mock=True),
        VGGTTool(use_mock=True),
    ]
    # 3. Create Agent
    agent = SPAgent(model=model, tools=tools)
    print("Registered tools:", agent.list_tools())
    # 4. Prepare test data 
    # Image 1: 3D education building (for VGGT). Image 2: desk scene (for RoboTracer)
    img_paths = prepare_test_images()
    question = (
        "You have two images: (1) a 3D education building, (2) a desk scene with objects. "
        "Use the VGGT tool to estimate 3D reconstruction from the building image, "
        "and use the RoboTracer tool to estimate the motion trajectory for the desk scene. "
        "Then summarize what you found."
    )
    # 5. Call Agent to solve problem
    result = agent.solve_problem(
        img_paths,
        question,
        max_iterations=5,
    )
    # 6. Print result - check if the model called the tools
    print("\n=== EVALUATE TEMP RESULT ===")
    print("Answer:", result.get("answer"))
    print("Used tools:", result.get("used_tools"))
    print("Tool calls:", result.get("tool_calls"))
    print("Additional images:", result.get("additional_images"))


if __name__ == "__main__":
    main()
