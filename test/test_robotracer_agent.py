from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import RoboTracerTool


def main():
    model = GPTModel(model_name="gpt-4o-mini")
    tools = [RoboTracerTool(use_mock=True)]

    agent = SPAgent(model=model, tools=tools)

    print("Registered tools:", agent.list_tools())

    result = agent.solve_problem(
        ["assets/dog.jpeg", "assets/dog.jpeg", "assets/dog.jpeg"],
        "Estimate the movement trajectory across these frames. Use the available tool if needed."
    )

    print(result)
    print("Answer:", result.get("answer"))
    print("Used tools:", result.get("used_tools"))
    print("Additional images:", result.get("additional_images"))


if __name__ == "__main__":
    main()
