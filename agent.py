
import json
import pandas as pd

from google import genai
from google.genai import types

from agent_tools import run_analysis_tool


AGENT_TOOLS = [
    {
        "name": "get_dataset_overview",
        "description": (
            "Get the dataset row count, column names, data types, "
            "missing cells, and duplicate rows."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "calculate_numeric_summary",
        "description": (
            "Calculate sums, means, minimums, and maximums for "
            "numeric dataset columns."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "find_data_quality_issues",
        "description": (
            "Identify missing values, duplicate rows, and constant columns."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        },
    },
    {
        "name": "analyze_category",
        "description": (
            "Aggregate a numeric column by a category and return the "
            "highest totals. Use actual column names from the dataset."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category_column": {
                    "type": "STRING",
                    "description": "Existing category column name.",
                },
                "value_column": {
                    "type": "STRING",
                    "description": "Existing numeric column name.",
                },
                "top_n": {
                    "type": "INTEGER",
                    "description": "Number of categories to return, from 1 to 50.",
                },
            },
            "required": ["category_column", "value_column"],
        },
    },
]


def run_agent(
    question: str,
    df: pd.DataFrame,
    client: genai.Client,
    model_name: str,
    max_tool_rounds: int = 5,
) -> dict:
    """
    Let Gemini choose approved analysis tools, execute them in Python,
    and use the results to prepare a final answer.
    """
    if df is None:
        return {
            "answer": "Please upload a CSV or Excel dataset first.",
            "tool_runs": [],
        }

    if not question.strip():
        return {
            "answer": "Please enter a question.",
            "tool_runs": [],
        }

    column_info = {
        str(column): str(dtype)
        for column, dtype in df.dtypes.items()
    }

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text=(
                        "You are NexusAI, an analytical AI agent. "
                        "Use the approved tools whenever actual calculations "
                        "or dataset checks are needed. Never invent results. "
                        "Only use the supplied dataset columns. "
                        "Do not claim to have performed actions that you "
                        "did not perform.\n\n"
                        f"Dataset rows: {len(df)}\n"
                        f"Dataset columns and types: "
                        f"{json.dumps(column_info)}\n\n"
                        f"User request: {question}"
                    )
                )
            ],
        )
    ]

    tool_runs = []

    for _ in range(max_tool_rounds):
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name=tool["name"],
                                description=tool["description"],
                                parameters=tool["parameters"],
                            )
                            for tool in AGENT_TOOLS
                        ]
                    )
                ],
                temperature=0.2,
            ),
        )

        candidate = (
            response.candidates[0]
            if response.candidates
            else None
        )

        if candidate is None or candidate.content is None:
            return {
                "answer": "The AI agent did not return a usable response.",
                "tool_runs": tool_runs,
            }

        contents.append(candidate.content)

        function_calls = [
            part.function_call
            for part in candidate.content.parts
            if part.function_call is not None
        ]

        if not function_calls:
            return {
                "answer": response.text or "No answer was generated.",
                "tool_runs": tool_runs,
            }

        function_results = []

        for function_call in function_calls:
            name = function_call.name
            arguments = dict(function_call.args or {})

            result = run_analysis_tool(
                tool_name=name,
                df=df,
                arguments=arguments,
            )

            tool_runs.append({
                "tool": name,
                "arguments": arguments,
                "result": result,
            })

            function_results.append(
                types.Part.from_function_response(
                    name=name,
                    response={"result": result},
                )
            )

        contents.append(
            types.Content(
                role="user",
                parts=function_results,
            )
        )

    # If the tool-round limit is reached, ask for a concise summary
    # of the results collected so far without executing more tools.
    summary_prompt = (
        "Provide the best answer to the user's original request using "
        "only the available tool results. Do not claim that additional "
        "tools were run. If something remains unanswered, say so.\n\n"
        f"Original request: {question}\n\n"
        f"Tool results: {json.dumps(tool_runs, default=str)}"
    )

    final_response = client.models.generate_content(
        model=model_name,
        contents=summary_prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )

    return {
        "answer": final_response.text or "No final answer was generated.",
        "tool_runs": tool_runs,
    }
