"""Entry point for IdeaForge autonomous ideation pipeline."""

from __future__ import annotations

import json
from typing import Any

from crewai import Crew, Process
from dotenv import load_dotenv

from agents import (
    create_market_scout,
    create_solutions_architect,
    create_strict_qa_engineer,
    create_tech_lead,
)
from database import IdeaDatabase, IdeaRecord
from tasks import (
    QAResult,
    TechSpec,
    create_qa_task,
    create_scout_task,
    create_solutions_task,
    create_tech_spec_task,
)


def _extract_pydantic(task_output: Any, model_cls):
    if task_output is None:
        raise ValueError(f"No output found for {model_cls.__name__}")

    if hasattr(task_output, "pydantic") and task_output.pydantic is not None:
        return task_output.pydantic

    raw = getattr(task_output, "raw", None)
    if raw is None:
        raise ValueError(f"Task output for {model_cls.__name__} has no raw content")

    if isinstance(raw, dict):
        return model_cls.model_validate(raw)

    if isinstance(raw, str):
        return model_cls.model_validate_json(raw)

    raise ValueError(f"Unsupported output type for {model_cls.__name__}: {type(raw)}")


def _render_markdown_spec(spec: TechSpec) -> str:
    stack = "\n".join(f"- {item}" for item in spec.tech_stack)
    plan = "\n".join(f"{idx}. {step}" for idx, step in enumerate(spec.mvp_3_step_execution_plan, start=1))
    return (
        f"# {spec.name}\n\n"
        f"## Target Audience\n{spec.target_audience}\n\n"
        f"## Tech Stack\n{stack}\n\n"
        f"## 3-Step MVP Execution Plan\n{plan}\n\n"
        f"## Operations Mode\n{spec.operations_mode}\n"
    )


def run() -> None:
    load_dotenv()

    market_scout = create_market_scout()
    solutions_architect = create_solutions_architect()
    strict_qa = create_strict_qa_engineer()
    tech_lead = create_tech_lead()

    scout_task = create_scout_task(market_scout)
    solutions_task = create_solutions_task(solutions_architect, context=[scout_task])
    qa_task = create_qa_task(strict_qa, context=[solutions_task])
    spec_task = create_tech_spec_task(tech_lead, context=[solutions_task, qa_task])

    crew = Crew(
        agents=[market_scout, solutions_architect, strict_qa, tech_lead],
        tasks=[scout_task, solutions_task, qa_task, spec_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    outputs = getattr(result, "tasks_output", None)
    if not outputs or len(outputs) < 4:
        raise RuntimeError("Crew execution completed without expected task outputs")

    qa_result: QAResult = _extract_pydantic(outputs[2], QAResult)
    tech_spec: TechSpec = _extract_pydantic(outputs[3], TechSpec)

    print("\n=== QA VERDICT ===")
    print(json.dumps(qa_result.model_dump(), indent=2))

    if qa_result.approved:
        db = IdeaDatabase()
        markdown_spec = _render_markdown_spec(tech_spec)
        row_id = db.save_approved_idea(
            IdeaRecord(
                idea_name=tech_spec.name,
                target_audience=tech_spec.target_audience,
                tech_stack=", ".join(tech_spec.tech_stack),
                markdown_spec=markdown_spec,
            )
        )
        print(f"\nApproved idea saved to database with row id: {row_id}")
    else:
        print("\nNo idea approved by strict QA. Nothing saved to database.")


if __name__ == "__main__":
    run()
