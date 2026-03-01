"""Task definitions for IdeaForge crew execution."""

from __future__ import annotations

from crewai import Task
from pydantic import BaseModel, Field


class PainPoint(BaseModel):
    audience: str = Field(description="Specific user segment experiencing pain")
    complaint: str = Field(description="Real complaint in plain language")
    repetitive_task: str = Field(description="Manual recurring task they struggle with")
    source_hint: str = Field(description="Where this signal was observed")


class ScoutReport(BaseModel):
    findings: list[PainPoint] = Field(min_length=5, max_length=8)


class ProductIdea(BaseModel):
    name: str
    one_line_summary: str
    target_user: str
    delivery_model: str
    core_automation: str
    build_scope_days: int
    monetization: str
    risks: list[str]


class SolutionsReport(BaseModel):
    ideas: list[ProductIdea] = Field(min_length=3, max_length=3)


class QAResult(BaseModel):
    approved: bool
    selected_idea_name: str
    rejection_reasons: list[str]
    strict_checks: dict[str, bool]


class TechSpec(BaseModel):
    name: str
    target_audience: str
    tech_stack: list[str]
    mvp_3_step_execution_plan: list[str] = Field(min_length=3, max_length=3)
    operations_mode: str


def create_scout_task(agent) -> Task:
    """Task for identifying repeated complaints and opportunities."""
    return Task(
        description=(
            "Perform trend scouting for automation opportunities from forums, X/Twitter-style "
            "public discussions, indie hacker communities, freelancer pain points, crypto ops, "
            "and small-business workflows.\n"
            "\n"
            "Rules:\n"
            "1) Focus ONLY on repetitive manual digital tasks.\n"
            "2) Exclude ideas that imply physical inventory, shipping, or local services.\n"
            "3) Gather 5-8 pain points with audience, complaint, repetitive task, and source hint.\n"
            "4) Prefer pain points that suggest recurring willingness to pay.\n"
            "5) Keep findings concrete and implementation-oriented."
        ),
        expected_output="Structured report of validated pain points.",
        output_pydantic=ScoutReport,
        agent=agent,
    )


def create_solutions_task(agent, context: list[Task]) -> Task:
    """Task for generating constrained software solutions."""
    return Task(
        description=(
            "Using the scout report, generate exactly 3 software product ideas.\n"
            "\n"
            "Hard constraints for each idea:\n"
            "- Fully digital and backend-centric.\n"
            "- Can be built by one skilled backend developer in <=14 days.\n"
            "- Must avoid complex custom UI requirements.\n"
            "- Should run mostly autonomously after deployment.\n"
            "- Should target B2B or prosumer willingness-to-pay.\n"
            "\n"
            "For each idea include: name, one-line summary, target user, delivery model, "
            "core automation logic, realistic build days estimate, monetization model, and risks."
        ),
        expected_output="Three constrained solution concepts.",
        output_pydantic=SolutionsReport,
        context=context,
        agent=agent,
    )


def create_qa_task(agent, context: list[Task]) -> Task:
    """Task for strict acceptance/rejection validation."""
    return Task(
        description=(
            "RUTHLESS QA FILTER. Evaluate all 3 ideas and choose ONLY one idea to approve, "
            "or approve none if all fail.\n"
            "\n"
            "Automatic rejection triggers:\n"
            "- Physical goods, shipping, field operations, or logistics\n"
            "- Customer support call dependency, onboarding-heavy operations\n"
            "- Complex UI/UX, mobile-first polished interfaces, or many frontend states\n"
            "- Heavy moderation, community management, legal review, or trust & safety overhead\n"
            "- Build complexity likely >14 days for solo backend developer\n"
            "- High ongoing maintenance due to unstable integrations\n"
            "\n"
            "You MUST output strict_checks booleans for these criteria:\n"
            "digital_only, low_support, backend_buildable, <=14_day_scope, low_moderation, "
            "low_maintenance.\n"
            "\n"
            "If no idea passes, set approved=false and selected_idea_name='NONE'."
        ),
        expected_output="Approval verdict with explicit kill reasons and strict checks.",
        output_pydantic=QAResult,
        context=context,
        agent=agent,
    )


def create_tech_spec_task(agent, context: list[Task]) -> Task:
    """Task for final technical specification documentation."""
    return Task(
        description=(
            "Create a concise technical specification ONLY for the QA-approved idea.\n"
            "If no idea was approved, output a spec named 'NO_APPROVED_IDEA' with exactly "
            "three steps describing how to refine input constraints and rerun discovery.\n"
            "\n"
            "Spec format fields:\n"
            "- Name\n"
            "- Target Audience\n"
            "- Tech Stack (pragmatic, minimal)\n"
            "- 3-Step MVP Execution Plan\n"
            "- Operations Mode (why this is deploy-and-forget)"
        ),
        expected_output="Structured, implementation-ready technical specification.",
        output_pydantic=TechSpec,
        context=context,
        agent=agent,
    )
