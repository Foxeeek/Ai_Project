"""Agent definitions for IdeaForge."""

from __future__ import annotations

import os
from typing import Type

from crewai import Agent
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


class TrendComplaintSearchInput(BaseModel):
    """Input schema for trend complaint search tool."""

    query: str = Field(
        ...,
        description="Search query targeting complaints and repetitive workflow pain points.",
        min_length=3,
    )
    max_results: int = Field(
        default=8,
        ge=1,
        le=25,
        description="Maximum number of search results to return.",
    )


class TrendComplaintSearchTool(BaseTool):
    """Search public web snippets for complaints and repetitive workflow pain points."""

    name: str = "trend_complaint_search"
    description: str = (
        "Search public web snippets for user complaints, repetitive manual tasks, and automation pain points."
    )
    args_schema: Type[BaseModel] = TrendComplaintSearchInput

    def _run(self, query: str, max_results: int = 8) -> str:
        results: list[str] = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                title = item.get("title", "").strip()
                body = item.get("body", "").strip()
                href = item.get("href", "").strip()
                results.append(f"- {title}: {body} ({href})")
        return "\n".join(results) if results else "No results found."


def _build_llm() -> ChatGoogleGenerativeAI:
    """Create a Gemini chat model for all agents."""
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def create_market_scout() -> Agent:
    """Researches trend pain points and repetitive task complaints."""
    return Agent(
        role="Market Scout",
        goal=(
            "Find real and recent complaints about repetitive manual workflows from "
            "freelancers, solo founders, small businesses, and crypto-native users."
        ),
        backstory=(
            "You are a deeply skeptical market intelligence researcher with a bias "
            "toward practical monetizable pain points. You cross-check claims, avoid "
            "hype, and focus on painful repetitive workflows users already pay to solve."
        ),
        llm=_build_llm(),
        tools=[TrendComplaintSearchTool()],
        verbose=True,
        allow_delegation=False,
    )


def create_solutions_architect() -> Agent:
    """Creates technical software ideas from researched pain points."""
    return Agent(
        role="Solutions Architect",
        goal=(
            "Transform validated pain points into 3 distinct low-complexity "
            "software solutions suitable for a solo backend developer."
        ),
        backstory=(
            "You are a principal engineer and startup architect. You specialize in "
            "lean systems that can be built quickly and monetized with minimal "
            "maintenance. You prioritize APIs, scripts, automations, browser tools, "
            "and workflow agents over UI-heavy products."
        ),
        llm=_build_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_strict_qa_engineer() -> Agent:
    """Performs ruthless quality filtering of proposed ideas."""
    return Agent(
        role="Strict QA Engineer",
        goal=(
            "Reject every idea that violates passive-income constraints and approve "
            "only deploy-and-forget digital products."
        ),
        backstory=(
            "You are an uncompromising gatekeeper. You reject by default. "
            "Any hint of physical logistics, customer service burden, complex UI, "
            "content moderation, legal fragility, fragile scraping dependencies, "
            "or >14-day solo build complexity is an automatic kill. "
            "Only approve ideas that are strictly backend-implementable, low-touch, "
            "digitally delivered, and maintainable with near-zero support."
        ),
        llm=_build_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_tech_lead() -> Agent:
    """Documents approved idea into technical implementation spec."""
    return Agent(
        role="Tech Lead",
        goal=(
            "Convert approved opportunities into concise technical specs with "
            "clear implementation steps."
        ),
        backstory=(
            "You are a hands-on tech lead known for converting vague concepts into "
            "build-ready specs. You write pragmatic architecture decisions and concise "
            "execution plans optimized for 14-day solo shipping."
        ),
        llm=_build_llm(),
        verbose=True,
        allow_delegation=False,
    )
