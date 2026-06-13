# Copyright (c) Microsoft. All rights reserved.

"""Run high-signal Agent Framework profiling scenarios with optional NVTX ranges.

This script is intentionally self-contained so it can be copied to or run from an
HPC checkout without relying on sample-specific Azure/Foundry configuration.
It uses the OpenAI-compatible Chat Completions client, which works with
OpenRouter when configured with:

    AF_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
    AF_OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b
    OPENROUTER_API_KEY=...

Set MAF_NVTX_ENABLE=1 to enable the framework NVTX ranges added in core.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import time
from collections.abc import Awaitable, Callable, Iterator
from typing import Literal

from typing_extensions import Never

from agent_framework import (
    Agent,
    AgentResponseUpdate,
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
    tool,
)
from agent_framework.openai import OpenAIChatCompletionClient

ScenarioName = Literal["baseline", "streaming", "tools", "agent_workflow", "local_fanout"]


def _env(name: str, fallback: str | None = None) -> str:
    value = os.environ.get(name, fallback)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def _openrouter_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if referer := os.environ.get("AF_OPENROUTER_REFERER"):
        headers["HTTP-Referer"] = referer
    if title := os.environ.get("AF_OPENROUTER_TITLE"):
        headers["X-OpenRouter-Title"] = title
    return headers


def make_client() -> OpenAIChatCompletionClient:
    """Create an OpenAI-compatible client configured for OpenRouter by default."""
    return OpenAIChatCompletionClient(
        model=_env("AF_OPENROUTER_MODEL", os.environ.get("OPENAI_MODEL")),
        api_key=_env("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY")),
        base_url=os.environ.get(
            "AF_OPENROUTER_BASE_URL",
            os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        ),
        default_headers=_openrouter_headers() or None,
    )


@contextlib.contextmanager
def scenario_range(name: str) -> Iterator[None]:
    """Add a coarse top-level scenario range when nvtx is installed and enabled."""
    if os.environ.get("MAF_NVTX_ENABLE", "").lower() not in {"1", "true", "yes"}:
        yield
        return

    try:
        import nvtx
    except ImportError:
        yield
        return

    with nvtx.annotate(message=f"af_suite.scenario:{name}"):
        yield


@tool(approval_mode="never_require")
def lookup_ticket(ticket_id: str) -> str:
    """Look up an incident ticket in the synthetic support system."""
    time.sleep(0.15)
    return f"{ticket_id}: intermittent timeout in tool dispatcher; 3 recent retries."


@tool(approval_mode="never_require")
def search_runtime_notes(query: str) -> str:
    """Search synthetic runtime notes."""
    time.sleep(0.20)
    return f"notes for {query}: inspect agent loop, tool loop, middleware, and workflow delivery latency."


@tool(approval_mode="never_require")
def estimate_priority(retries: int, affected_users: int) -> str:
    """Estimate a synthetic operational priority score."""
    time.sleep(0.10)
    score = retries * 2 + affected_users // 100
    return f"priority_score={score}; recommendation={'urgent' if score > 8 else 'normal'}"


async def baseline(args: argparse.Namespace) -> None:
    """Single non-streaming agent call: clean agent/model lifecycle baseline."""
    agent = Agent(
        client=make_client(),
        name="baseline",
        instructions="Answer concisely. Avoid tool calls.",
    )
    result = await agent.run(args.prompt or "Explain one runtime bottleneck in agent frameworks in 4 bullets.")
    print(result.text[: args.max_output_chars])


async def streaming(args: argparse.Namespace) -> None:
    """Single streaming agent call: stream pull and finalization ranges."""
    agent = Agent(
        client=make_client(),
        name="streamer",
        instructions="Stream a concise technical analysis.",
    )
    stream = agent.run(
        args.prompt or "Describe the lifecycle of an agentic tool call in a runtime.",
        stream=True,
    )
    async for update in stream:
        print(update.text, end="", flush=True)
    final = await stream.get_final_response()
    print(f"\n[final chars] {len(final.text)}")


async def tools(args: argparse.Namespace) -> None:
    """Tool-calling agent: function loop, model round trips, and tool invocation."""
    agent = Agent(
        client=make_client(),
        name="tool_stress",
        instructions=(
            "You are a runtime diagnostics agent. You must call the available tools before answering. "
            "Call lookup_ticket, search_runtime_notes, and estimate_priority when the inputs are available. "
            "After using tools, summarize root cause and next action."
        ),
        tools=[lookup_ticket, search_runtime_notes, estimate_priority],
    )
    result = await agent.run(
        args.prompt
        or (
            "Investigate ticket AF-742. It has 4 retries and affects 900 users. "
            "Use all tools, then summarize root cause and next action."
        )
    )
    print(result.text[: args.max_output_chars])


async def agent_workflow(args: argparse.Namespace) -> None:
    """Three-agent workflow chain: workflow + agent executor + streaming output."""
    writer = Agent(
        client=make_client(),
        name="writer",
        instructions="Draft a compact incident report from the user's request.",
    )
    reviewer = Agent(
        client=make_client(),
        name="reviewer",
        instructions="Review and improve the incident report. Be concise.",
    )
    planner = Agent(
        client=make_client(),
        name="planner",
        instructions="Turn the report into 3 prioritized engineering actions.",
    )

    workflow = WorkflowBuilder(start_executor=writer).add_edge(writer, reviewer).add_edge(reviewer, planner).build()
    stream = workflow.run(args.prompt or "Runtime latency spikes during multi-tool agent runs.", stream=True)

    async for event in stream:
        if event.type == "output" and isinstance(event.data, AgentResponseUpdate):
            print(event.data.text, end="", flush=True)
    print()


class Dispatch(Executor):
    """Dispatch one message to all workers."""

    @handler
    async def dispatch(self, message: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(message)


class Worker(Executor):
    """CPU-bound local worker for workflow routing and executor timing."""

    def __init__(self, *, executor_id: str, work_factor: int) -> None:
        super().__init__(id=executor_id)
        self._work_factor = work_factor

    @handler
    async def work(self, message: str, ctx: WorkflowContext[str]) -> None:
        total = 0
        for i in range(self._work_factor):
            total += (i * 17) % 23
        await ctx.send_message(f"{self.id}: processed {len(message)} chars total={total}")


class Join(Executor):
    """Fan-in executor that yields the joined worker outputs."""

    @handler
    async def join(self, results: list[str], ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(" | ".join(sorted(results)))


async def local_fanout(args: argparse.Namespace) -> None:
    """Model-free workflow: pure workflow routing/executor/fan-in signal."""
    dispatch = Dispatch(id="dispatch")
    workers = [Worker(executor_id=f"worker_{i}", work_factor=args.work_factor) for i in range(args.workers)]
    join = Join(id="join")
    workflow = (
        WorkflowBuilder(start_executor=dispatch)
        .add_fan_out_edges(dispatch, workers)
        .add_fan_in_edges(workers, join)
        .build()
    )

    async for event in workflow.run(args.prompt or "stress workflow routing and executor delivery", stream=True):
        if event.type == "output":
            print(event.data)


SCENARIOS: dict[ScenarioName, Callable[[argparse.Namespace], Awaitable[None]]] = {
    "baseline": baseline,
    "streaming": streaming,
    "tools": tools,
    "agent_workflow": agent_workflow,
    "local_fanout": local_fanout,
}


async def run_scenario(name: ScenarioName, args: argparse.Namespace) -> None:
    with scenario_range(name):
        await SCENARIOS[name](args)


async def amain() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=[*SCENARIOS.keys(), "all"],
        default="baseline",
        help="Scenario to run.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the selected scenario this many times.")
    parser.add_argument("--workers", type=int, default=4, help="Worker count for local_fanout.")
    parser.add_argument("--work-factor", type=int, default=750_000, help="CPU loop iterations per local_fanout worker.")
    parser.add_argument("--prompt", default=None, help="Optional prompt override for the selected scenario.")
    parser.add_argument("--max-output-chars", type=int, default=1600, help="Truncate non-streaming textual output.")
    args = parser.parse_args()

    scenario_names: list[ScenarioName]
    if args.scenario == "all":
        scenario_names = list(SCENARIOS.keys())
    else:
        scenario_names = [args.scenario]

    for iteration in range(args.repeat):
        if args.repeat > 1:
            print(f"\n=== repeat {iteration + 1}/{args.repeat} ===")
        for name in scenario_names:
            print(f"\n=== scenario: {name} ===")
            await run_scenario(name, args)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
