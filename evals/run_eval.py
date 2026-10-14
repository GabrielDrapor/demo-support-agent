"""
Run the golden-set evaluation against one version of the support agent.

    python evals/run_eval.py                                   # stable version (this checkout)
    python evals/run_eval.py --ref fault/f3-refund-limit-config
    python evals/run_eval.py --dataset-version 1               # publish and use version 1 only

A `--ref` other than the current checkout is put into a git worktree under
`.worktrees/`, so the run uses that version's agent code. The run's candidate
version is the branch name and its commit.
"""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import importlib
import importlib.util
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cases  # noqa: E402
import traceroot  # noqa: E402
from traceroot import Integration, Scorer, evaluate  # noqa: E402

from run_demo import checkout_fault  # noqa: E402

traceroot.initialize(integrations=[Integration.OPENAI])

JUDGE_MODEL = "gpt-6-sol"


def load_agent(ref: str | None):
    """Import support_agent and shop_data from `ref`, or from this checkout."""
    path = ROOT if ref is None else checkout_fault(ref)
    sys.path.insert(0, str(path))
    for name in ("shop_data", "support_agent"):
        sys.modules.pop(name, None)
    agent = importlib.import_module("support_agent")
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=path, capture_output=True, text=True
    ).stdout.strip()
    return agent, f"{ref or 'main'}@{sha}"


_calls: contextvars.ContextVar[dict[str, list[str]]] = contextvars.ContextVar("calls")


def _record(kind: str, name: str, func):
    """Wrap `func` so each call adds `name` to the current case's record."""

    def wrapper(*args, **kwargs):
        with contextlib.suppress(LookupError):
            _calls.get()[kind].append(name)
        return func(*args, **kwargs)

    return wrapper


def make_task(agent):
    """Return the eval task. Cases run at the same time, so each case keeps its own record."""
    agent.TOOL_FUNCS.update({n: _record("tools", n, f) for n, f in agent.TOOL_FUNCS.items()})
    agent.AGENTS.update({n: _record("agents", n, f) for n, f in agent.AGENTS.items()})

    def run_case(input: dict) -> dict:
        """Answer one customer message, and record which tools and agents ran."""
        calls = {"tools": [], "agents": []}
        _calls.set(calls)
        answer = agent.handle_message(
            input["customer_id"], f"eval-{uuid.uuid4().hex[:8]}", [], input["message"], "eval"
        )
        return {"answer": answer, **calls}

    return run_case


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


def expected_facts(input, output, expected=None, metadata=None) -> bool:
    """The answer contains at least one phrase from each expected group."""
    answer = output["answer"].lower()
    return all(any(p in answer for p in group) for group in expected["any_of_each"])


def required_tools(input, output, expected=None, metadata=None) -> bool:
    """The agent called every tool that the case needs."""
    return set(metadata["required_tools"]) <= set(output["tools"])


def no_handoff_loop(input, output, expected=None, metadata=None) -> bool:
    """No sub-agent handled the message more than once."""
    return len(output["agents"]) == len(set(output["agents"]))


def _stable_shop_data():
    """Load shop_data from this checkout, so the judge grades every version by the stable policy."""
    spec = importlib.util.spec_from_file_location("stable_shop_data", ROOT / "shop_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy_judge():
    shop = _stable_shop_data()
    policies = "\n".join(f"- {a['title']}: {a['body']}" for a in shop.ARTICLES)

    @Scorer.llm_judge(
        name="follows_policy",
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You grade answers from the support agent of Acme Outfitters, "
                "an outdoor gear shop. The shop policies are:\n"
                f"{policies}\n"
                "Facts about the customer's own orders come from shop tools and are correct. "
                "Reply 0.0 only if the answer states a shop policy that disagrees with these "
                "policies or that they do not contain. An answer with no policy statement gets "
                "1.0. Reply with the number only.",
            },
            {
                "role": "user",
                "content": "CUSTOMER TIER: {{tier}}\n\nCUSTOMER:\n{{message}}\n\nANSWER:\n{{answer}}",
            },
        ],
        value_type="numeric",
        threshold=1.0,
    )
    def follows_policy(ctx):
        tier = shop.CUSTOMERS[ctx.input["customer_id"]]["tier"]
        return {"tier": tier, "message": ctx.input["message"], "answer": ctx.output["answer"]}

    return follows_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref", default=None, help="branch to evaluate, e.g. fault/f1-us-date-format"
    )
    parser.add_argument("--dataset-version", type=int, default=2, choices=[1, 2])
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--local", action="store_true", help="run without reporting to TraceRoot")
    args = parser.parse_args()

    agent, candidate = load_agent(args.ref)
    with contextlib.suppress(AttributeError):
        agent.warm_customer_cache()
    result = evaluate(
        name="Support agent golden set",
        evaluation_key="support-golden",
        dataset=cases.build(args.dataset_version),
        task=make_task(agent),
        scorers=[expected_facts, required_tools, no_handoff_loop, policy_judge()],
        candidate_version=candidate,
        max_concurrency=args.concurrency,
        local=args.local,
    )
    print(result.summary())
    for item in result.item_results:
        failed = [s.name for s in item.scores if s.value in (False, 0)]
        if failed or item.error:
            answer = (item.output or {}).get("answer", "")[:160].replace("\n", " ")
            agents = (item.output or {}).get("agents")
            print(f"  FAIL {item.case_id}: {', '.join(failed) or item.error} | {agents} | {answer}")
    traceroot.flush()


if __name__ == "__main__":
    main()
