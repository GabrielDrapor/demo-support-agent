# Acme Outfitters support agent

Traffic source for the public TraceRoot demo project. A support agent for a
fake outdoor gear shop answers synthetic customers. The traffic driver plays
the customers.

This code is the source that TraceRoot shows in the demo project. Root cause
analysis and "Show code" links in demo traces point to commits in this repo.

## What it shows

- A router span (`gpt-6-luna`) sends each message to one of three sub-agents.
- Sub-agents (`gpt-6-luna`, reasoning off) run a tool loop. Tools from one model step run in
  parallel, so the timeline shows overlapping spans.
- The orders agent and the refunds agent can hand a conversation to each other.
- Refunds above $200 go to a supervisor review with `gpt-6.1-sol`, a reasoning model.
- Each session is one customer conversation with one to three turns. The
  customer ID is the user ID.
- Unknown order IDs make a tool fail, so some traces contain errors.

## Fault versions

`main` is the stable version. Each `fault/*` branch is `main` plus one commit
that adds a bug. The demo project runs a small share of its traffic on these
branches, so TraceRoot detectors and alerts have real problems to find, and
root cause analysis can point to the commit that caused them.

| Branch | What goes wrong |
|---|---|
| `fault/f1-us-date-format` | `list_orders` parses dates as MM/DD/YYYY, but the model sends YYYY-MM-DD, so the tool fails. |
| `fault/f2-draft-filter` | Help-center search keeps only articles with `status: published`, but no article has that field, so search silently returns nothing and the agents cannot answer policy questions. |
| `fault/f3-refund-limit-config` | The refund limit default is $2000 instead of $200, so large refunds skip supervisor review. |
| `fault/f4-no-customer-cache` | The customer profile cache is removed, so every message waits about 2 s for the slow CRM. |
| `fault/f5-bigger-agent-model` | Sub-agents use `gpt-6-sol` instead of `gpt-6-luna`, so each trace costs about twenty times more. |
| `fault/f6-chained-handoffs` | Agents can pass a handoff back, so two-part requests bounce between the orders and refunds agents. |

## Run

```bash
export TRACEROOT_API_KEY=tr-...            # key of the demo project
export TRACEROOT_HOST_URL=https://app.traceroot.ai   # or your own TraceRoot host
export TRACEROOT_ENVIRONMENT=production
export OPENAI_API_KEY=sk-...

# Stable version only
uv run --no-project --python 3.13 --with-requirements requirements.txt \
  python traffic_driver.py --sessions 40 --concurrency 6

# Stable version plus faults F1 and F3, with 15% of sessions on the faults
uv run --no-project --python 3.13 --with-requirements requirements.txt \
  python run_demo.py --sessions 40 --faults f1,f3 --fault-share 0.15
```

`run_demo.py` checks out each fault branch into `.worktrees/` and runs the
traffic driver there, so the traces from a fault carry that fault's commit.

One session costs about $0.001 to $0.003 on the stable version.

## Evaluation

`evals/` holds a golden set of 16 customer messages and an evaluation with
four scorers:

| Scorer | Type | Passes when |
|---|---|---|
| `expected_facts` | code | The answer contains the key facts of the case. |
| `required_tools` | code | The agent called the tools that the case needs. |
| `no_handoff_loop` | code | No sub-agent handled the message twice. |
| `follows_policy` | LLM judge (`gpt-6-sol`) | The answer states no shop policy that the help center does not contain. |

Run it against the stable version or a fault version:

```bash
python evals/run_eval.py                                     # stable version
python evals/run_eval.py --ref fault/f3-refund-limit-config  # one fault
python evals/run_eval.py --local                             # do not report to TraceRoot
```

Each fault fails the case that reaches its bug, so the run comparison in
TraceRoot shows which cases a commit broke.
