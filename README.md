# Acme Outfitters support agent

Traffic source for the public TraceRoot demo project. A support agent for a
fake outdoor gear shop answers synthetic customers. The traffic driver plays
the customers.

## What it shows

- A router span (`gpt-4.1-nano`) sends each message to one of three sub-agents.
- Sub-agents (`gpt-4.1-mini`) run a tool loop. Tools from one model step run in
  parallel, so the timeline shows overlapping spans.
- Refunds above $200 go to a supervisor review with `gpt-4.1`.
- Each session is one customer conversation with one to three turns. The
  customer ID is the user ID.
- Unknown order IDs make a tool fail, so some traces contain errors.

## Run

```bash
export TRACEROOT_API_KEY=tr-...            # key of the demo project
export TRACEROOT_HOST_URL=https://app.traceroot.ai   # or your own TraceRoot host
export TRACEROOT_ENVIRONMENT=production
export OPENAI_API_KEY=sk-...
uv run --no-project --python 3.13 --with-requirements requirements.txt \
  python traffic_driver.py --sessions 40 --concurrency 6
```

One session costs about $0.001 to $0.003.

This code is the source that TraceRoot shows in the demo project. Root cause
analysis and "Show code" links in demo traces point to commits in this repo.
