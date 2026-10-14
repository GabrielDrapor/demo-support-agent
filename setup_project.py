"""
One-time setup of a TraceRoot project for the demo: detectors, alerts and a dashboard.

The script is safe to run again. Detectors and the dashboard are matched by
name. Alerts and widgets are created only when one with the same name does
not exist yet.

Writes need a user login, not a project key:

    TRACEROOT_API_KEY=tr-...            # project key; selects the project
    TRACEROOT_ACCESS_TOKEN=...          # session token or CLI access token of a member
    TRACEROOT_HOST_URL=https://app.traceroot.ai
    python setup_project.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

HOST = os.environ.get("TRACEROOT_HOST_URL", "https://app.traceroot.ai").rstrip("/")
API = f"{HOST}/api/v1/public"

DETECTORS = [
    {"name": "Tool failures", "template": "failure", "sample_rate": 100, "enable_rca": True},
    {
        "name": "Ungrounded policy answers",
        "template": "hallucination",
        "sample_rate": 100,
        "enable_rca": True,
        "prompt": (
            "This trace is a customer-support agent for an outdoor gear shop. Check the agent's "
            "final answer to the customer for shop policy facts: return windows, refund rules, "
            "shipping times, warranty terms, loyalty benefits, sizing advice.\n\n"
            "A policy fact is grounded if it appears in a search_help_center tool result in this "
            "trace, or in the agent's own system prompt (for example the $200 supervisor-review "
            "rule). Facts about this customer's orders that come from other tool results are "
            "also grounded.\n\n"
            "Report identified=true only if the final answer states a specific policy fact (a "
            "number, a time period, a benefit or a rule) that is grounded nowhere in the trace. "
            "This includes answers given after search_help_center returned no articles. Do not "
            "flag answers that say the information could not be found, polite filler such as "
            '"you should see the funds soon", or offers to help further.'
        ),
        "output_schema": [
            {"name": "hallucinated_claim", "type": "string"},
            {"name": "search_returned_results", "type": "string"},
        ],
    },
    {
        "name": "Large refund without supervisor review",
        "template": "blank",
        "sample_rate": 100,
        "enable_rca": True,
        "prompt": (
            "This trace is a customer-support agent that can issue refunds. Shop policy: a "
            "refund above $200 must go through supervisor review. In the trace, supervisor "
            "review is a span named supervisor_review under the issue_refund tool span.\n\n"
            "Report identified=true if an issue_refund tool call has amount_usd above 200 and "
            'its result has status "refunded" with no supervisor_review span under it. Report '
            "identified=false if no refund above 200 was issued."
        ),
        "output_schema": [
            {"name": "order_id", "type": "string"},
            {"name": "amount_usd", "type": "string"},
        ],
    },
    {
        "name": "Agent handoff loop",
        "template": "logic",
        "sample_rate": 100,
        "enable_rca": True,
        "prompt": (
            "This trace is a customer-support agent with sub-agents (orders_agent, "
            "refunds_agent, product_agent) that can hand a conversation to each other with the "
            "transfer_to_agent tool.\n\n"
            "Report identified=true if the conversation is handed back to an agent that "
            "already handled it in this trace (for example refunds_agent, then orders_agent, "
            'then refunds_agent again), or if the final answer is the fallback "I could not '
            'finish this request". One handoff from one agent to another is normal.'
        ),
        "output_schema": [{"name": "agent_sequence", "type": "string"}],
    },
]

ALERTS = [
    {
        "name": "Slow customer lookups",
        "view": "SPANS",
        "measure": "latency",
        "aggregation": "p95",
        "filters": [{"field": "name", "op": "=", "value": "get_customer"}],
        "window": "30m",
        "threshold_operator": ">",
        "threshold": 1000,
        "renotify": {"mode": "OFF"},
        "no_data_mode": "HOLD",
    },
    {
        "name": "gpt-6-sol spend per hour",
        "view": "SPANS",
        "measure": "cost",
        "aggregation": "sum",
        "filters": [
            {"field": "model_name", "op": "=", "value": "gpt-6-sol"},
            {"field": "span_kind", "op": "=", "value": "LLM"},
        ],
        "window": "1h",
        "threshold_operator": ">",
        "threshold": 0.005,
        "renotify": {"mode": "OFF"},
        "no_data_mode": "ZERO",
    },
]

DASHBOARD = {
    "name": "Support agent health",
    "description": "Volume, cost, latency and errors of the Acme Outfitters support agent.",
}


def _chart(view, measure, agg, display, breakdown=None, filters=()):
    return {
        "view": view,
        "filters": [{"field": f, "op": op, "value": v} for f, op, v in filters],
        "metric": {"measure": measure, "agg": agg},
        "breakdown": breakdown,
        "display": {"type": display},
    }


WIDGETS = [
    ("Traces", _chart("traces", "count", "count", "line")),
    (
        "Cost by model",
        _chart("spans", "cost", "sum", "area", "model_name", [("span_kind", "=", "LLM")]),
    ),
    ("Trace latency p95", _chart("traces", "duration_ms", "p95", "line")),
    (
        "Tool errors",
        _chart(
            "spans",
            "count",
            "count",
            "bar",
            "name",
            [("span_kind", "=", "TOOL"), ("status", "=", "ERROR")],
        ),
    ),
    (
        "Sub-agent calls",
        _chart(
            "spans",
            "count",
            "count",
            "pie",
            "name",
            [
                ("span_kind", "=", "AGENT"),
                ("is_root", "=", "false"),
                ("name", "contains", "_agent"),
            ],
        ),
    ),
    (
        "Customer lookup latency p95",
        _chart("spans", "duration_ms", "p95", "line", None, [("name", "=", "get_customer")]),
    ),
    ("Cost by customer", _chart("traces", "cost", "sum", "table", "user_id")),
]


def call(method: str, path: str, token: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"{API}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        sys.exit(f"{method} {path} failed: {exc.code} {exc.read().decode()[:300]}")


def main() -> None:
    project_key = os.environ["TRACEROOT_API_KEY"]
    token = os.environ["TRACEROOT_ACCESS_TOKEN"]
    project = call("GET", "/whoami", project_key)
    pid = project["project_id"]
    print(f"Project: {project['workspace_name']} / {project['project_name']} ({pid})")

    for detector in DETECTORS:
        result = call("POST", "/detectors", token, {"project_id": pid, **detector})
        print(f"detector  {'created' if result['created'] else 'exists '}  {detector['name']}")

    existing = {a["name"] for a in call("GET", "/alerts?limit=100", project_key)["data"]}
    for alert in ALERTS:
        if alert["name"] in existing:
            print(f"alert     exists   {alert['name']}")
            continue
        call("POST", "/alerts", token, {"project_id": pid, **alert})
        print(f"alert     created  {alert['name']}")

    dashboard = call("POST", "/dashboards", token, {"project_id": pid, **DASHBOARD})
    print(f"dashboard {'created' if dashboard['created'] else 'exists '}  {DASHBOARD['name']}")
    current = call("GET", f"/dashboards/{dashboard['id']}?project_id={pid}", project_key)
    titles = {w["title"] for w in current.get("widgets", [])}
    for title, spec in WIDGETS:
        if title in titles:
            continue
        body = {
            "project_id": pid,
            "dashboard_id": dashboard["id"],
            "title": title,
            "type": "query",
            "spec": spec,
        }
        call("POST", "/widgets", token, body)
        print(f"widget    created  {title}")


if __name__ == "__main__":
    main()
