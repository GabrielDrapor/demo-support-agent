"""
Acme Outfitters support agent, instrumented with TraceRoot.

A router model reads each customer message and sends it to one of three
sub-agents. Each sub-agent runs an OpenAI tool loop. When the model asks
for more than one tool in a step, the tools run in parallel, so the trace
timeline shows overlapping spans. Refunds above the limit go to a stronger
model for a supervisor review.
"""

from __future__ import annotations

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor

import openai
import shop_data
from shop_data import ARTICLES, CUSTOMERS, ORDERS, REFUND_LIMIT_USD, TRACKING_EVENTS

from traceroot import observe, update_current_span, update_current_trace

ROUTER_MODEL = "gpt-4.1-nano"
AGENT_MODEL = "gpt-4.1-mini"
REVIEW_MODEL = "gpt-4.1"
MAX_TOOL_STEPS = 6

_client: openai.OpenAI | None = None


def client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI()
    return _client


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """A tool failed in a way that the model should see and handle."""


@observe(name="search_help_center", type="tool")
def search_help_center(query: str) -> list[dict]:
    """Return the two help-center articles that best match the query."""
    shop_data.simulated_latency(0.1, 0.3)
    words = {w.strip("?.,!").lower() for w in query.split() if len(w) > 2}

    def score(article: dict) -> int:
        text = f"{article['title']} {article['body']}".lower()
        return sum(1 for w in words if w in text)

    ranked = sorted(ARTICLES, key=score, reverse=True)
    return [a for a in ranked[:2] if score(a) > 0]


@observe(name="lookup_order", type="tool")
def lookup_order(order_id: str) -> dict:
    shop_data.simulated_latency(0.2, 0.6)
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        raise ToolError(f"Order {order_id} not found")
    return order


@observe(name="get_tracking", type="tool")
def get_tracking(order_id: str) -> dict:
    shop_data.simulated_latency(0.4, 1.2)
    order = lookup_order(order_id)
    if order["status"] == "processing":
        return {"order_id": order["order_id"], "events": [], "note": "Not shipped yet"}
    steps = len(TRACKING_EVENTS) if order["status"] == "delivered" else 3
    events = TRACKING_EVENTS[:steps] + (["Delivered"] if order["status"] == "delivered" else [])
    return {"order_id": order["order_id"], "carrier": order["carrier"], "events": events}


@observe(name="get_customer", type="tool")
def get_customer(customer_id: str) -> dict:
    shop_data.simulated_latency(0.1, 0.4)
    customer = CUSTOMERS.get(customer_id)
    if customer is None:
        raise ToolError(f"Customer {customer_id} not found")
    orders = [o["order_id"] for o in ORDERS.values() if o["customer_id"] == customer_id]
    return {"customer_id": customer_id, **customer, "orders": orders}


@observe(name="issue_refund", type="tool")
def issue_refund(order_id: str, amount_usd: float, reason: str) -> dict:
    shop_data.simulated_latency(0.3, 0.8)
    order = lookup_order(order_id)
    if amount_usd > order["total_usd"]:
        raise ToolError(f"Refund {amount_usd} is more than the order total {order['total_usd']}")
    if amount_usd > REFUND_LIMIT_USD:
        return supervisor_review(order, amount_usd, reason)
    return {"order_id": order["order_id"], "status": "refunded", "amount_usd": amount_usd}


@observe(name="create_ticket", type="tool")
def create_ticket(summary: str, priority: str = "normal") -> dict:
    shop_data.simulated_latency(0.1, 0.3)
    return {"ticket_id": f"T-{abs(hash(summary)) % 90000 + 10000}", "priority": priority}


TOOL_FUNCS = {
    "search_help_center": search_help_center,
    "lookup_order": lookup_order,
    "get_tracking": get_tracking,
    "get_customer": get_customer,
    "issue_refund": issue_refund,
    "create_ticket": create_ticket,
}


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_ORDER_ID = {"order_id": {"type": "string", "description": "Order ID, for example A-48213"}}

TOOL_SCHEMAS = {
    "search_help_center": _schema(
        "search_help_center",
        "Search help-center articles about policies, shipping, warranty and sizing.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    "lookup_order": _schema("lookup_order", "Get one order by its ID.", _ORDER_ID, ["order_id"]),
    "get_tracking": _schema(
        "get_tracking", "Get carrier tracking events for an order.", _ORDER_ID, ["order_id"]
    ),
    "get_customer": _schema(
        "get_customer",
        "Get the customer profile and the IDs of their orders.",
        {"customer_id": {"type": "string"}},
        ["customer_id"],
    ),
    "issue_refund": _schema(
        "issue_refund",
        "Refund part or all of an order to the original payment method.",
        {**_ORDER_ID, "amount_usd": {"type": "number"}, "reason": {"type": "string"}},
        ["order_id", "amount_usd", "reason"],
    ),
    "create_ticket": _schema(
        "create_ticket",
        "Create a ticket for the human support team.",
        {
            "summary": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "normal", "high"]},
        },
        ["summary"],
    ),
}


# ---------------------------------------------------------------------------
# Supervisor review
# ---------------------------------------------------------------------------


@observe(name="supervisor_review", type="agent")
def supervisor_review(order: dict, amount_usd: float, reason: str) -> dict:
    """Ask a stronger model whether a refund above the limit is justified."""
    policy = next(a["body"] for a in ARTICLES if a["id"] == "kb-returns")
    response = client().chat.completions.create(
        model=REVIEW_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You review refund requests above the agent limit. "
                f"Policy: {policy} Answer as JSON: "
                '{"decision": "approve" | "deny", "rationale": "..."}',
            },
            {
                "role": "user",
                "content": json.dumps({"order": order, "amount_usd": amount_usd, "reason": reason}),
            },
        ],
    )
    verdict = json.loads(response.choices[0].message.content)
    status = "pending_supervisor" if verdict.get("decision") == "approve" else "denied"
    return {"order_id": order["order_id"], "status": status, **verdict}


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def _run_tool(call) -> dict:
    func = TOOL_FUNCS[call.function.name]
    try:
        result = func(**json.loads(call.function.arguments))
    except ToolError as exc:
        result = {"error": str(exc)}
    return {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}


def _run_tools_in_parallel(calls) -> list[dict]:
    """Run the tool calls of one model step at the same time.

    Each thread gets a copy of the current context, so the tool spans stay
    children of the agent span.
    """
    if len(calls) == 1:
        return [_run_tool(calls[0])]
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = [pool.submit(contextvars.copy_context().run, _run_tool, c) for c in calls]
        return [f.result() for f in futures]


def _tool_loop(system: str, history: list[dict], tool_names: list[str]) -> str:
    messages = [{"role": "system", "content": system}, *history]
    tools = [TOOL_SCHEMAS[n] for n in tool_names]
    for _ in range(MAX_TOOL_STEPS):
        response = client().chat.completions.create(
            model=AGENT_MODEL, messages=messages, tools=tools, parallel_tool_calls=True
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content or ""
        messages.append(message.model_dump(exclude_none=True))
        messages.extend(_run_tools_in_parallel(message.tool_calls))
    return "I'm sorry, I could not finish this request. I have passed it to our support team."


_BASE_RULES = (
    "You are a support agent for Acme Outfitters, an outdoor gear shop. "
    "Be brief and friendly. Use tools to get facts; never guess order details. "
    "If a tool returns an error, tell the customer what you need from them."
)


@observe(name="orders_agent", type="agent")
def orders_agent(customer_id: str, history: list[dict]) -> str:
    system = (
        f"{_BASE_RULES} You handle order status and delivery questions. "
        f"The customer ID is {customer_id}. If the customer has several orders, check them all at once."
    )
    return _tool_loop(
        system, history, ["get_customer", "lookup_order", "get_tracking", "create_ticket"]
    )


@observe(name="refunds_agent", type="agent")
def refunds_agent(customer_id: str, history: list[dict]) -> str:
    system = (
        f"{_BASE_RULES} You handle returns and refunds. The customer ID is {customer_id}. "
        "Check the return policy and the order before you refund. "
        f"Refunds above ${REFUND_LIMIT_USD:.0f} go to supervisor review; tell the customer the result."
    )
    return _tool_loop(
        system,
        history,
        ["get_customer", "lookup_order", "search_help_center", "issue_refund", "create_ticket"],
    )


@observe(name="product_agent", type="agent")
def product_agent(customer_id: str, history: list[dict]) -> str:
    system = (
        f"{_BASE_RULES} You answer questions about products, sizing, warranty and shipping. "
        "Answer only from help-center articles."
    )
    return _tool_loop(system, history, ["search_help_center", "create_ticket"])


ROUTES = {"orders": orders_agent, "refunds": refunds_agent, "product": product_agent}


@observe(name="route_message", type="span")
def route_message(message: str) -> str:
    response = client().chat.completions.create(
        model=ROUTER_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Classify the customer message. Answer as JSON: "
                '{"route": "orders" | "refunds" | "product"}. '
                "orders = order status, tracking, delivery. refunds = returns, refunds, damaged items. "
                "product = everything else.",
            },
            {"role": "user", "content": message},
        ],
    )
    route = json.loads(response.choices[0].message.content).get("route", "product")
    return route if route in ROUTES else "product"


@observe(name="support_agent", type="agent")
def handle_message(
    customer_id: str,
    session_id: str,
    history: list[dict],
    message: str,
    channel: str,
) -> str:
    """Answer one customer message. `history` holds the earlier turns of the session."""
    tier = CUSTOMERS[customer_id]["tier"]
    update_current_trace(
        user_id=customer_id,
        session_id=session_id,
        tags=[f"channel:{channel}", f"tier:{tier}"],
        metadata={"channel": channel, "customer_tier": tier, "turn": len(history) // 2 + 1},
    )
    route = route_message(message)
    update_current_span(metadata={"route": route})
    return ROUTES[route](customer_id, [*history, {"role": "user", "content": message}])
