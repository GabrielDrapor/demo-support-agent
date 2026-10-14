"""
Golden cases for the support agent.

Each case is one customer message. `expected` lists phrases: the answer passes
when it contains at least one phrase from each group. `metadata` names the
tools that the agent must call.

Version 1 of the dataset has the first 12 cases. Version 2 adds 4 cases that
came from problems in demo traffic.
"""

from __future__ import annotations

from traceroot import Dataset

DATASET_NAME = "Support agent golden set"
DATASET_KEY = "support-golden"

V1 = [
    # (id, customer, message, expected phrase groups, required tools)
    (
        "track-shipped",
        "C-1001",
        "Where is my order A-48230?",
        [["fedex", "salt lake"]],
        ["get_tracking"],
    ),
    (
        "all-open-orders",
        "C-1001",
        "Can you check the status of all my open orders?",
        [["a-48213"], ["a-48230"]],
        [],
    ),
    (
        "small-refund",
        "C-1005",
        "The headlamp from order A-48315 arrived broken. Please refund the full amount.",
        [["49"]],
        ["issue_refund"],
    ),
    (
        "large-refund-review",
        "C-1006",
        "The sleeping bag from A-48333 arrived torn. Refund the full $419 please.",
        [["supervisor", "review"]],
        ["issue_refund"],
    ),
    (
        "gold-late-return",
        "C-1004",
        "Can I still return the down jacket from order A-48290? It doesn't fit.",
        [["60"]],
        ["search_help_center"],
    ),
    (
        "warranty",
        "C-1003",
        "Does the warranty cover a broken tent pole?",
        [["2-year", "2 year", "two-year", "two year"]],
        ["search_help_center"],
    ),
    (
        "shipping-time",
        "C-1003",
        "How long does standard shipping take to Seattle?",
        [["3-5", "3–5", "3 to 5", "three to five"]],
        ["search_help_center"],
    ),
    ("gold-benefits", "C-1001", "What do gold members get?", [["express"]], ["search_help_center"]),
    (
        "worn-shoes",
        "C-1003",
        "Can I return shoes I already wore on a trail?",
        [["defective", "defect"]],
        ["search_help_center"],
    ),
    (
        "wrong-order-id",
        "C-1001",
        "Where is order A-84230?",
        [["not find", "not found", "couldn't find", "could not find", "check"]],
        [],
    ),
    (
        "orders-since",
        "C-1004",
        "What have I ordered in the last two weeks?",
        [["a-48302"]],
        ["list_orders"],
    ),
    (
        "track-and-return",
        "C-1004",
        "Where is order A-48302? Also I want to return the down jacket from A-48290.",
        [["a-48302"], ["return"]],
        [],
    ),
]

V2_ADDITIONS = [
    (
        "base-layer-sizing",
        "C-1002",
        "Do your base layers run small?",
        [["athletic", "size up"]],
        ["search_help_center"],
    ),
    (
        "manager-complaint",
        "C-1002",
        "My delivery was late again. I want to talk to a manager about order A-48360.",
        [["ticket"]],
        ["create_ticket"],
    ),
    ("refund-over-total", "C-1005", "Refund $500 for order A-48315.", [["49"]], []),
    (
        "express-next-time",
        "C-1001",
        "Is express shipping free for me?",
        [["express"]],
        ["search_help_center"],
    ),
]


def build(version: int = 2) -> Dataset:
    """Return the dataset as it was at `version` (1 or 2)."""
    ds = Dataset(DATASET_NAME, key=DATASET_KEY)
    cases = V1 + (V2_ADDITIONS if version >= 2 else [])
    for case_id, customer_id, message, phrases, tools in cases:
        ds.add(
            {"customer_id": customer_id, "message": message},
            expected={"any_of_each": phrases},
            metadata={"required_tools": tools},
            id=case_id,
        )
    return ds
