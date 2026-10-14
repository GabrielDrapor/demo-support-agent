"""
Traffic driver for the TraceRoot demo project.

Plays synthetic customers who talk to the support agent. Each session is
one customer conversation with one or more turns, so the demo project gets
traces, sessions and users.

Usage:
    TRACEROOT_API_KEY=tr-... TRACEROOT_HOST_URL=https://app.traceroot.ai \\
    OPENAI_API_KEY=sk-... python traffic_driver.py --sessions 40
"""

from __future__ import annotations

import argparse
import logging
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import traceroot
from traceroot import Integration

traceroot.initialize(integrations=[Integration.OPENAI])

import support_agent
from shop_data import ORDERS

logger = logging.getLogger("traffic_driver")


def _orders_of(customer_id: str, status: str | None = None) -> list[dict]:
    return [
        o
        for o in ORDERS.values()
        if o["customer_id"] == customer_id and (status is None or o["status"] == status)
    ]


# Each scenario returns (customer_id, list of customer turns), or None if it does not fit the data.


def where_is_my_order():
    order = random.choice([o for o in ORDERS.values() if o["status"] in ("shipped", "processing")])
    return order["customer_id"], [
        random.choice(
            [
                f"Hi, where is my order {order['order_id']}?",
                f"When will {order['order_id']} arrive?",
                f"My order {order['order_id']} hasn't shown up yet, can you check?",
            ]
        ),
        random.choice(["Thanks! Is express shipping available next time?", "Great, thank you."]),
    ]


def track_all_orders():
    customer_id = random.choice(["C-1001", "C-1004", "C-1006"])
    return customer_id, ["Can you check the status of all my open orders?"]


def small_refund():
    order = random.choice(
        [o for o in ORDERS.values() if o["status"] == "delivered" and o["total_usd"] <= 200]
    )
    return order["customer_id"], [
        f"The {order['item']} from order {order['order_id']} arrived damaged. I'd like a refund.",
        "Yes, a full refund please.",
    ]


def large_refund():
    order = random.choice(
        [o for o in ORDERS.values() if o["total_usd"] > 200 and o["status"] == "delivered"]
    )
    return order["customer_id"], [
        f"I want to return the {order['item']} (order {order['order_id']}). It is not what I expected.",
        f"Please refund the full ${order['total_usd']:.0f}.",
        "How long will the review take?",
    ]


def late_return():
    return "C-1004", ["I want to return the down jacket from order A-48290, it doesn't fit."]


def product_question():
    customer_id = random.choice(list({o["customer_id"] for o in ORDERS.values()}))
    return customer_id, [
        random.choice(
            [
                "Does the warranty cover a broken tent pole?",
                "Do your base layers run small?",
                "How long does standard shipping take to Seattle?",
                "What do gold members get?",
                "Can I return shoes I already wore on a trail?",
            ]
        )
    ]


def wrong_order_id():
    order = random.choice(list(ORDERS.values()))
    typo = order["order_id"].replace("A-48", "A-84")
    return order["customer_id"], [
        f"Where is order {typo}?",
        f"Sorry, I meant {order['order_id']}.",
    ]


def complaint():
    customer_id = "C-1002"
    return customer_id, [
        "This is the second time my delivery was late. I want to talk to a manager.",
        "Order A-48360.",
    ]


def orders_since():
    customer_id = random.choice(["C-1001", "C-1004", "C-1006"])
    return customer_id, [
        random.choice(
            [
                "What have I ordered in the last two weeks?",
                "Can you list my orders from the past month?",
                "Show me everything I bought since the start of last week.",
            ]
        )
    ]


def track_and_refund():
    """Ask about two orders at once: one for the orders agent, one for refunds."""
    customer_id = random.choice(["C-1001", "C-1004", "C-1006"])
    shipped = _orders_of(customer_id, "shipped")[0]["order_id"]
    delivered = _orders_of(customer_id, "delivered")[0]
    return customer_id, [
        random.choice(
            [
                f"Where is order {shipped}? Also I want to return the {delivered['item']} "
                f"from {delivered['order_id']}.",
                f"Two things: refund {delivered['order_id']} please, and when will {shipped} arrive?",
                f"Where is {shipped}? If it won't arrive this week, I want my money back.",
            ]
        )
    ]


SCENARIOS = [
    (where_is_my_order, 5),
    (track_all_orders, 2),
    (small_refund, 3),
    (large_refund, 2),
    (late_return, 1),
    (product_question, 4),
    (wrong_order_id, 2),
    (complaint, 1),
    (orders_since, 2),
    (track_and_refund, 2),
]


def pick_scenario(focus: str | None, focus_share: float):
    """Pick a scenario by weight. With `focus`, that scenario gets `focus_share` of sessions."""
    if focus and random.random() < focus_share:
        return next(s for s, _ in SCENARIOS if s.__name__ == focus)
    return random.choices([s for s, _ in SCENARIOS], weights=[w for _, w in SCENARIOS])[0]


def run_session(index: int, focus: str | None = None, focus_share: float = 0.0) -> None:
    scenario = pick_scenario(focus, focus_share)
    customer_id, turns = scenario()
    session_id = f"sess-{uuid.uuid4().hex[:12]}"
    channel = random.choices(["web", "mobile", "email"], weights=[6, 3, 1])[0]
    history: list[dict] = []
    for message in turns:
        try:
            reply = support_agent.handle_message(customer_id, session_id, history, message, channel)
        except Exception:
            logger.exception("session %s failed", session_id)
            return
        history += [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ]
    logger.info(
        "session %d done: %s (%s, %d turns)",
        index,
        scenario.__name__,
        customer_id,
        len(turns),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--focus", choices=[s.__name__ for s, _ in SCENARIOS], default=None)
    parser.add_argument("--focus-share", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    random.seed(args.seed)
    support_agent.warm_customer_cache()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        run = partial(run_session, focus=args.focus, focus_share=args.focus_share)
        list(pool.map(run, range(args.sessions)))
    traceroot.flush()


if __name__ == "__main__":
    main()
