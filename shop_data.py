"""
In-memory data for the Acme Outfitters support agent.

The demo project needs realistic but fake records, so this module holds a
small catalog of customers, orders and help-center articles.
"""

from __future__ import annotations

import random
import time
from datetime import date, timedelta

REFUND_LIMIT_USD = 200.0
RETURN_WINDOW_DAYS = 30

CUSTOMERS = {
    "C-1001": {"name": "Maya Chen", "tier": "gold", "email": "maya@example.com"},
    "C-1002": {"name": "Jonas Weber", "tier": "standard", "email": "jonas@example.com"},
    "C-1003": {"name": "Priya Nair", "tier": "standard", "email": "priya@example.com"},
    "C-1004": {"name": "Diego Alvarez", "tier": "gold", "email": "diego@example.com"},
    "C-1005": {"name": "Aiko Tanaka", "tier": "standard", "email": "aiko@example.com"},
    "C-1006": {"name": "Sam Okafor", "tier": "platinum", "email": "sam@example.com"},
}

_TODAY = date.today()


def _order(order_id, customer_id, item, total, status, days_ago, carrier="UPS"):
    return {
        "order_id": order_id,
        "customer_id": customer_id,
        "item": item,
        "total_usd": total,
        "status": status,
        "order_date": (_TODAY - timedelta(days=days_ago)).isoformat(),
        "carrier": carrier,
    }


ORDERS = {
    o["order_id"]: o
    for o in [
        _order("A-48213", "C-1001", "Trailblazer 40L backpack", 149.00, "delivered", 12),
        _order("A-48230", "C-1001", "Merino base layer (M)", 89.00, "shipped", 3, "FedEx"),
        _order("A-48251", "C-1002", "Summit 2-person tent", 329.00, "delivered", 20),
        _order("A-48277", "C-1003", "Trail running shoes (W 8)", 135.00, "processing", 1),
        _order("A-48290", "C-1004", "Down jacket (L)", 279.00, "delivered", 41),
        _order("A-48302", "C-1004", "Insulated bottle 1L", 34.00, "shipped", 4, "USPS"),
        _order("A-48315", "C-1005", "Headlamp 400lm", 49.00, "delivered", 7),
        _order("A-48333", "C-1006", "Ultralight sleeping bag", 419.00, "delivered", 9),
        _order("A-48347", "C-1006", "Camp stove kit", 119.00, "shipped", 2, "UPS"),
        _order("A-48360", "C-1002", "Rain shell (M)", 189.00, "delivered", 28),
    ]
}

ARTICLES = [
    {
        "id": "kb-returns",
        "title": "Return policy",
        "body": f"Items can be returned within {RETURN_WINDOW_DAYS} days of delivery in original condition. "
        "Worn footwear can be returned only if defective.",
    },
    {
        "id": "kb-refunds",
        "title": "Refunds",
        "body": f"Support agents can issue refunds up to ${REFUND_LIMIT_USD:.0f} without approval. "
        "Larger refunds need a supervisor review, which takes 1-2 business days. "
        "Refunds go back to the original payment method within 5-7 business days.",
    },
    {
        "id": "kb-shipping",
        "title": "Shipping times",
        "body": "Standard shipping takes 3-5 business days. Express takes 1-2 business days. "
        "Orders ship from Denver, CO.",
    },
    {
        "id": "kb-warranty",
        "title": "Warranty",
        "body": "Tents, backpacks and sleeping bags have a 2-year warranty against manufacturing defects. "
        "Normal wear and tear is not covered.",
    },
    {
        "id": "kb-loyalty",
        "title": "Loyalty tiers",
        "body": "Gold and platinum members get free express shipping and extended 60-day returns.",
    },
    {
        "id": "kb-sizing",
        "title": "Sizing help",
        "body": "Our jackets run true to size. Base layers have an athletic fit; size up for a relaxed fit.",
    },
]

TRACKING_EVENTS = [
    "Label created",
    "Picked up in Denver, CO",
    "In transit - Salt Lake City, UT",
    "In transit - Reno, NV",
    "Out for delivery",
]


def simulated_latency(low: float, high: float) -> None:
    """Sleep for a random time, like a call to a slow backend service."""
    time.sleep(random.uniform(low, high))
