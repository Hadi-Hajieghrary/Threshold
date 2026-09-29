"""Packet fabric: one directed delay ring per edge of the communication graph (spec).

Packets are sent every 100 ms on every outgoing edge and delivered ``round(tau / 10 ms)``
ticks later.  Each edge owns a drop gate seeded by ``SeedSequence([master, 13, src, dst])``
and drawn once per packet (drop probability 0, declared).  Drake-free.
"""

from __future__ import annotations

from typing import Any

import numpy as np

COMMS_STREAM = 13
TICK = 0.01
PACKET_PERIOD_TICKS = 10


def comms_generator(master_seed: int, source: int, destination: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([master_seed, COMMS_STREAM, source, destination]))


def delay_ticks(tau: float) -> int:
    if tau < 0.0:
        raise ValueError("latency must be nonnegative")
    return int(round(tau / TICK))


def graph_edges(graph: str, vessel_count: int) -> tuple[tuple[int, int], ...]:
    """Directed edges: ``cycle`` is i -> i+1 and i -> i-1 (mod N); ``complete`` is every pair."""
    if graph == "cycle":
        edges: list[tuple[int, int]] = []
        for source in range(vessel_count):
            for destination in ((source + 1) % vessel_count, (source - 1) % vessel_count):
                if destination != source and (source, destination) not in edges:
                    edges.append((source, destination))
        return tuple(edges)
    if graph == "complete":
        return tuple((s, d) for s in range(vessel_count) for d in range(vessel_count) if s != d)
    raise ValueError(f"unknown communication graph: {graph}")


class DelayRing:
    """``delay + 1`` slots: a packet pushed at tick k is popped at tick k + delay."""

    def __init__(self, delay: int) -> None:
        self.delay = int(delay)
        self._payloads: list[Any] = [None] * (self.delay + 1)
        self._stamps = [-1] * (self.delay + 1)

    def push(self, tick: int, payload: Any) -> None:
        slot = tick % len(self._stamps)
        self._payloads[slot] = payload
        self._stamps[slot] = tick

    def pop(self, tick: int) -> Any:
        sent = tick - self.delay
        if sent < 0:
            return None
        slot = sent % len(self._stamps)
        if self._stamps[slot] != sent:
            return None
        payload = self._payloads[slot]
        self._payloads[slot] = None
        self._stamps[slot] = -1
        return payload


class PacketFabric:
    """Directed delay rings and drop gates over a cycle (default) or complete graph."""

    def __init__(
        self,
        vessel_count: int,
        delay: int,
        master_seed: int,
        graph: str = "cycle",
        drop_probability: float = 0.0,
    ) -> None:
        if not 0.0 <= drop_probability <= 1.0:
            raise ValueError("drop probability must lie in [0, 1]")
        self.edges = graph_edges(graph, vessel_count)
        self.delay = int(delay)
        self.drop_probability = float(drop_probability)
        self._rings = {edge: DelayRing(delay) for edge in self.edges}
        self._gates = {edge: comms_generator(master_seed, *edge) for edge in self.edges}
        self._outgoing = {
            source: [edge for edge in self.edges if edge[0] == source] for source in range(vessel_count)
        }
        self.sent = 0
        self.dropped = 0

    @staticmethod
    def is_send_tick(tick: int) -> bool:
        return tick % PACKET_PERIOD_TICKS == 0

    def send(self, tick: int, source: int, payload: Any) -> None:
        for edge in self._outgoing[source]:
            if self._gates[edge].random() < self.drop_probability:
                self.dropped += 1
                continue
            self._rings[edge].push(tick, payload)
            self.sent += 1

    def deliver(self, tick: int) -> list[tuple[int, int, Any]]:
        """Packets due at ``tick``, in edge order."""
        delivered = []
        for edge in self.edges:
            payload = self._rings[edge].pop(tick)
            if payload is not None:
                delivered.append((edge[0], edge[1], payload))
        return delivered
