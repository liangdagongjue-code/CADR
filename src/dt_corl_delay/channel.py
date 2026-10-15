"""A delayed observation channel with post-decision acknowledgements."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import random
from typing import Generic, TypeVar

from .delay_process import DelayProcess


ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True)
class Packet(Generic[ObservationT]):
    packet_id: int
    observation: ObservationT
    generated_step: int
    delivery_step: int
    delay: int


@dataclass(frozen=True)
class Ack:
    """Delay information revealed only after the corresponding decision."""

    packet_id: int
    generated_step: int
    delivered_step: int
    revealed_step: int
    delay: int


@dataclass(frozen=True)
class PolicyInput(Generic[ObservationT]):
    """The complete information exposed to the deployed policy."""

    observation: ObservationT
    acknowledgements: tuple[Ack, ...]


@dataclass(frozen=True)
class ChannelStep(Generic[ObservationT]):
    policy_input: PolicyInput[ObservationT]
    delivered_packets: tuple[Packet[ObservationT], ...]
    selected_packet: Packet[ObservationT] | None


class DelayedObservationChannel(Generic[ObservationT]):
    """Queue observations without leaking the current delay to the policy.

    ACKs are scheduled at ``delivery_step + ack_lag``. With the default
    ``ack_lag=1``, a packet's delay cannot be used for the decision at which the
    packet first becomes visible.
    """

    def __init__(
        self,
        delay_process: DelayProcess,
        rng: random.Random,
        ack_lag: int = 1,
    ) -> None:
        if ack_lag < 1:
            raise ValueError("ack_lag must be at least one to prevent same-decision leakage")
        self.delay_process = delay_process
        self.rng = rng
        self.ack_lag = ack_lag
        self._delivery_heap: list[tuple[int, int, Packet[ObservationT]]] = []
        self._ack_heap: list[tuple[int, int, Ack]] = []
        self._next_packet_id = 0
        self._selected_packet: Packet[ObservationT] | None = None
        self._last_decision_step = -1

    def reset(self, initial_observation: ObservationT, initial_step: int = 0) -> None:
        self.delay_process.reset()
        self._delivery_heap.clear()
        self._ack_heap.clear()
        self._next_packet_id = 1
        self._last_decision_step = initial_step - 1
        self._selected_packet = Packet(
            packet_id=0,
            observation=initial_observation,
            generated_step=initial_step,
            delivery_step=initial_step,
            delay=0,
        )

    def send(self, observation: ObservationT, generated_step: int) -> Packet[ObservationT]:
        if self._selected_packet is None:
            raise RuntimeError("reset must be called before send")
        delay = self.delay_process.sample(generated_step, self.rng)
        packet = Packet(
            packet_id=self._next_packet_id,
            observation=observation,
            generated_step=generated_step,
            delivery_step=generated_step + delay,
            delay=delay,
        )
        self._next_packet_id += 1
        heapq.heappush(self._delivery_heap, (packet.delivery_step, packet.packet_id, packet))
        return packet

    def advance(self, decision_step: int) -> ChannelStep[ObservationT]:
        if self._selected_packet is None:
            raise RuntimeError("reset must be called before advance")
        if decision_step <= self._last_decision_step:
            raise ValueError("decision steps must be strictly increasing")
        self._last_decision_step = decision_step

        acknowledgements: list[Ack] = []
        while self._ack_heap and self._ack_heap[0][0] <= decision_step:
            _, _, acknowledgement = heapq.heappop(self._ack_heap)
            acknowledgements.append(acknowledgement)

        delivered: list[Packet[ObservationT]] = []
        while self._delivery_heap and self._delivery_heap[0][0] <= decision_step:
            _, _, packet = heapq.heappop(self._delivery_heap)
            delivered.append(packet)
            revealed_step = decision_step + self.ack_lag
            acknowledgement = Ack(
                packet_id=packet.packet_id,
                generated_step=packet.generated_step,
                delivered_step=decision_step,
                revealed_step=revealed_step,
                delay=packet.delay,
            )
            heapq.heappush(
                self._ack_heap,
                (acknowledgement.revealed_step, packet.packet_id, acknowledgement),
            )

        if delivered:
            newest = max(delivered, key=lambda packet: (packet.generated_step, packet.packet_id))
            if newest.generated_step >= self._selected_packet.generated_step:
                self._selected_packet = newest

        policy_input = PolicyInput(
            observation=self._selected_packet.observation,
            acknowledgements=tuple(acknowledgements),
        )
        return ChannelStep(
            policy_input=policy_input,
            delivered_packets=tuple(delivered),
            selected_packet=self._selected_packet,
        )

