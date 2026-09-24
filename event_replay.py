"""Physical observed-request replay; NOT policy-responsive PDST closed loop.

The CLI validates the full 567-station day without simulated truck dispatch.
It does not add a fifth optimization arm or replay inferred operator moves.
The reusable engine carries customer trips, failed-return queues, truck loads,
and unfinished truck journeys forward. Historical successful trips are requests,
not uncensored latent demand. See EVENT_REPLAY_SCOPE.md before interpreting output.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass, field
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_V7 = Path("D:/physic_predict_bike/divvy_2017_physics_closed_active_v7_10min_from_v64")
NS_PER_SECOND = 1_000_000_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def integer_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    if not np.allclose(array, np.rint(array), rtol=0, atol=1e-7):
        raise ValueError(f"{name} must contain integer bike counts; no silent rounding")
    return np.rint(array).astype(np.int64)


@dataclass(frozen=True)
class TripRequest:
    trip_id: str
    start_ns: int
    stop_ns: int
    origin: int
    destination: int


@dataclass
class TruckState:
    capacity: int
    position: int
    load: int = 0
    pending: deque[int] = field(default_factory=deque)
    targets: np.ndarray | None = None
    moving_to: int | None = None
    departed_ns: int | None = None
    arrival_ns: int | None = None
    busy: bool = False
    distance_km: float = 0.0
    service_direction: int = 0
    service_remaining: int = 0


class ObservedRequestReplay:
    """Replay state with no historical-inventory resets and no future-trip policy input.

    Node indices 0..n-1 are physical stations; index n is the depot, if truck
    distances are supplied. The customer external state is stored separately.
    Dispatch is allowed only for an idle truck. A dispatched route persists
    across ``advance`` calls; it is not instantly executed at decision time.
    """

    def __init__(self, inventory: Any, capacities: Any, start_ns: int,
                 crossing_trips: list[TripRequest], *, truck_count: int = 4,
                 truck_capacity: int = 30, distance_km: np.ndarray | None = None,
                 speed_kmh: float = 18.0, stop_seconds: float = 15.0,
                 handling_seconds: float = 6.0):
        self.inventory = integer_vector(inventory, "initial inventory")
        self.capacities = integer_vector(capacities, "station capacities")
        if self.inventory.shape != self.capacities.shape:
            raise ValueError("Inventory and capacities must have equal shape")
        if np.any(self.inventory < 0) or np.any(self.inventory > self.capacities):
            raise ValueError("Initial physical state violates station capacity")
        if truck_count < 0 or truck_capacity <= 0:
            raise ValueError("Invalid truck resource values")
        if min(speed_kmh, handling_seconds) <= 0 or stop_seconds < 0:
            raise ValueError("Travel/handling must take positive time")
        self.n = len(self.inventory)
        self.start_ns = self.now_ns = int(start_ns)
        self.distance_matrix = None if distance_km is None else np.asarray(distance_km, float)
        if self.distance_matrix is not None:
            if self.distance_matrix.shape != (self.n + 1, self.n + 1):
                raise ValueError("Distance matrix order must be stations then depot")
            if (not np.all(np.isfinite(self.distance_matrix))
                    or np.any(self.distance_matrix < 0)):
                raise ValueError("Distances must be finite and nonnegative")
        self.speed_kmh = float(speed_kmh)
        self.stop_ns = int(round(stop_seconds * NS_PER_SECOND))
        self.handling_ns = int(round(handling_seconds * NS_PER_SECOND))
        self.trucks = [TruckState(int(truck_capacity), self.n) for _ in range(truck_count)]
        self.waiting: list[deque[str]] = [deque() for _ in range(self.n)]
        self.riding: set[str] = set()
        self.waiting_count = 0
        self.seen_trip_ids: set[str] = set()
        self.events: list[tuple[int, int, int, str, Any]] = []
        self.serial = 0
        self.counts: Counter[str] = Counter()
        self.fleet = int(self.inventory.sum()) + len(crossing_trips)
        self.invariant_checks = 0
        self.max_abs_fleet_error = 0
        for trip in crossing_trips:
            self._validate_trip(trip)
            if not (trip.start_ns <= self.start_ns < trip.stop_ns):
                raise ValueError("Initial crossing trip is not active at the boundary")
            if trip.trip_id in self.seen_trip_ids:
                raise ValueError("Duplicate crossing trip")
            self.seen_trip_ids.add(trip.trip_id)
            self.riding.add(trip.trip_id)
            self._push(trip.stop_ns, 0, "return", trip)
        self._check_invariants()

    def _validate_trip(self, trip: TripRequest) -> None:
        if trip.stop_ns <= trip.start_ns:
            raise ValueError("Nonpositive trip duration")
        if not (0 <= trip.origin < self.n and 0 <= trip.destination < self.n):
            raise ValueError("Trip endpoint outside the physical station universe")

    def _push(self, time_ns: int, priority: int, kind: str, payload: Any) -> None:
        if time_ns < self.now_ns:
            raise ValueError("Cannot schedule an event into the past")
        self.serial += 1
        heapq.heappush(self.events, (int(time_ns), priority, self.serial, kind, payload))

    def add_requests(self, requests: list[TripRequest]) -> None:
        # Canonical order is frozen, not dependent on a treatment's outcomes.
        for trip in sorted(requests, key=lambda t: (t.start_ns, t.trip_id)):
            self._validate_trip(trip)
            if trip.start_ns <= self.now_ns:
                raise ValueError("Initial state already contains events at its boundary")
            if trip.trip_id in self.seen_trip_ids:
                raise ValueError(f"Duplicate trip ID: {trip.trip_id}")
            self.seen_trip_ids.add(trip.trip_id)
            self._push(trip.start_ns, 2, "pickup", trip)

    def _admit_waiting_returns(self, station: int) -> None:
        # No teleportation or rerouting: the customer waits at the requested dock.
        while self.waiting[station] and self.inventory[station] < self.capacities[station]:
            self.waiting[station].popleft()
            self.waiting_count -= 1
            self.inventory[station] += 1
            self.counts["eventually_docked_after_failed_return"] += 1

    def _pickup(self, trip: TripRequest) -> None:
        self.counts["observed_pickup_requests"] += 1
        if self.inventory[trip.origin] == 0:
            self.counts["unserved_observed_pickup_requests"] += 1
            # No bike left: in particular, do not schedule its historical return.
            return
        self.inventory[trip.origin] -= 1
        self.riding.add(trip.trip_id)
        self.counts["successful_observed_pickups"] += 1
        self._push(trip.stop_ns, 0, "return", trip)
        self._admit_waiting_returns(trip.origin)

    def _return(self, trip: TripRequest) -> None:
        if trip.trip_id not in self.riding:
            raise AssertionError("A return has no accepted pickup or initial crossing trip")
        self.riding.remove(trip.trip_id)
        self.counts["observed_return_attempts"] += 1
        if self.inventory[trip.destination] < self.capacities[trip.destination]:
            self.inventory[trip.destination] += 1
            self.counts["successful_first_return_attempts"] += 1
        else:
            self.counts["unserved_observed_return_requests"] += 1
            self.waiting[trip.destination].append(trip.trip_id)
            self.waiting_count += 1

    def dispatch_route(self, truck_index: int, route: list[int], targets: Any) -> None:
        """Dispatch one route from the current truck position using current-state service.

        Targets must be integer stocks fixed by the calling policy at dispatch.
        Arrival service moves towards that target using only then-current stock,
        truck load, and capacity. This is a causal execution heuristic, not the
        static two-stage recourse optimizer. No optimizer is called by this file.
        """
        if self.distance_matrix is None:
            raise ValueError("Truck travel distances have not been supplied")
        truck = self.trucks[truck_index]
        if truck.busy:
            raise ValueError("Cannot reset a truck still travelling or handling bikes")
        target = integer_vector(targets, "truck service targets")
        if target.shape != self.inventory.shape or np.any(target < 0) or np.any(target > self.capacities):
            raise ValueError("Targets violate station capacity")
        if len(route) != len(set(route)) or any(not (0 <= s < self.n) for s in route):
            raise ValueError("Dispatch requires a simple route of physical station indices")
        if not route:
            return
        truck.pending = deque(int(s) for s in route)
        truck.targets = target.copy()
        truck.busy = True
        self._next_truck_leg(truck_index)

    def _next_truck_leg(self, k: int) -> None:
        truck = self.trucks[k]
        if truck.pending:
            destination = truck.pending.popleft()
        elif truck.position != self.n:
            destination = self.n
        else:
            truck.busy = False
            return
        distance = float(self.distance_matrix[truck.position, destination])
        travel_ns = max(1, int(round(distance / self.speed_kmh * 3600 * NS_PER_SECOND)))
        truck.moving_to = destination
        truck.departed_ns = self.now_ns
        truck.arrival_ns = self.now_ns + travel_ns
        self._push(truck.arrival_ns, 1, "truck_arrival", (k, distance))

    def _truck_arrival(self, k: int, distance: float) -> None:
        truck = self.trucks[k]
        truck.position = int(truck.moving_to)
        truck.moving_to = truck.departed_ns = truck.arrival_ns = None
        truck.distance_km += distance
        if truck.position == self.n:
            # A depot is not a free source or sink. Remaining load stays onboard.
            truck.busy = False
            return
        desired = int(truck.targets[truck.position] - self.inventory[truck.position])
        truck.service_direction = int(np.sign(desired))
        truck.service_remaining = abs(desired)
        self._push(self.now_ns + self.stop_ns + self.handling_ns, 1, "truck_transfer", k)

    def _truck_transfer(self, k: int) -> None:
        truck = self.trucks[k]
        station, direction = truck.position, truck.service_direction
        feasible = truck.service_remaining > 0 and (
            (direction > 0 and truck.load > 0 and self.inventory[station] < self.capacities[station])
            or (direction < 0 and truck.load < truck.capacity and self.inventory[station] > 0))
        if feasible:
            self.inventory[station] += direction
            truck.load -= direction
            truck.service_remaining -= 1
            self.counts["truck_delivered_bikes" if direction > 0 else "truck_loaded_bikes"] += 1
            if direction < 0:
                self._admit_waiting_returns(station)
        if feasible and truck.service_remaining > 0:
            self._push(self.now_ns + self.handling_ns, 1, "truck_transfer", k)
        else:
            truck.service_remaining = 0
            self._next_truck_leg(k)

    def _check_invariants(self) -> None:
        current = (int(self.inventory.sum()) + len(self.riding) + self.waiting_count
                   + sum(t.load for t in self.trucks))
        error = abs(current - self.fleet)
        self.max_abs_fleet_error = max(self.max_abs_fleet_error, error)
        self.invariant_checks += 1
        if error or np.any(self.inventory < 0) or np.any(self.inventory > self.capacities):
            raise AssertionError("Fleet conservation or station capacity failed")
        if any(t.load < 0 or t.load > t.capacity for t in self.trucks):
            raise AssertionError("Truck capacity failed")
        if self.waiting_count != sum(len(queue) for queue in self.waiting):
            raise AssertionError("Failed-return queue accounting failed")

    def advance(self, until_ns: int) -> dict[str, Any]:
        if until_ns < self.now_ns:
            raise ValueError("Cannot move replay time backwards")
        while self.events and self.events[0][0] <= until_ns:
            time_ns, _, _, kind, payload = heapq.heappop(self.events)
            self.now_ns = time_ns
            if kind == "pickup":
                self._pickup(payload)
            elif kind == "return":
                self._return(payload)
            elif kind == "truck_arrival":
                self._truck_arrival(*payload)
            elif kind == "truck_transfer":
                self._truck_transfer(payload)
            else:
                raise AssertionError(f"Unknown event {kind}")
            self._check_invariants()
        self.now_ns = int(until_ns)
        self._check_invariants()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        completed_distance = sum(t.distance_km for t in self.trucks)
        partial_distance = 0.0
        for truck in self.trucks:
            if truck.moving_to is not None:
                elapsed = self.now_ns - truck.departed_ns
                partial_distance += (elapsed / (truck.arrival_ns - truck.departed_ns)
                                     * self.distance_matrix[truck.position, truck.moving_to])
        return {
            "timestamp": str(pd.Timestamp(self.now_ns)),
            "station_bikes": int(self.inventory.sum()),
            "customer_riding_bikes": len(self.riding),
            "customer_waiting_to_return_bikes": self.waiting_count,
            "truck_bikes": sum(t.load for t in self.trucks),
            "fleet_bikes": self.fleet,
            "unserved_observed_pickup_requests": self.counts["unserved_observed_pickup_requests"],
            "unserved_observed_return_requests": self.counts["unserved_observed_return_requests"],
            "observed_pickup_requests": self.counts["observed_pickup_requests"],
            "observed_return_attempts": self.counts["observed_return_attempts"],
            "truck_distance_km": float(completed_distance + partial_distance),
            "trucks_in_motion": sum(t.moving_to is not None for t in self.trucks),
            "max_abs_fleet_error_bikes": self.max_abs_fleet_error,
        }


def load_day(start: pd.Timestamp, end: pd.Timestamp, v7_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    signal_path = ROOT / "data/DIVVY/DIVVY.npz"
    capacity_path = HERE.parent / "paper_alns_v3/results/station_capacities.csv"
    trips_path = v7_root / "accepted_july_trips_v7.csv.gz"
    with np.load(signal_path, allow_pickle=False) as signal:
        indices = np.flatnonzero(signal["state_times_ns"] == start.value)
        if len(indices) != 1:
            raise ValueError("Start must match one reconstructed state boundary")
        initial = signal["data"][int(indices[0]), :, 0].copy()
        station_ids = signal["node_ids"][1:].astype(str)
    if len(station_ids) != 567:
        raise ValueError("This validation is explicitly the complete 567-station network")
    capacities = pd.read_csv(capacity_path, dtype={"station_id": str})
    if not np.array_equal(capacities.station_id.to_numpy(str), station_ids):
        raise ValueError("Capacity order does not match inventory station order")
    station_map = {sid: i for i, sid in enumerate(station_ids)}
    frame = pd.read_csv(trips_path, usecols=["trip_id", "start_time", "stop_time", "from_station_id", "to_station_id"],
                        dtype={"trip_id": str, "from_station_id": str, "to_station_id": str})
    frame["start_time"] = pd.to_datetime(frame.start_time)
    frame["stop_time"] = pd.to_datetime(frame.stop_time)
    crossing_frame = frame[(frame.start_time <= start) & (frame.stop_time > start)]
    request_frame = frame[(frame.start_time > start) & (frame.start_time <= end)]
    def convert(rows: pd.DataFrame) -> list[TripRequest]:
        return [TripRequest(str(row.trip_id), int(row.start_time.value), int(row.stop_time.value),
                            station_map[row.from_station_id], station_map[row.to_station_id])
                for row in rows.itertuples(index=False)]
    crossing, requests = convert(crossing_frame), convert(request_frame)
    if len(crossing) != int(initial[0]):
        raise ValueError("Cross-boundary trips do not reconcile with initial customer node 0")
    payload = dict(inventory=initial[1:], capacities=capacities.station_capacity.to_numpy(),
                   start_ns=start.value, crossing_trips=crossing, requests=requests)
    audit = {
        "start": str(start), "end_inclusive": str(end), "station_count": len(station_ids),
        "initial_station_bikes": int(initial[1:].sum()), "initial_customer_riding_bikes": len(crossing),
        "initial_fleet_bikes": int(initial.sum()), "observed_pickup_requests_in_window": len(requests),
        "initial_boundary_pickups_already_in_state": int((frame.start_time == start).sum()),
        "timing_rule": "State includes initial-boundary events; replay (start,end] without duplicate boundary events",
        "event_timestamp_resolution": "Source timestamps retained; accepted Divvy records have minute granularity",
        "zero_or_negative_duration_rows_in_source": int((frame.stop_time <= frame.start_time).sum()),
        "source_hashes": {str(p): sha256(p) for p in (signal_path, capacity_path, trips_path)},
        "inferred_operator_relocations_replayed": False,
        "historical_inventory_reset_after_start": False,
        "capacity_policy": "Existing full-period reconstructed maximum; retrospective capacity, not measured docks",
    }
    return payload, audit


def run_validation(day: str, output: Path, v7_root: Path) -> dict[str, Any]:
    start = pd.Timestamp(day).normalize()
    end = start + pd.Timedelta(days=1)
    payload, audit = load_day(start, end, v7_root)
    output.mkdir(parents=True, exist_ok=True)
    freeze = dict(audit, execution_code_sha256=sha256(Path(__file__)),
                  validation_role="Full-network engine/data validation only; not a fifth policy comparison arm",
                  return_failure_rule="One failed return recorded; bike waits outside destination FIFO until a dock opens",
                  tie_order="Returns, truck events, pickups; within type canonical source insertion order",
                  model_comparison=False, uses_predictions=False, policy_responsive_PDST_closed_loop=False)
    (output / "protocol.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    requests = payload.pop("requests")
    engine = ObservedRequestReplay(**payload)
    engine.add_requests(requests)
    snapshots = [engine.snapshot()]
    for timestamp in pd.date_range(start + pd.Timedelta(minutes=10), end, freq="10min"):
        snapshots.append(engine.advance(timestamp.value))
    pd.DataFrame(snapshots).to_csv(output / "physical_state_10min.csv", index=False)
    np.savez_compressed(output / "terminal_state.npz", inventory=engine.inventory,
                        waiting_by_station=np.array([len(q) for q in engine.waiting], dtype=np.int64))
    result = dict(freeze, status="completed", final_state=engine.snapshot(),
                  event_counters=dict(engine.counts), invariant_checks=engine.invariant_checks,
                  max_abs_fleet_error_bikes=engine.max_abs_fleet_error,
                  pending_events_after_end=len(engine.events),
                  interpretation="Physical execution validation only. Counts are unserved observed requests under an artificial no-dispatch replay, not latent real-world unmet demand or B/C benefits.")
    (output / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", default="2017-07-27")
    parser.add_argument("--output", type=Path, default=HERE / "results/event_validation")
    parser.add_argument("--v7-root", type=Path, default=DEFAULT_V7)
    args = parser.parse_args()
    result = run_validation(args.day, args.output, args.v7_root)
    print(json.dumps({"status": result["status"], "station_count": result["station_count"],
                      "invariant_checks": result["invariant_checks"],
                      "max_abs_fleet_error_bikes": result["max_abs_fleet_error_bikes"],
                      "output": str(args.output)}, ensure_ascii=False))
