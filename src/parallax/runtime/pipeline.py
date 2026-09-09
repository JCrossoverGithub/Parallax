"""Incremental packet-to-prediction runtime pipeline."""

from dataclasses import dataclass
from typing import Protocol

from parallax.data import (
    IncrementalWindowTracker,
    PacketMetadata,
    RuntimeFlowTracker,
    RuntimeObservationWindow,
    WindowExtractionConfig,
)
from parallax.features import (
    RuntimeWindowFeature,
    calculate_runtime_window_feature,
)
from parallax.modeling.runtime import PrototypeRuntimePrediction
from parallax.runtime.events import RuntimePredictionEvent


class RuntimePipelineError(ValueError):
    """Raised when packet prediction runtime state is invalid."""


@dataclass(frozen=True, slots=True)
class RuntimePipelineStats:
    """Immutable operational counters for one prediction pipeline."""

    packets_processed: int
    tracked_flow_count: int
    prediction_events_emitted: int
    finished: bool


class RuntimeScorer(Protocol):
    """Structural contract for frozen runtime feature scorers."""

    def score_feature(
        self,
        feature: RuntimeWindowFeature,
    ) -> PrototypeRuntimePrediction:
        """Score one runtime feature."""


class PacketPredictionPipeline:
    """Incrementally transform packet metadata into prediction events."""

    __slots__ = (
        "_events_emitted",
        "_finished",
        "_flow_tracker",
        "_run_id",
        "_scorer",
        "_window_tracker",
    )

    def __init__(
        self,
        *,
        run_id: str,
        capture_id: str,
        scorer: RuntimeScorer,
        window_config: WindowExtractionConfig | None = None,
    ) -> None:
        if not run_id:
            raise RuntimePipelineError("runtime pipeline run ID must not be empty")

        self._run_id = run_id
        self._scorer = scorer
        self._flow_tracker = RuntimeFlowTracker()
        self._window_tracker = IncrementalWindowTracker(
            capture_id,
            config=window_config,
        )
        self._events_emitted = 0
        self._finished = False

    @property
    def stats(self) -> RuntimePipelineStats:
        """Return current operational counters without mutating runtime state."""
        return RuntimePipelineStats(
            packets_processed=self._flow_tracker.packet_count,
            tracked_flow_count=self._flow_tracker.tracked_flow_count,
            prediction_events_emitted=self._events_emitted,
            finished=self._finished,
        )

    def push(
        self,
        packet: PacketMetadata,
    ) -> tuple[RuntimePredictionEvent, ...]:
        """Consume one packet and emit any newly completed predictions."""
        if self._finished:
            raise RuntimePipelineError("cannot push packets after runtime pipeline is finished")

        assignment = self._flow_tracker.push(packet)
        windows = self._window_tracker.push(assignment)
        return self._score_windows(windows)

    def finish(self) -> tuple[RuntimePredictionEvent, ...]:
        """Flush final eligible windows and close the pipeline."""
        if self._finished:
            return ()

        self._finished = True
        return self._score_windows(self._window_tracker.finish())

    def _score_windows(
        self,
        windows: tuple[RuntimeObservationWindow, ...],
    ) -> tuple[RuntimePredictionEvent, ...]:
        events: list[RuntimePredictionEvent] = []

        for window in windows:
            feature = calculate_runtime_window_feature(window)
            prediction = self._scorer.score_feature(feature)

            events.append(
                RuntimePredictionEvent(
                    run_id=self._run_id,
                    prediction=prediction,
                )
            )

        self._events_emitted += len(events)
        return tuple(events)
