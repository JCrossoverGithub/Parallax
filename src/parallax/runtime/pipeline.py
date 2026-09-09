"""Incremental packet-to-prediction runtime pipeline."""

from typing import Protocol

from parallax.data import (
    BidirectionalFlowTracker,
    IncrementalWindowTracker,
    ObservationWindow,
    PacketMetadata,
    WindowExtractionConfig,
)
from parallax.features import VnatWindowFeature, calculate_vnat_window_feature
from parallax.modeling.runtime import PrototypeRuntimePrediction
from parallax.runtime.events import RuntimePredictionEvent


class RuntimePipelineError(ValueError):
    """Raised when packet prediction runtime state is invalid."""


class RuntimeScorer(Protocol):
    """Structural contract for frozen runtime feature scorers."""

    def score_feature(
        self,
        feature: VnatWindowFeature,
    ) -> PrototypeRuntimePrediction:
        """Score one runtime feature."""


class PacketPredictionPipeline:
    """Incrementally transform packet metadata into prediction events."""

    __slots__ = (
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
        capture_name: str,
        scorer: RuntimeScorer,
        window_config: WindowExtractionConfig | None = None,
    ) -> None:
        if not run_id:
            raise RuntimePipelineError("runtime pipeline run ID must not be empty")

        self._run_id = run_id
        self._scorer = scorer
        self._flow_tracker = BidirectionalFlowTracker()
        self._window_tracker = IncrementalWindowTracker(
            capture_name,
            config=window_config,
        )
        self._finished = False

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
        windows: tuple[ObservationWindow, ...],
    ) -> tuple[RuntimePredictionEvent, ...]:
        events: list[RuntimePredictionEvent] = []

        for window in windows:
            feature = calculate_vnat_window_feature(window)
            prediction = self._scorer.score_feature(feature)
            events.append(
                RuntimePredictionEvent(
                    run_id=self._run_id,
                    prediction=prediction,
                )
            )

        return tuple(events)
