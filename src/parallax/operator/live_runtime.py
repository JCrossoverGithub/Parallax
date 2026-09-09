"""Production runtime execution for operator-owned live sensor sessions."""

from collections.abc import Callable
from threading import Event

from parallax.operator.live import OperatorLiveConfiguration
from parallax.runtime import (
    LiveRuntimeSummary,
    PacketPredictionPipeline,
    RuntimePredictionEvent,
    RuntimeScorer,
    run_live_packet_predictions,
)
from parallax.sensor import (
    CaptureInterface,
    LiveEthernetCapture,
    LivePacketSource,
    resolve_capture_interface,
)

LiveCaptureFactory = Callable[
    [CaptureInterface],
    LiveEthernetCapture,
]


class OperatorLiveRuntimeExecutor:
    """Compose one operator live session into the shared runtime pipeline."""

    __slots__ = (
        "_capture_factory",
        "_interface_resolver",
        "_scorer",
    )

    def __init__(
        self,
        scorer: RuntimeScorer,
        *,
        interface_resolver: Callable[[str], CaptureInterface] = resolve_capture_interface,
        capture_factory: LiveCaptureFactory = LiveEthernetCapture,
    ) -> None:
        self._scorer = scorer
        self._interface_resolver = interface_resolver
        self._capture_factory = capture_factory

    def __call__(
        self,
        run_id: str,
        configuration: OperatorLiveConfiguration,
        stop_event: Event,
        handle_event: Callable[
            [RuntimePredictionEvent],
            None,
        ],
    ) -> LiveRuntimeSummary:
        """Run one live sensor until the operator requests termination."""
        interface = self._interface_resolver(configuration.interface)
        capture = self._capture_factory(interface)
        source = LivePacketSource(capture)

        pipeline = PacketPredictionPipeline(
            run_id=run_id,
            capture_id=(f"live:{interface.name}:{run_id}"),
            scorer=self._scorer,
            flow_config=configuration.flow_config(),
        )

        return run_live_packet_predictions(
            source,
            pipeline=pipeline,
            stop_requested=stop_event.is_set,
            handle_event=handle_event,
        )
