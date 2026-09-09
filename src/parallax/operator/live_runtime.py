"""Production execution for operator-owned live sensor sessions."""

from collections.abc import Callable
from pathlib import Path
from threading import Event

from parallax.data import RuntimeFlowCapacityError
from parallax.operator.live import (
    OperatorLiveConfiguration,
    OperatorLiveExecutionError,
)
from parallax.runtime import (
    LiveRuntimeSummary,
    PacketPredictionPipeline,
    RuntimePredictionEvent,
    RuntimeScorer,
    run_live_packet_predictions,
)
from parallax.runtime.live import RuntimePacketSource
from parallax.sensor import (
    DEFAULT_SENSOR_IPC_SOCKET_PATH,
    SensorIpcPacketSource,
)
from parallax.sensor.ipc_client import SensorIpcClientError

LivePacketSourceFactory = Callable[
    [Path, str],
    RuntimePacketSource,
]


def _default_source_factory(
    socket_path: Path,
    interface: str,
) -> RuntimePacketSource:
    return SensorIpcPacketSource(
        socket_path,
        interface,
    )


class OperatorLiveRuntimeExecutor:
    """Run operator live inference through the local sensor IPC service."""

    __slots__ = (
        "_scorer",
        "_sensor_socket_path",
        "_source_factory",
    )

    def __init__(
        self,
        scorer: RuntimeScorer,
        *,
        sensor_socket_path: str | Path = (DEFAULT_SENSOR_IPC_SOCKET_PATH),
        source_factory: LivePacketSourceFactory = (_default_source_factory),
    ) -> None:
        self._scorer = scorer
        self._sensor_socket_path = Path(sensor_socket_path)
        self._source_factory = source_factory

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
        try:
            source = self._source_factory(
                self._sensor_socket_path,
                configuration.interface,
            )

            pipeline = PacketPredictionPipeline(
                run_id=run_id,
                capture_id=(f"live:{configuration.interface}:{run_id}"),
                scorer=self._scorer,
                flow_config=configuration.flow_config(),
            )

            return run_live_packet_predictions(
                source,
                pipeline=pipeline,
                stop_requested=stop_event.is_set,
                handle_event=handle_event,
            )
        except RuntimeFlowCapacityError as error:
            raise OperatorLiveExecutionError(
                str(error),
                code="flow_capacity_exceeded",
            ) from error
        except SensorIpcClientError as error:
            raise OperatorLiveExecutionError(
                str(error),
                code=(error.code if error.code is not None else "sensor_ipc_error"),
            ) from error
