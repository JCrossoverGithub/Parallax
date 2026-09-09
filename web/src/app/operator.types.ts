export type ReplayState = 'created' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';

export type LiveState = 'starting' | 'running' | 'stopping' | 'completed' | 'failed';

export interface ModelIdentity {
  model_bundle_sha256: string;
  calibration_artifact_sha256: string;
  feature_artifact_sha256: string;
  split_manifest_sha256: string;
}

export interface HealthResponse {
  status: string;
  active_model: ModelIdentity;
  sessions: {
    total: number;
    active: number;
  };
}

export interface ReplayFailure {
  code: string;
  message: string;
}

export interface ReplaySnapshot {
  run_id: string;
  source_id: string;
  source_sha256: string;
  state: ReplayState;
  time_scale: number | null;
  event_count: number;
  retained_event_count: number;
  failure: ReplayFailure | null;
}

export interface ReplayHistoryRecord {
  run_id: string;
  source_id: string;
  source_sha256: string;
  state: ReplayState;
  time_scale: number | null;
  event_count: number;
  failure: ReplayFailure | null;
}

export interface ReplayHistoryResponse {
  replays: ReplayHistoryRecord[];
}

export interface LiveInterface {
  index: number;
  name: string;
}

export interface LiveInterfacesResponse {
  interfaces: LiveInterface[];
}

export interface LiveConfiguration {
  interface: string;
  stale_after_seconds: number;
  max_tracked_flows: number;
}

export interface LiveFailure {
  code: string;
  message: string;
}

export interface LiveSessionSnapshot {
  run_id: string;
  state: LiveState;
  configuration: LiveConfiguration;
  failure: LiveFailure | null;
}

export interface RuntimePredictionEvent {
  schema_version: string;
  run_id: string;

  window: {
    window_id: string;
    capture_id: string;
    flow_id: string;
    window_index: number;
    start_offset_seconds: number;
    end_offset_seconds: number;
    packet_count: number;
  };

  classification: {
    category_order: string[];
    class_probabilities: number[];
    predicted_class_index: number;
    predicted_category: string;
    raw_confidence: number;
  };

  uncertainty: {
    relative_mahalanobis_distance: number;
    ood_score: number;
  };

  provenance: ModelIdentity;
}

export interface PredictionEventsResponse {
  run_id: string;
  events: RuntimePredictionEvent[];
}

export type ReplayHistoryEventsResponse = PredictionEventsResponse;

export type LiveEventsResponse = PredictionEventsResponse & {
  last_sequence: number;
  state: LiveState;
};

export interface StartReplayRequest {
  capture: string;
  time_scale: number | null;
}

export interface StartLiveRequest {
  interface: string;
}

export interface ControlResponse {
  run_id: string;
  action: string;
  accepted: boolean;
}

export interface LiveControlResponse extends ControlResponse {
  state: LiveState;
}

export interface ReplayTerminalEvent {
  run_id: string;
  state: ReplayState;
}

export interface LiveTerminalEvent {
  run_id: string;
  state: LiveState;
}
