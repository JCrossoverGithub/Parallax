export type ReplayState =
  | 'created'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled';

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

export interface ReplayHistoryEventsResponse {
  run_id: string;
  events: RuntimePredictionEvent[];
}

export interface StartReplayRequest {
  capture: string;
  time_scale: number | null;
}

export interface ControlResponse {
  run_id: string;
  action: string;
  accepted: boolean;
}

export interface ReplayTerminalEvent {
  run_id: string;
  state: ReplayState;
}
