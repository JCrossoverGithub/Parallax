import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { OperatorApi, ReplayStreamHandlers } from './operator-api';
import { ReplayTerminalEvent, RuntimePredictionEvent } from './operator.types';

const PREDICTION: RuntimePredictionEvent = {
  schema_version: 'parallax-runtime-prediction-1',
  run_id: 'run-001',
  window: {
    window_id: 'window-001',
    capture_id: 'capture.pcap',
    flow_id: 'flow-001',
    window_index: 0,
    start_offset_seconds: 0,
    end_offset_seconds: 40,
    packet_count: 21,
  },
  classification: {
    category_order: ['Streaming', 'VoIP', 'Chat', 'C2', 'File Transfer'],
    class_probabilities: [0.7, 0.1, 0.1, 0.05, 0.05],
    predicted_class_index: 0,
    predicted_category: 'Streaming',
    raw_confidence: 0.7,
  },
  uncertainty: {
    relative_mahalanobis_distance: 1.5,
    ood_score: 0.2,
  },
  provenance: {
    model_bundle_sha256: 'a'.repeat(64),
    calibration_artifact_sha256: 'b'.repeat(64),
    feature_artifact_sha256: 'c'.repeat(64),
    split_manifest_sha256: 'd'.repeat(64),
  },
};

class FakeEventSource {
  static instances: FakeEventSource[] = [];

  readonly url: string;
  closed = false;
  onerror: (() => void) | null = null;

  private readonly listeners = new Map<string, Array<(event: MessageEvent<string>) => void>>();

  constructor(url: string | URL) {
    this.url = String(url);
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: EventListenerOrEventListenerObject | null): void {
    if (listener === null) {
      return;
    }

    const callback = (event: MessageEvent<string>): void => {
      if (typeof listener === 'function') {
        listener(event);
      } else {
        listener.handleEvent(event);
      }
    };

    const listeners = this.listeners.get(type) ?? [];
    listeners.push(callback);
    this.listeners.set(type, listeners);
  }

  emit(type: string, payload: object): void {
    const event = new MessageEvent<string>(type, {
      data: JSON.stringify(payload),
    });

    for (const listener of this.listeners.get(type) ?? []) {
      listener(event);
    }
  }

  fail(): void {
    this.onerror?.();
  }

  close(): void {
    this.closed = true;
  }
}

describe('OperatorApi', () => {
  let api: OperatorApi;
  let http: HttpTestingController;

  beforeEach(() => {
    FakeEventSource.instances = [];

    vi.stubGlobal('EventSource', FakeEventSource as unknown as typeof EventSource);

    TestBed.configureTestingModule({
      providers: [OperatorApi, provideHttpClient(), provideHttpClientTesting()],
    });

    api = TestBed.inject(OperatorApi);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
    vi.unstubAllGlobals();
  });

  it('loads service health', () => {
    api.health().subscribe((response) => {
      expect(response.status).toBe('ok');
    });

    const request = http.expectOne('/health');

    expect(request.request.method).toBe('GET');

    request.flush({
      status: 'ok',
      active_model: {},
      sessions: {
        total: 0,
        active: 0,
      },
    });
  });

  it('starts a replay', () => {
    api
      .startReplay({
        capture: 'capture.pcap',
        time_scale: null,
      })
      .subscribe((response) => {
        expect(response.run_id).toBe('run-001');
      });

    const request = http.expectOne('/api/v1/replays');

    expect(request.request.method).toBe('POST');
    expect(request.request.body).toEqual({
      capture: 'capture.pcap',
      time_scale: null,
    });

    request.flush({
      run_id: 'run-001',
    });
  });

  it('loads a replay snapshot', () => {
    api.getReplay('run 001').subscribe();

    const request = http.expectOne('/api/v1/replays/run%20001');

    expect(request.request.method).toBe('GET');
    request.flush({});
  });

  it.each([
    ['pause', 'pauseReplay'],
    ['resume', 'resumeReplay'],
    ['cancel', 'cancelReplay'],
  ] as const)('sends the %s replay control', (action, method) => {
    api[method]('run-001').subscribe();

    const request = http.expectOne(`/api/v1/replays/run-001/${action}`);

    expect(request.request.method).toBe('POST');
    expect(request.request.body).toEqual({});

    request.flush({
      run_id: 'run-001',
      action,
      accepted: true,
    });
  });

  it('delivers prediction events from SSE', () => {
    const prediction = vi.fn();
    const handlers: ReplayStreamHandlers = {
      prediction,
      terminal: vi.fn(),
      streamError: vi.fn(),
      connectionError: vi.fn(),
    };

    api.openReplayStream('run 001', handlers);

    const source = FakeEventSource.instances[0];

    expect(source.url).toBe('/api/v1/replays/run%20001/stream');

    source.emit('prediction', PREDICTION);

    expect(prediction).toHaveBeenCalledWith(PREDICTION);
  });

  it('delivers terminal events and closes SSE', () => {
    const terminal = vi.fn();

    api.openReplayStream('run-001', {
      prediction: vi.fn(),
      terminal,
      streamError: vi.fn(),
      connectionError: vi.fn(),
    });

    const source = FakeEventSource.instances[0];

    const event: ReplayTerminalEvent = {
      run_id: 'run-001',
      state: 'completed',
    };

    source.emit('replay-terminal', event);

    expect(terminal).toHaveBeenCalledWith(event);
    expect(source.closed).toBe(true);
  });

  it('delivers structured stream errors and closes SSE', () => {
    const streamError = vi.fn();

    api.openReplayStream('run-001', {
      prediction: vi.fn(),
      terminal: vi.fn(),
      streamError,
      connectionError: vi.fn(),
    });

    const source = FakeEventSource.instances[0];

    source.emit('stream-error', {
      detail: 'cursor expired',
    });

    expect(streamError).toHaveBeenCalledWith('cursor expired');
    expect(source.closed).toBe(true);
  });

  it('reports EventSource connection errors', () => {
    const connectionError = vi.fn();

    api.openReplayStream('run-001', {
      prediction: vi.fn(),
      terminal: vi.fn(),
      streamError: vi.fn(),
      connectionError,
    });

    FakeEventSource.instances[0].fail();

    expect(connectionError).toHaveBeenCalledOnce();
  });

  it('loads replay history with the default limit', () => {
    api.listHistory().subscribe((response) => {
      expect(response.replays).toEqual([]);
    });

    const request = http.expectOne('/api/v1/history?limit=100');

    expect(request.request.method).toBe('GET');

    request.flush({
      replays: [],
    });
  });

  it('loads replay history with an explicit limit', () => {
    api.listHistory(25).subscribe();

    const request = http.expectOne('/api/v1/history?limit=25');

    expect(request.request.method).toBe('GET');

    request.flush({
      replays: [],
    });
  });

  it('loads one historical replay', () => {
    api.getHistoryReplay('run 001').subscribe();

    const request = http.expectOne('/api/v1/history/run%20001');

    expect(request.request.method).toBe('GET');

    request.flush({
      run_id: 'run 001',
      source_id: 'capture.pcap',
      source_sha256: 'a'.repeat(64),
      state: 'completed',
      time_scale: 1,
      event_count: 1,
      failure: null,
    });
  });

  it('loads historical prediction events', () => {
    api.getHistoryEvents('run 001').subscribe();

    const request = http.expectOne('/api/v1/history/run%20001/events');

    expect(request.request.method).toBe('GET');

    request.flush({
      run_id: 'run 001',
      events: [],
    });
  });
});

describe('OperatorApi live sensor', () => {
  let api: OperatorApi;
  let http: HttpTestingController;

  beforeEach(() => {
    FakeEventSource.instances = [];

    vi.stubGlobal('EventSource', FakeEventSource as unknown as typeof EventSource);

    TestBed.configureTestingModule({
      providers: [OperatorApi, provideHttpClient(), provideHttpClientTesting()],
    });

    api = TestBed.inject(OperatorApi);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
    vi.unstubAllGlobals();
  });

  it('loads live capture interfaces', () => {
    api.listLiveInterfaces().subscribe((response) => {
      expect(response.interfaces).toEqual([
        {
          index: 2,
          name: 'eth0',
        },
      ]);
    });

    const request = http.expectOne('/api/v1/live/interfaces');

    expect(request.request.method).toBe('GET');

    request.flush({
      interfaces: [
        {
          index: 2,
          name: 'eth0',
        },
      ],
    });
  });

  it('starts a live sensor session', () => {
    api
      .startLive({
        interface: 'eth0',
      })
      .subscribe((response) => {
        expect(response.run_id).toBe('live-001');
        expect(response.state).toBe('starting');
      });

    const request = http.expectOne('/api/v1/live');

    expect(request.request.method).toBe('POST');
    expect(request.request.body).toEqual({
      interface: 'eth0',
    });

    request.flush({
      run_id: 'live-001',
      state: 'starting',
      configuration: {
        interface: 'eth0',
        stale_after_seconds: 120,
        max_tracked_flows: 4096,
      },
      failure: null,
    });
  });

  it('loads a live session snapshot', () => {
    api.getLive('live 001').subscribe();

    const request = http.expectOne('/api/v1/live/live%20001');

    expect(request.request.method).toBe('GET');

    request.flush({
      run_id: 'live 001',
      state: 'running',
      configuration: {
        interface: 'eth0',
        stale_after_seconds: 120,
        max_tracked_flows: 4096,
      },
      failure: null,
    });
  });

  it('loads retained live prediction events', () => {
    api.getLiveEvents('live 001').subscribe();

    const request = http.expectOne('/api/v1/live/live%20001/events');

    expect(request.request.method).toBe('GET');

    request.flush({
      run_id: 'live 001',
      events: [],
    });
  });

  it('stops a live sensor session', () => {
    api.stopLive('live 001').subscribe((response) => {
      expect(response.action).toBe('stop');
      expect(response.state).toBe('stopping');
    });

    const request = http.expectOne('/api/v1/live/live%20001/stop');

    expect(request.request.method).toBe('POST');
    expect(request.request.body).toEqual({});

    request.flush({
      run_id: 'live 001',
      action: 'stop',
      accepted: true,
      state: 'stopping',
    });
  });

  it('delivers live prediction events from SSE', () => {
    const prediction = vi.fn();

    api.openLiveStream('live 001', {
      prediction,
      terminal: vi.fn(),
      streamError: vi.fn(),
      connectionError: vi.fn(),
    });

    const source = FakeEventSource.instances.at(-1);

    expect(source).toBeDefined();
    expect(source?.url).toBe('/api/v1/live/live%20001/stream');

    source?.emit('prediction', PREDICTION);

    expect(prediction).toHaveBeenCalledWith(PREDICTION);
  });

  it('delivers live terminal events and closes SSE', () => {
    const terminal = vi.fn();

    api.openLiveStream('live-001', {
      prediction: vi.fn(),
      terminal,
      streamError: vi.fn(),
      connectionError: vi.fn(),
    });

    const source = FakeEventSource.instances.at(-1);

    const event = {
      run_id: 'live-001',
      state: 'completed',
    };

    source?.emit('live-terminal', event);

    expect(terminal).toHaveBeenCalledWith(event);
    expect(source?.closed).toBe(true);
  });

  it('shares structured stream-error behavior for live SSE', () => {
    const streamError = vi.fn();

    api.openLiveStream('live-001', {
      prediction: vi.fn(),
      terminal: vi.fn(),
      streamError,
      connectionError: vi.fn(),
    });

    const source = FakeEventSource.instances.at(-1);

    source?.emit('stream-error', {
      detail: 'live cursor expired',
    });

    expect(streamError).toHaveBeenCalledWith('live cursor expired');

    expect(source?.closed).toBe(true);
  });

  it('reports live EventSource connection failures', () => {
    const connectionError = vi.fn();

    api.openLiveStream('live-001', {
      prediction: vi.fn(),
      terminal: vi.fn(),
      streamError: vi.fn(),
      connectionError,
    });

    FakeEventSource.instances.at(-1)?.fail();

    expect(connectionError).toHaveBeenCalledOnce();
  });
});
