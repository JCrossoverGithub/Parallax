import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Observable, of, throwError } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { App } from './app';
import { LiveStreamHandlers, OperatorApi, ReplayStreamHandlers } from './operator-api';
import {
  HealthResponse,
  LiveEventsResponse,
  LiveInterfacesResponse,
  LiveSessionSnapshot,
  ReplayHistoryResponse,
  ReplaySnapshot,
  RuntimePredictionEvent,
} from './operator.types';

const HEALTH: HealthResponse = {
  status: 'ok',
  active_model: {
    model_bundle_sha256: 'a'.repeat(64),
    calibration_artifact_sha256: 'b'.repeat(64),
    feature_artifact_sha256: 'c'.repeat(64),
    split_manifest_sha256: 'd'.repeat(64),
  },
  sessions: {
    total: 0,
    active: 0,
  },
};

const CREATED: ReplaySnapshot = {
  run_id: 'run-001',
  source_id: 'nonvpn_ssh_capture4.pcap',
  source_sha256: 'e'.repeat(64),
  state: 'created',
  time_scale: 1,
  event_count: 0,
  retained_event_count: 0,
  failure: null,
};

const RUNNING: ReplaySnapshot = {
  ...CREATED,
  state: 'running',
};

const PAUSED: ReplaySnapshot = {
  ...CREATED,
  state: 'paused',
};

const COMPLETED: ReplaySnapshot = {
  ...CREATED,
  state: 'completed',
  event_count: 1,
  retained_event_count: 1,
};

const LIVE_STARTING: LiveSessionSnapshot = {
  run_id: 'live-001',
  state: 'starting',
  configuration: {
    interface: 'eth0',
    stale_after_seconds: 120,
    max_tracked_flows: 4096,
  },
  failure: null,
};

const LIVE_RUNNING: LiveSessionSnapshot = {
  ...LIVE_STARTING,
  state: 'running',
};

const LIVE_COMPLETED: LiveSessionSnapshot = {
  ...LIVE_STARTING,
  state: 'completed',
};

const PREDICTION: RuntimePredictionEvent = {
  schema_version: 'parallax-runtime-prediction-1',
  run_id: 'run-001',
  window: {
    window_id: 'window-001',
    capture_id: 'nonvpn_ssh_capture4.pcap',
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
  provenance: HEALTH.active_model,
};

class FakeOperatorApi {
  readonly health = vi.fn<() => Observable<HealthResponse>>(() => of(HEALTH));

  readonly listHistory = vi.fn<() => Observable<ReplayHistoryResponse>>(() =>
    of({
      replays: [],
    }),
  );

  readonly getHistoryReplay = vi.fn((runId: string) =>
    of({
      ...COMPLETED,
      run_id: runId,
    }),
  );

  readonly getHistoryEvents = vi.fn((runId: string) =>
    of({
      run_id: runId,
      events: [PREDICTION],
    }),
  );

  readonly listLiveInterfaces = vi.fn<() => Observable<LiveInterfacesResponse>>(() =>
    of({
      interfaces: [
        {
          index: 1,
          name: 'lo',
        },
        {
          index: 2,
          name: 'eth0',
        },
      ],
    }),
  );

  readonly getActiveLive = vi.fn(() => of<LiveSessionSnapshot | null>(null));

  readonly startLive = vi.fn(() => of(LIVE_STARTING));

  readonly getLive = vi.fn(() => of(LIVE_RUNNING));

  readonly getLiveEvents = vi.fn((runId: string): Observable<LiveEventsResponse> =>
    of({
      run_id: runId,
      events: [],
      last_sequence: 0,
      state: 'running',
    }),
  );

  readonly stopLive = vi.fn(() =>
    of({
      run_id: 'live-001',
      action: 'stop',
      accepted: true,
      state: 'stopping' as const,
    }),
  );

  readonly startReplay = vi.fn(() => of(CREATED));

  readonly getReplay = vi.fn(() => of(RUNNING));

  readonly pauseReplay = vi.fn(() =>
    of({
      run_id: 'run-001',
      action: 'pause',
      accepted: true,
    }),
  );

  readonly resumeReplay = vi.fn(() =>
    of({
      run_id: 'run-001',
      action: 'resume',
      accepted: true,
    }),
  );

  readonly cancelReplay = vi.fn(() =>
    of({
      run_id: 'run-001',
      action: 'cancel',
      accepted: true,
    }),
  );

  handlers: ReplayStreamHandlers | null = null;

  readonly source = {
    close: vi.fn(),
  };

  liveHandlers: LiveStreamHandlers | null = null;

  readonly liveSource = {
    close: vi.fn(),
  };

  readonly openLiveStream = vi.fn(
    (_runId: string, handlers: LiveStreamHandlers, _afterSequence = 0): EventSource => {
      this.liveHandlers = handlers;

      return this.liveSource as unknown as EventSource;
    },
  );

  readonly openReplayStream = vi.fn(
    (_runId: string, handlers: ReplayStreamHandlers): EventSource => {
      this.handlers = handlers;
      return this.source as unknown as EventSource;
    },
  );
}

describe('App', () => {
  let fixture: ComponentFixture<App>;
  let component: App;
  let api: FakeOperatorApi;

  beforeEach(async () => {
    vi.useFakeTimers();

    api = new FakeOperatorApi();

    await TestBed.configureTestingModule({
      imports: [App],
      providers: [
        {
          provide: OperatorApi,
          useValue: api,
        },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(App);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  afterEach(() => {
    fixture.destroy();
    vi.useRealTimers();
  });

  it('creates the dashboard and loads health', () => {
    expect(component).toBeTruthy();
    expect(api.health).toHaveBeenCalledOnce();
    expect(component.health()).toEqual(HEALTH);

    const compiled = fixture.nativeElement as HTMLElement;

    expect(compiled.querySelector('h1')?.textContent).toContain('Network Intelligence Console');

    expect(compiled.textContent).toContain('Operator online');

    expect(compiled.textContent).toContain('Start the sensor to begin classifying');
  });

  it('starts replay at configured speed and connects SSE', () => {
    component.capture = ' nonvpn_ssh_capture4.pcap ';
    component.timeScale = 2;

    component.startReplay();

    expect(api.startReplay).toHaveBeenCalledWith({
      capture: 'nonvpn_ssh_capture4.pcap',
      time_scale: 2,
    });

    expect(component.replay()).toEqual(CREATED);
    expect(api.openReplayStream).toHaveBeenCalledWith('run-001', expect.any(Object));
  });

  it('starts maximum-speed replay with a null time scale', () => {
    component.maximumSpeed = true;

    component.startReplay();

    expect(api.startReplay).toHaveBeenCalledWith({
      capture: 'nonvpn_ssh_capture4.pcap',
      time_scale: null,
    });
  });

  it('ignores blank capture and duplicate busy starts', () => {
    component.capture = ' ';
    component.startReplay();

    expect(api.startReplay).not.toHaveBeenCalled();

    component.capture = 'capture.pcap';
    component.busy.set(true);
    component.startReplay();

    expect(api.startReplay).not.toHaveBeenCalled();
  });

  it('renders predictions received from SSE', () => {
    component.startReplay();

    expect(api.handlers).not.toBeNull();

    api.handlers?.prediction(PREDICTION);
    fixture.detectChanges();

    expect(component.predictions()).toEqual([PREDICTION]);
    expect(component.latestPrediction()).toEqual(PREDICTION);
    expect(component.streamConnected()).toBe(true);

    const compiled = fixture.nativeElement as HTMLElement;

    expect(compiled.textContent).toContain('Streaming');
    expect(compiled.textContent).toContain('70.0%');
    expect(compiled.textContent).toContain('0.200');
    expect(compiled.textContent).toContain('flow-001');
  });

  it('handles terminal SSE state', () => {
    api.getReplay.mockReturnValue(of(COMPLETED));

    component.startReplay();

    api.handlers?.terminal({
      run_id: 'run-001',
      state: 'completed',
    });

    expect(component.replay()).toEqual(COMPLETED);
    expect(component.streamConnected()).toBe(false);
    expect(api.health).toHaveBeenCalledTimes(3);
  });

  it('surfaces structured SSE errors', () => {
    component.startReplay();

    api.handlers?.streamError('cursor expired');

    expect(component.error()).toBe('cursor expired');
    expect(component.streamConnected()).toBe(false);
  });

  it('handles an SSE connection failure while active', () => {
    component.startReplay();
    component.streamConnected.set(true);

    api.handlers?.connectionError();

    expect(component.streamConnected()).toBe(false);
  });

  it('does not change stream state for connection failure after completion', () => {
    component.startReplay();
    component.replay.set(COMPLETED);
    component.streamConnected.set(true);

    api.handlers?.connectionError();

    expect(component.streamConnected()).toBe(true);
  });

  it('polls replay state while a replay is active', () => {
    component.startReplay();

    vi.advanceTimersByTime(250);

    expect(api.getReplay).toHaveBeenCalledWith('run-001');
  });

  it('stops polling after a terminal snapshot', () => {
    api.getReplay.mockReturnValue(of(COMPLETED));

    component.startReplay();

    vi.advanceTimersByTime(250);
    const calls = api.getReplay.mock.calls.length;

    vi.advanceTimersByTime(1000);

    expect(api.getReplay.mock.calls.length).toBe(calls);
  });

  it('pauses a running replay', () => {
    component.replay.set(RUNNING);
    api.getReplay.mockReturnValue(of(PAUSED));

    component.pause();

    expect(api.pauseReplay).toHaveBeenCalledWith('run-001');
    expect(component.replay()).toEqual(PAUSED);
  });

  it('resumes a paused replay', () => {
    component.replay.set(PAUSED);
    api.getReplay.mockReturnValue(of(RUNNING));

    component.resume();

    expect(api.resumeReplay).toHaveBeenCalledWith('run-001');
    expect(component.replay()).toEqual(RUNNING);
  });

  it('cancels an active replay', () => {
    component.replay.set(RUNNING);

    component.cancel();

    expect(api.cancelReplay).toHaveBeenCalledWith('run-001');
  });

  it('ignores control actions without a replay', () => {
    component.replay.set(null);

    component.pause();
    component.resume();
    component.cancel();

    expect(api.pauseReplay).not.toHaveBeenCalled();
    expect(api.resumeReplay).not.toHaveBeenCalled();
    expect(api.cancelReplay).not.toHaveBeenCalled();
  });

  it('surfaces start failures', () => {
    api.startReplay.mockReturnValue(
      throwError(() => ({
        error: {
          detail: 'capture does not exist',
        },
      })),
    );

    component.startReplay();

    expect(component.error()).toBe('capture does not exist');
    expect(component.busy()).toBe(false);
  });

  it('uses fallback start error text', () => {
    api.startReplay.mockReturnValue(throwError(() => ({})));

    component.startReplay();

    expect(component.error()).toBe('Unable to start replay.');
  });

  it.each([
    ['pause', 'pauseReplay', 'Unable to pause replay.'],
    ['resume', 'resumeReplay', 'Unable to resume replay.'],
    ['cancel', 'cancelReplay', 'Unable to cancel replay.'],
  ] as const)('surfaces %s control failures', (action, method, fallback) => {
    component.replay.set(RUNNING);

    api[method].mockReturnValue(throwError(() => ({})));

    component[action]();

    expect(component.error()).toBe(fallback);
  });

  it('surfaces the health connection failure', () => {
    api.health.mockReturnValue(throwError(() => new Error('offline')));

    component.loadHealth();

    expect(component.error()).toBe('Unable to reach the Parallax operator service.');
  });

  it('provides stable prediction helpers', () => {
    expect(component.trackPrediction(0, PREDICTION)).toBe('window-001');

    expect(component.probability(PREDICTION, 0)).toBe(0.7);

    expect(component.probability(PREDICTION, 999)).toBe(0);
  });

  it('closes the active EventSource on destroy', () => {
    component.startReplay();

    fixture.destroy();

    expect(api.source.close).toHaveBeenCalled();
  });

  it('loads persisted replay history', () => {
    api.listHistory.mockReturnValue(
      of({
        replays: [
          {
            run_id: 'history-001',
            source_id: 'capture.pcap',
            source_sha256: 'f'.repeat(64),
            state: 'completed',
            time_scale: 1,
            event_count: 1,
            failure: null,
          },
        ],
      }),
    );

    component.loadHistory();

    expect(component.history()).toEqual([
      {
        run_id: 'history-001',
        source_id: 'capture.pcap',
        source_sha256: 'f'.repeat(64),
        state: 'completed',
        time_scale: 1,
        event_count: 1,
        failure: null,
      },
    ]);
  });

  it('surfaces replay history loading failures', () => {
    api.listHistory.mockReturnValue(throwError(() => new Error('offline')));

    component.loadHistory();

    expect(component.error()).toBe('Unable to load replay history.');
  });

  it('opens a persisted replay and restores its predictions', () => {
    api.getHistoryReplay.mockReturnValue(
      of({
        ...COMPLETED,
        run_id: 'history-001',
      }),
    );

    api.getHistoryEvents.mockReturnValue(
      of({
        run_id: 'history-001',
        events: [PREDICTION],
      }),
    );

    component.openHistory('history-001');

    expect(api.getHistoryReplay).toHaveBeenCalledWith('history-001');
    expect(api.getHistoryEvents).toHaveBeenCalledWith('history-001');

    expect(component.selectedHistoryRunId()).toBe('history-001');

    expect(component.replay()).toEqual({
      ...COMPLETED,
      run_id: 'history-001',
      retained_event_count: COMPLETED.event_count,
    });

    expect(component.predictions()).toEqual([PREDICTION]);

    expect(component.historyBusy()).toBe(false);
  });

  it('does not open history while a live replay is active', () => {
    component.replay.set(RUNNING);

    api.getHistoryReplay.mockClear();
    api.getHistoryEvents.mockClear();

    component.openHistory('history-001');

    expect(api.getHistoryReplay).not.toHaveBeenCalled();
    expect(api.getHistoryEvents).not.toHaveBeenCalled();
  });

  it('does not open another history item while history is loading', () => {
    component.historyBusy.set(true);

    api.getHistoryReplay.mockClear();
    api.getHistoryEvents.mockClear();

    component.openHistory('history-001');

    expect(api.getHistoryReplay).not.toHaveBeenCalled();
    expect(api.getHistoryEvents).not.toHaveBeenCalled();
  });

  it('surfaces structured history opening failures', () => {
    api.getHistoryReplay.mockReturnValue(
      throwError(() => ({
        error: {
          detail: 'persisted replay does not exist',
        },
      })),
    );

    component.openHistory('missing');

    expect(component.error()).toBe('persisted replay does not exist');
    expect(component.historyBusy()).toBe(false);
  });

  it('uses fallback text for history opening failures', () => {
    api.getHistoryReplay.mockReturnValue(throwError(() => ({})));

    component.openHistory('missing');

    expect(component.error()).toBe('Unable to open replay history.');
    expect(component.historyBusy()).toBe(false);
  });

  it('treats persisted sessions as read-only', () => {
    component.replay.set(RUNNING);
    component.selectedHistoryRunId.set('history-001');

    api.pauseReplay.mockClear();
    api.resumeReplay.mockClear();
    api.cancelReplay.mockClear();

    expect(component.active()).toBe(false);
    expect(component.canPause()).toBe(false);

    component.pause();
    component.resume();
    component.cancel();

    expect(api.pauseReplay).not.toHaveBeenCalled();
    expect(api.resumeReplay).not.toHaveBeenCalled();
    expect(api.cancelReplay).not.toHaveBeenCalled();
  });

  it('allows starting a new replay after viewing history', () => {
    component.selectedHistoryRunId.set('history-001');
    component.replay.set(COMPLETED);

    component.startReplay();

    expect(component.selectedHistoryRunId()).toBeNull();
    expect(api.startReplay).toHaveBeenCalledOnce();
  });

  it('opens an investigation for a selected prediction', () => {
    component.livePredictions.set([
      {
        ...PREDICTION,
        run_id: 'live-001',
      },
    ]);

    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;

    const row = compiled.querySelector('.event-row') as HTMLElement | null;

    expect(row).not.toBeNull();

    row?.click();
    fixture.detectChanges();

    expect(component.selectedPrediction()).not.toBeNull();
    expect(component.selectedPrediction()?.window.window_id).toBe('window-001');

    const drawer = compiled.querySelector('.investigation-drawer');

    expect(drawer).not.toBeNull();
    expect(drawer?.textContent).toContain('EVENT INVESTIGATION');
    expect(drawer?.textContent).toContain('flow-001');
    expect(drawer?.textContent).toContain('RELATIVE MD');
    expect(drawer?.textContent).toContain('1.500');
  });

  it('closes an active prediction investigation', () => {
    component.selectPrediction(PREDICTION);
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('.investigation-drawer')).not.toBeNull();

    component.closePrediction();
    fixture.detectChanges();

    expect(component.selectedPrediction()).toBeNull();

    expect(fixture.nativeElement.querySelector('.investigation-drawer')).toBeNull();
  });

  it('clears investigation state when changing workspace', () => {
    component.selectPrediction(PREDICTION);

    component.selectWorkspace('replay');

    expect(component.selectedPrediction()).toBeNull();
  });

  it('checks for an active live session on startup', () => {
    expect(api.getActiveLive).toHaveBeenCalledOnce();
  });

  it('restores an active live session and retained events', () => {
    const livePrediction: RuntimePredictionEvent = {
      ...PREDICTION,
      run_id: 'live-001',
      window: {
        ...PREDICTION.window,
        capture_id: 'live:eth0:live-001',
      },
    };

    api.getActiveLive.mockReturnValue(of(LIVE_RUNNING));

    api.getLiveEvents.mockReturnValue(
      of({
        run_id: 'live-001',
        events: [livePrediction],
        last_sequence: 7,
        state: 'running' as const,
      }),
    );

    api.openLiveStream.mockClear();

    component.recoverActiveLive();

    expect(component.workspace()).toBe('live');
    expect(component.live()).toEqual(LIVE_RUNNING);
    expect(component.liveInterface).toBe('eth0');
    expect(component.livePredictions()).toEqual([livePrediction]);

    expect(api.getLiveEvents).toHaveBeenCalledWith('live-001');

    expect(api.openLiveStream).toHaveBeenCalledWith('live-001', expect.any(Object), 7);
  });

  it('does not reconnect when no active live session exists', () => {
    api.getActiveLive.mockReturnValue(of(null));

    api.openLiveStream.mockClear();

    component.recoverActiveLive();

    expect(api.openLiveStream).not.toHaveBeenCalled();
  });

  it('surfaces active live recovery failures', () => {
    api.getActiveLive.mockReturnValue(
      throwError(() => ({
        error: {
          detail: 'active session lookup failed',
        },
      })),
    );

    component.recoverActiveLive();

    expect(component.error()).toBe('active session lookup failed');
  });

  it('surfaces retained live event recovery failures', () => {
    api.getActiveLive.mockReturnValue(of(LIVE_RUNNING));

    api.getLiveEvents.mockReturnValue(
      throwError(() => ({
        error: {
          detail: 'live events unavailable',
        },
      })),
    );

    component.recoverActiveLive();

    expect(component.error()).toBe('live events unavailable');
  });

  it('stops startup polling once live reaches running', () => {
    api.getLive.mockReturnValue(of(LIVE_RUNNING));

    component.startLive();

    vi.advanceTimersByTime(500);

    expect(api.getLive).toHaveBeenCalledWith('live-001');

    const calls = api.getLive.mock.calls.length;

    vi.advanceTimersByTime(2000);

    expect(api.getLive.mock.calls.length).toBe(calls);
  });
});
