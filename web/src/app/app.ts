import { DecimalPipe } from '@angular/common';
import { Component, OnDestroy, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { forkJoin } from 'rxjs';

import { OperatorApi } from './operator-api';
import {
  HealthResponse,
  LiveHistoryRecord,
  LiveInterface,
  LiveSessionSnapshot,
  ReplayHistoryRecord,
  ReplaySnapshot,
  RuntimePredictionEvent,
} from './operator.types';

type Workspace = 'live' | 'replay' | 'history';

type HistoryKind = 'live' | 'replay';

@Component({
  selector: 'app-root',
  imports: [DecimalPipe, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnDestroy {
  private readonly api = inject(OperatorApi);

  readonly health = signal<HealthResponse | null>(null);

  readonly replay = signal<ReplaySnapshot | null>(null);
  readonly live = signal<LiveSessionSnapshot | null>(null);

  readonly replayPredictions = signal<RuntimePredictionEvent[]>([]);
  readonly livePredictions = signal<RuntimePredictionEvent[]>([]);
  readonly historyPredictions = signal<RuntimePredictionEvent[]>([]);
  readonly selectedPrediction = signal<RuntimePredictionEvent | null>(null);

  readonly history = signal<ReplayHistoryRecord[]>([]);
  readonly liveHistory = signal<LiveHistoryRecord[]>([]);
  readonly liveInterfaces = signal<LiveInterface[]>([]);

  readonly workspace = signal<Workspace>('live');

  readonly error = signal<string | null>(null);
  readonly streamConnected = signal(false);
  readonly busy = signal(false);
  readonly liveBusy = signal(false);
  readonly historyBusy = signal(false);

  readonly selectedHistoryRunId = signal<string | null>(null);
  readonly selectedHistoryKind = signal<HistoryKind | null>(null);

  capture = 'nonvpn_ssh_capture4.pcap';
  timeScale = 1;
  maximumSpeed = false;
  liveInterface = '';

  private eventSource: EventSource | null = null;
  private pollTimer: ReturnType<typeof setInterval> | null = null;

  readonly active = computed(() => {
    if (this.selectedHistoryRunId() !== null) {
      return false;
    }

    const state = this.replay()?.state;

    return state === 'created' || state === 'running' || state === 'paused';
  });

  readonly liveActive = computed(() => {
    const state = this.live()?.state;

    return state === 'starting' || state === 'running' || state === 'stopping';
  });

  readonly anyActive = computed(() => this.active() || this.liveActive());

  readonly canPause = computed(
    () =>
      this.workspace() === 'replay' &&
      this.selectedHistoryRunId() === null &&
      this.replay()?.state === 'running',
  );

  readonly canResume = computed(
    () =>
      this.workspace() === 'replay' &&
      this.selectedHistoryRunId() === null &&
      this.replay()?.state === 'paused',
  );

  readonly predictions = computed(() => {
    switch (this.workspace()) {
      case 'live':
        return this.livePredictions();

      case 'history':
        return this.historyPredictions();

      case 'replay':
        return this.replayPredictions();
    }
  });

  readonly latestPrediction = computed(() => {
    const values = this.predictions();

    return values.length === 0 ? null : values[values.length - 1];
  });

  readonly historyCount = computed(() => this.history().length + this.liveHistory().length);

  readonly showingReplayContext = computed(
    () =>
      this.workspace() === 'replay' ||
      (this.workspace() === 'history' && this.selectedHistoryKind() === 'replay'),
  );

  readonly currentState = computed(() => {
    if (this.workspace() === 'live') {
      return this.live()?.state ?? 'idle';
    }

    if (this.workspace() === 'history') {
      if (this.selectedHistoryKind() === 'live') {
        return this.live()?.state ?? 'idle';
      }

      if (this.selectedHistoryKind() === 'replay') {
        return this.replay()?.state ?? 'idle';
      }

      return 'idle';
    }

    return this.replay()?.state ?? 'idle';
  });

  readonly currentRunId = computed(() => {
    if (this.workspace() === 'live') {
      return this.live()?.run_id ?? null;
    }

    if (this.workspace() === 'history') {
      if (this.selectedHistoryKind() === 'live') {
        return this.live()?.run_id ?? null;
      }

      if (this.selectedHistoryKind() === 'replay') {
        return this.replay()?.run_id ?? null;
      }

      return null;
    }

    return this.replay()?.run_id ?? null;
  });

  readonly currentSource = computed(() => {
    if (this.workspace() === 'live') {
      return this.live()?.configuration.interface ?? this.liveInterface ?? '—';
    }

    if (this.workspace() === 'history') {
      if (this.selectedHistoryKind() === 'live') {
        return this.live()?.configuration.interface ?? '—';
      }

      if (this.selectedHistoryKind() === 'replay') {
        return this.replay()?.source_id ?? '—';
      }

      return '—';
    }

    return this.replay()?.source_id ?? '—';
  });

  readonly workspaceTitle = computed(() => {
    switch (this.workspace()) {
      case 'live':
        return 'Live Sensor';

      case 'replay':
        return 'Replay Lab';

      case 'history':
        return 'Session History';
    }
  });

  readonly workspaceDescription = computed(() => {
    switch (this.workspace()) {
      case 'live':
        return 'Observe supported network metadata and runtime classifications as traffic arrives.';

      case 'replay':
        return 'Reproduce and inspect frozen PCAP workloads through the same prediction runtime.';

      case 'history':
        return 'Review persisted live and replay sessions, predictions, and model provenance.';
    }
  });

  readonly eventChannelState = computed(() => {
    if (this.streamConnected()) {
      return 'receiving';
    }

    if (this.anyActive()) {
      return 'listening';
    }

    return 'idle';
  });

  readonly modelFingerprint = computed(() => {
    const sha = this.health()?.active_model.model_bundle_sha256;

    return sha ? sha.slice(0, 12) : 'unavailable';
  });

  constructor() {
    this.loadHealth();
    this.loadHistory();
    this.loadLiveInterfaces();
    this.recoverActiveLive();
  }

  ngOnDestroy(): void {
    this.closeStream();
    this.stopPolling();
  }

  selectWorkspace(workspace: Workspace): void {
    if (this.anyActive() && workspace !== this.workspace()) {
      return;
    }

    this.workspace.set(workspace);
    this.error.set(null);
    this.selectedPrediction.set(null);

    if (workspace !== 'history') {
      this.selectedHistoryRunId.set(null);
      this.selectedHistoryKind.set(null);
    }
  }

  loadHealth(): void {
    this.api.health().subscribe({
      next: (health) => {
        this.health.set(health);
      },
      error: () => {
        this.error.set('Unable to reach the Parallax operator service.');
      },
    });
  }

  loadLiveInterfaces(): void {
    this.api.listLiveInterfaces().subscribe({
      next: (response) => {
        this.liveInterfaces.set(response.interfaces);

        if (!this.liveInterface) {
          const preferred =
            response.interfaces.find((networkInterface) => networkInterface.name !== 'lo') ??
            response.interfaces[0];

          this.liveInterface = preferred?.name ?? '';
        }
      },
      error: (response) => {
        this.error.set(response?.error?.detail ?? 'Unable to discover capture interfaces.');
      },
    });
  }

  loadHistory(): void {
    forkJoin({
      replay: this.api.listHistory(),
      live: this.api.listLiveHistory(),
    }).subscribe({
      next: ({ replay, live }) => {
        this.history.set(replay.replays);
        this.liveHistory.set(live.live_sessions);
      },
      error: () => {
        this.error.set('Unable to load session history.');
      },
    });
  }

  recoverActiveLive(): void {
    this.api.getActiveLive().subscribe({
      next: (session) => {
        if (session === null) {
          return;
        }

        this.workspace.set('live');
        this.live.set(session);
        this.liveInterface = session.configuration.interface;

        this.closeStream();
        this.stopPolling();

        this.api.getLiveEvents(session.run_id).subscribe({
          next: (snapshot) => {
            this.livePredictions.set(snapshot.events);

            this.live.update((current) => {
              if (current === null) {
                return null;
              }

              return {
                ...current,
                state: snapshot.state,
              };
            });

            this.connectLiveStream(session.run_id, snapshot.last_sequence);

            if (snapshot.state === 'starting') {
              this.startLivePolling(session.run_id);
            }
          },

          error: (response) => {
            this.error.set(response?.error?.detail ?? 'Unable to restore live sensor events.');
          },
        });
      },

      error: (response) => {
        this.error.set(response?.error?.detail ?? 'Unable to recover active live sensor.');
      },
    });
  }

  startLive(): void {
    if (!this.liveInterface || this.liveBusy() || this.active()) {
      return;
    }

    this.workspace.set('live');
    this.liveBusy.set(true);
    this.error.set(null);
    this.selectedHistoryRunId.set(null);
    this.selectedHistoryKind.set(null);
    this.livePredictions.set([]);
    this.selectedPrediction.set(null);

    this.closeStream();
    this.stopPolling();

    this.api
      .startLive({
        interface: this.liveInterface,
      })
      .subscribe({
        next: (session) => {
          this.live.set(session);
          this.liveBusy.set(false);

          this.connectLiveStream(session.run_id);
          this.startLivePolling(session.run_id);
          this.loadHealth();
        },
        error: (response) => {
          this.liveBusy.set(false);

          this.error.set(response?.error?.detail ?? 'Unable to start live sensor.');
        },
      });
  }

  stopLive(): void {
    const current = this.live();

    if (!current || !this.liveActive()) {
      return;
    }

    this.liveBusy.set(true);
    this.error.set(null);

    this.api.stopLive(current.run_id).subscribe({
      next: (response) => {
        this.live.update((session) => {
          if (session === null) {
            return null;
          }

          return {
            ...session,
            state: response.state,
          };
        });

        this.liveBusy.set(false);
      },
      error: (response) => {
        this.liveBusy.set(false);

        this.error.set(response?.error?.detail ?? 'Unable to stop live sensor.');
      },
    });
  }

  openHistory(runId: string): void {
    if (this.anyActive() || this.historyBusy()) {
      return;
    }

    this.historyBusy.set(true);
    this.error.set(null);

    this.closeStream();
    this.stopPolling();
    this.selectedPrediction.set(null);

    forkJoin({
      replay: this.api.getHistoryReplay(runId),
      events: this.api.getHistoryEvents(runId),
    }).subscribe({
      next: ({ replay, events }) => {
        this.workspace.set('history');
        this.selectedHistoryRunId.set(runId);
        this.selectedHistoryKind.set('replay');

        this.replay.set({
          ...replay,
          retained_event_count: replay.event_count,
        });

        this.historyPredictions.set(events.events);
        this.historyBusy.set(false);
      },
      error: (response) => {
        this.historyBusy.set(false);

        this.error.set(response?.error?.detail ?? 'Unable to open replay history.');
      },
    });
  }

  openLiveHistory(runId: string): void {
    if (this.anyActive() || this.historyBusy()) {
      return;
    }

    this.historyBusy.set(true);
    this.error.set(null);

    this.closeStream();
    this.stopPolling();
    this.selectedPrediction.set(null);

    forkJoin({
      live: this.api.getHistoryLive(runId),
      events: this.api.getHistoryLiveEvents(runId),
    }).subscribe({
      next: ({ live, events }) => {
        this.workspace.set('history');
        this.selectedHistoryRunId.set(runId);
        this.selectedHistoryKind.set('live');

        this.live.set(live);
        this.historyPredictions.set(events.events);
        this.historyBusy.set(false);
      },
      error: (response) => {
        this.historyBusy.set(false);

        this.error.set(response?.error?.detail ?? 'Unable to open live history.');
      },
    });
  }

  startReplay(): void {
    if (!this.capture.trim() || this.busy() || this.liveActive()) {
      return;
    }

    this.workspace.set('replay');
    this.busy.set(true);
    this.error.set(null);
    this.selectedHistoryRunId.set(null);
    this.selectedHistoryKind.set(null);
    this.replayPredictions.set([]);
    this.selectedPrediction.set(null);

    this.closeStream();
    this.stopPolling();

    this.api
      .startReplay({
        capture: this.capture.trim(),
        time_scale: this.maximumSpeed ? null : this.timeScale,
      })
      .subscribe({
        next: (snapshot) => {
          this.replay.set(snapshot);
          this.busy.set(false);

          this.connectReplayStream(snapshot.run_id);
          this.startReplayPolling(snapshot.run_id);
          this.loadHealth();
        },
        error: (response) => {
          this.busy.set(false);

          this.error.set(response?.error?.detail ?? 'Unable to start replay.');
        },
      });
  }

  pause(): void {
    const current = this.replay();

    if (!current || this.selectedHistoryRunId() !== null) {
      return;
    }

    this.api.pauseReplay(current.run_id).subscribe({
      next: () => {
        this.refreshReplay(current.run_id);
      },
      error: (response) => {
        this.error.set(response?.error?.detail ?? 'Unable to pause replay.');
      },
    });
  }

  resume(): void {
    const current = this.replay();

    if (!current || this.selectedHistoryRunId() !== null) {
      return;
    }

    this.api.resumeReplay(current.run_id).subscribe({
      next: () => {
        this.refreshReplay(current.run_id);
      },
      error: (response) => {
        this.error.set(response?.error?.detail ?? 'Unable to resume replay.');
      },
    });
  }

  cancel(): void {
    const current = this.replay();

    if (!current || this.selectedHistoryRunId() !== null) {
      return;
    }

    this.api.cancelReplay(current.run_id).subscribe({
      next: () => {
        this.refreshReplay(current.run_id);
      },
      error: (response) => {
        this.error.set(response?.error?.detail ?? 'Unable to cancel replay.');
      },
    });
  }

  trackPrediction(_index: number, event: RuntimePredictionEvent): string {
    return event.window.window_id;
  }

  selectPrediction(event: RuntimePredictionEvent): void {
    this.selectedPrediction.set(event);
  }

  closePrediction(): void {
    this.selectedPrediction.set(null);
  }

  probability(event: RuntimePredictionEvent, index: number): number {
    return event.classification.class_probabilities[index] ?? 0;
  }

  private connectReplayStream(runId: string): void {
    this.eventSource = this.api.openReplayStream(runId, {
      prediction: (event) => {
        this.streamConnected.set(true);

        this.replayPredictions.update((current) => [...current, event]);
      },

      terminal: (event) => {
        this.streamConnected.set(false);
        this.refreshReplay(event.run_id);
        this.stopPolling();
        this.loadHealth();
      },

      streamError: (detail) => {
        this.streamConnected.set(false);
        this.error.set(detail);
      },

      connectionError: () => {
        const state = this.replay()?.state;

        if (state !== 'completed' && state !== 'failed' && state !== 'cancelled') {
          this.streamConnected.set(false);
        }
      },
    });
  }

  private connectLiveStream(runId: string, afterSequence = 0): void {
    this.eventSource = this.api.openLiveStream(
      runId,
      {
        prediction: (event) => {
          this.streamConnected.set(true);

          this.livePredictions.update((current) => [...current, event]);
        },

        terminal: (event) => {
          this.streamConnected.set(false);

          this.live.update((session) => {
            if (session === null) {
              return null;
            }

            return {
              ...session,
              state: event.state,
            };
          });

          this.stopPolling();
          this.loadHealth();
          this.loadHistory();
        },

        streamError: (detail) => {
          this.streamConnected.set(false);
          this.error.set(detail);
        },

        connectionError: () => {
          const state = this.live()?.state;

          if (state !== 'completed' && state !== 'failed') {
            this.streamConnected.set(false);
          }
        },
      },
      afterSequence,
    );
  }

  private startReplayPolling(runId: string): void {
    this.pollTimer = setInterval(() => this.refreshReplay(runId), 250);
  }

  private startLivePolling(runId: string): void {
    this.pollTimer = setInterval(() => this.refreshLive(runId), 500);
  }

  private refreshReplay(runId: string): void {
    this.api.getReplay(runId).subscribe({
      next: (snapshot) => {
        this.replay.set(snapshot);

        if (
          snapshot.state === 'completed' ||
          snapshot.state === 'failed' ||
          snapshot.state === 'cancelled'
        ) {
          this.stopPolling();
          this.loadHistory();
        }
      },
    });
  }

  private refreshLive(runId: string): void {
    this.api.getLive(runId).subscribe({
      next: (session) => {
        this.live.set(session);

        if (
          session.state === 'running' ||
          session.state === 'completed' ||
          session.state === 'failed'
        ) {
          this.stopPolling();

          if (session.state === 'completed' || session.state === 'failed') {
            this.loadHistory();
          }
        }
      },
    });
  }

  private closeStream(): void {
    this.eventSource?.close();
    this.eventSource = null;
    this.streamConnected.set(false);
  }

  private stopPolling(): void {
    if (this.pollTimer !== null) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }
}
