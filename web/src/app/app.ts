import {
  Component,
  OnDestroy,
  computed,
  inject,
  signal,
} from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { OperatorApi } from './operator-api';
import {
  HealthResponse,
  ReplaySnapshot,
  RuntimePredictionEvent,
} from './operator.types';

@Component({
  selector: 'app-root',
  imports: [
    DecimalPipe,
    FormsModule,
  ],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnDestroy {
  private readonly api = inject(OperatorApi);

  readonly health = signal<HealthResponse | null>(null);
  readonly replay = signal<ReplaySnapshot | null>(null);
  readonly predictions = signal<RuntimePredictionEvent[]>([]);

  readonly error = signal<string | null>(null);
  readonly streamConnected = signal(false);
  readonly busy = signal(false);

  capture = 'nonvpn_ssh_capture4.pcap';
  timeScale = 1;
  maximumSpeed = false;

  private eventSource: EventSource | null = null;
  private pollTimer: ReturnType<typeof setInterval> | null = null;

  readonly active = computed(() => {
    const state = this.replay()?.state;

    return (
      state === 'created' ||
      state === 'running' ||
      state === 'paused'
    );
  });

  readonly canPause = computed(
    () => this.replay()?.state === 'running',
  );

  readonly canResume = computed(
    () => this.replay()?.state === 'paused',
  );

  readonly latestPrediction = computed(() => {
    const values = this.predictions();

    return values.length === 0
      ? null
      : values[values.length - 1];
  });

  constructor() {
    this.loadHealth();
  }

  ngOnDestroy(): void {
    this.closeStream();
    this.stopPolling();
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

  startReplay(): void {
    if (!this.capture.trim() || this.busy()) {
      return;
    }

    this.busy.set(true);
    this.error.set(null);
    this.predictions.set([]);
    this.closeStream();
    this.stopPolling();

    this.api
      .startReplay({
        capture: this.capture.trim(),
        time_scale: this.maximumSpeed
          ? null
          : this.timeScale,
      })
      .subscribe({
        next: (snapshot) => {
          this.replay.set(snapshot);
          this.busy.set(false);

          this.connectStream(snapshot.run_id);
          this.startPolling(snapshot.run_id);
          this.loadHealth();
        },
        error: (response) => {
          this.busy.set(false);
          this.error.set(
            response?.error?.detail ??
              'Unable to start replay.',
          );
        },
      });
  }

  pause(): void {
    const current = this.replay();

    if (!current) {
      return;
    }

    this.api.pauseReplay(current.run_id).subscribe({
      next: () => {
        this.refreshReplay(current.run_id);
      },
      error: (response) => {
        this.error.set(
          response?.error?.detail ??
            'Unable to pause replay.',
        );
      },
    });
  }

  resume(): void {
    const current = this.replay();

    if (!current) {
      return;
    }

    this.api.resumeReplay(current.run_id).subscribe({
      next: () => {
        this.refreshReplay(current.run_id);
      },
      error: (response) => {
        this.error.set(
          response?.error?.detail ??
            'Unable to resume replay.',
        );
      },
    });
  }

  cancel(): void {
    const current = this.replay();

    if (!current) {
      return;
    }

    this.api.cancelReplay(current.run_id).subscribe({
      next: () => {
        this.refreshReplay(current.run_id);
      },
      error: (response) => {
        this.error.set(
          response?.error?.detail ??
            'Unable to cancel replay.',
        );
      },
    });
  }

  trackPrediction(
    _index: number,
    event: RuntimePredictionEvent,
  ): string {
    return event.window.window_id;
  }

  probability(
    event: RuntimePredictionEvent,
    index: number,
  ): number {
    return event.classification.class_probabilities[index] ?? 0;
  }

  private connectStream(runId: string): void {
    this.eventSource = this.api.openReplayStream(
      runId,
      {
        prediction: (event) => {
          this.streamConnected.set(true);

          this.predictions.update((current) => [
            ...current,
            event,
          ]);
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

          if (
            state !== 'completed' &&
            state !== 'failed' &&
            state !== 'cancelled'
          ) {
            this.streamConnected.set(false);
          }
        },
      },
    );
  }

  private startPolling(runId: string): void {
    this.pollTimer = setInterval(
      () => this.refreshReplay(runId),
      250,
    );
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
