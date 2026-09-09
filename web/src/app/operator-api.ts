import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import {
  ControlResponse,
  HealthResponse,
  ReplaySnapshot,
  ReplayTerminalEvent,
  RuntimePredictionEvent,
  StartReplayRequest,
} from './operator.types';

export interface ReplayStreamHandlers {
  prediction: (event: RuntimePredictionEvent) => void;
  terminal: (event: ReplayTerminalEvent) => void;
  streamError: (detail: string) => void;
  connectionError: () => void;
}

@Injectable({
  providedIn: 'root',
})
export class OperatorApi {
  private readonly http = inject(HttpClient);

  health(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>('/health');
  }

  startReplay(request: StartReplayRequest): Observable<ReplaySnapshot> {
    return this.http.post<ReplaySnapshot>('/api/v1/replays', request);
  }

  getReplay(runId: string): Observable<ReplaySnapshot> {
    return this.http.get<ReplaySnapshot>(
      `/api/v1/replays/${encodeURIComponent(runId)}`,
    );
  }

  pauseReplay(runId: string): Observable<ControlResponse> {
    return this.control(runId, 'pause');
  }

  resumeReplay(runId: string): Observable<ControlResponse> {
    return this.control(runId, 'resume');
  }

  cancelReplay(runId: string): Observable<ControlResponse> {
    return this.control(runId, 'cancel');
  }

  openReplayStream(
    runId: string,
    handlers: ReplayStreamHandlers,
  ): EventSource {
    const source = new EventSource(
      `/api/v1/replays/${encodeURIComponent(runId)}/stream`,
    );

    source.addEventListener('prediction', (event) => {
      const message = event as MessageEvent<string>;

      handlers.prediction(
        JSON.parse(message.data) as RuntimePredictionEvent,
      );
    });

    source.addEventListener('replay-terminal', (event) => {
      const message = event as MessageEvent<string>;

      handlers.terminal(
        JSON.parse(message.data) as ReplayTerminalEvent,
      );

      source.close();
    });

    source.addEventListener('stream-error', (event) => {
      const message = event as MessageEvent<string>;
      const payload = JSON.parse(message.data) as {
        detail: string;
      };

      handlers.streamError(payload.detail);
      source.close();
    });

    source.onerror = () => {
      handlers.connectionError();
    };

    return source;
  }

  private control(
    runId: string,
    action: 'pause' | 'resume' | 'cancel',
  ): Observable<ControlResponse> {
    return this.http.post<ControlResponse>(
      `/api/v1/replays/${encodeURIComponent(runId)}/${action}`,
      {},
    );
  }
}
