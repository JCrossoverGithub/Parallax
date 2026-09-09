import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import {
  ControlResponse,
  HealthResponse,
  LiveControlResponse,
  LiveEventsResponse,
  LiveInterfacesResponse,
  LiveSessionSnapshot,
  LiveTerminalEvent,
  ReplayHistoryEventsResponse,
  ReplayHistoryRecord,
  ReplayHistoryResponse,
  ReplaySnapshot,
  ReplayTerminalEvent,
  RuntimePredictionEvent,
  StartLiveRequest,
  StartReplayRequest,
} from './operator.types';

export interface RuntimeStreamHandlers<TTerminal> {
  prediction: (event: RuntimePredictionEvent) => void;
  terminal: (event: TTerminal) => void;
  streamError: (detail: string) => void;
  connectionError: () => void;
}

export type ReplayStreamHandlers = RuntimeStreamHandlers<ReplayTerminalEvent>;

export type LiveStreamHandlers = RuntimeStreamHandlers<LiveTerminalEvent>;

@Injectable({
  providedIn: 'root',
})
export class OperatorApi {
  private readonly http = inject(HttpClient);

  health(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>('/health');
  }

  listLiveInterfaces(): Observable<LiveInterfacesResponse> {
    return this.http.get<LiveInterfacesResponse>('/api/v1/live/interfaces');
  }

  startLive(request: StartLiveRequest): Observable<LiveSessionSnapshot> {
    return this.http.post<LiveSessionSnapshot>('/api/v1/live', request);
  }

  getLive(runId: string): Observable<LiveSessionSnapshot> {
    return this.http.get<LiveSessionSnapshot>(`/api/v1/live/${encodeURIComponent(runId)}`);
  }

  getLiveEvents(runId: string): Observable<LiveEventsResponse> {
    return this.http.get<LiveEventsResponse>(`/api/v1/live/${encodeURIComponent(runId)}/events`);
  }

  stopLive(runId: string): Observable<LiveControlResponse> {
    return this.http.post<LiveControlResponse>(
      `/api/v1/live/${encodeURIComponent(runId)}/stop`,
      {},
    );
  }

  openLiveStream(runId: string, handlers: LiveStreamHandlers): EventSource {
    return this.openStream(
      `/api/v1/live/${encodeURIComponent(runId)}/stream`,
      'live-terminal',
      handlers,
    );
  }

  listHistory(limit = 100): Observable<ReplayHistoryResponse> {
    return this.http.get<ReplayHistoryResponse>(`/api/v1/history?limit=${limit}`);
  }

  getHistoryReplay(runId: string): Observable<ReplayHistoryRecord> {
    return this.http.get<ReplayHistoryRecord>(`/api/v1/history/${encodeURIComponent(runId)}`);
  }

  getHistoryEvents(runId: string): Observable<ReplayHistoryEventsResponse> {
    return this.http.get<ReplayHistoryEventsResponse>(
      `/api/v1/history/${encodeURIComponent(runId)}/events`,
    );
  }

  startReplay(request: StartReplayRequest): Observable<ReplaySnapshot> {
    return this.http.post<ReplaySnapshot>('/api/v1/replays', request);
  }

  getReplay(runId: string): Observable<ReplaySnapshot> {
    return this.http.get<ReplaySnapshot>(`/api/v1/replays/${encodeURIComponent(runId)}`);
  }

  pauseReplay(runId: string): Observable<ControlResponse> {
    return this.replayControl(runId, 'pause');
  }

  resumeReplay(runId: string): Observable<ControlResponse> {
    return this.replayControl(runId, 'resume');
  }

  cancelReplay(runId: string): Observable<ControlResponse> {
    return this.replayControl(runId, 'cancel');
  }

  openReplayStream(runId: string, handlers: ReplayStreamHandlers): EventSource {
    return this.openStream(
      `/api/v1/replays/${encodeURIComponent(runId)}/stream`,
      'replay-terminal',
      handlers,
    );
  }

  private openStream<TTerminal>(
    url: string,
    terminalEventName: string,
    handlers: RuntimeStreamHandlers<TTerminal>,
  ): EventSource {
    const source = new EventSource(url);

    source.addEventListener('prediction', (event) => {
      const message = event as MessageEvent<string>;

      handlers.prediction(JSON.parse(message.data) as RuntimePredictionEvent);
    });

    source.addEventListener(terminalEventName, (event) => {
      const message = event as MessageEvent<string>;

      handlers.terminal(JSON.parse(message.data) as TTerminal);

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

  private replayControl(
    runId: string,
    action: 'pause' | 'resume' | 'cancel',
  ): Observable<ControlResponse> {
    return this.http.post<ControlResponse>(
      `/api/v1/replays/${encodeURIComponent(runId)}/${action}`,
      {},
    );
  }
}
