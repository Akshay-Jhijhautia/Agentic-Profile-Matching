// The web API, typed from its OpenAPI schema (src/api/schema.ts, generated).
import type { components } from "./schema";
import { SseParser } from "./sse";

type Schemas = components["schemas"];
export type Snapshot = Schemas["Snapshot"];
export type Status = Schemas["Status"];
export type CandidateRow = Schemas["CandidateRow"];
export type SessionSummary = Schemas["SessionSummary"];
export type CandidateReport = Schemas["CandidateReport"];

/** What the agent does during a turn (profile_matching/conversation.py: Event). */
export interface AgentEvent {
  kind: "note" | "message" | "report" | "candidate_report" | "tool_failed" | "error";
  text: string;
  node: string;
}

export const TOKEN_HEADER = "X-Profile-Match-Token";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

/** The token the server wrote into the page; every API call carries it. */
export function pageToken(doc: Document = document): string {
  return doc.querySelector<HTMLMetaElement>('meta[name="profile-match-token"]')?.content ?? "";
}

function withToken(init: RequestInit = {}): RequestInit {
  return { ...init, headers: { ...(init.headers ?? {}), [TOKEN_HEADER]: pageToken() } };
}

async function failure(response: Response): Promise<ApiError> {
  let detail = response.statusText;
  try {
    const body = await response.json();
    detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    /* not JSON: keep the status text */
  }
  return new ApiError(detail || `Request failed (${response.status})`, response.status);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, withToken(init));
  if (!response.ok) throw await failure(response);
  return (response.status === 204 ? undefined : await response.json()) as T;
}

/** Runs a turn: calls ``onEvent`` as the agent works and resolves with the session afterwards. */
async function stream(path: string, init: RequestInit, onEvent: (event: AgentEvent) => void): Promise<Snapshot> {
  const response = await fetch(path, withToken({ ...init, method: "POST" }));
  if (!response.ok || !response.body) throw await failure(response);
  const parser = new SseParser();
  const decoder = new TextDecoder();
  const reader = response.body.getReader();
  let snapshot: Snapshot | undefined;
  for (;;) {
    const { done, value } = await reader.read();
    for (const message of parser.push(done ? "\n\n" : decoder.decode(value, { stream: true }))) {
      if (message.event === "event") onEvent(JSON.parse(message.data) as AgentEvent);
      else if (message.event === "snapshot") snapshot = JSON.parse(message.data) as Snapshot;
    }
    if (done) break;
  }
  if (!snapshot) throw new ApiError("The connection ended before the agent finished.", 0);
  return snapshot;
}

const json = (body: unknown): RequestInit => ({
  body: JSON.stringify(body),
  headers: { "Content-Type": "application/json" },
});

export const api = {
  status: () => request<Status>("/api/status"),
  sessions: () => request<SessionSummary[]>("/api/sessions"),
  createSession: () => request<Snapshot>("/api/sessions", { method: "POST" }),
  session: (id: string) => request<Snapshot>(`/api/sessions/${encodeURIComponent(id)}`),
  candidate: (id: string, candidate: string) =>
    request<CandidateReport>(`/api/sessions/${encodeURIComponent(id)}/candidates/${encodeURIComponent(candidate)}`),
  send: (id: string, text: string, onEvent: (event: AgentEvent) => void) =>
    stream(`/api/sessions/${encodeURIComponent(id)}/turns`, json({ text }), onEvent),
  loadFolder: (id: string, path: string, onEvent: (event: AgentEvent) => void) =>
    stream(`/api/sessions/${encodeURIComponent(id)}/folder`, json({ path }), onEvent),
  upload: (id: string, files: File[], onEvent: (event: AgentEvent) => void) => {
    const form = new FormData();
    for (const file of files) form.append("files", file, file.name);
    return stream(`/api/sessions/${encodeURIComponent(id)}/files`, { body: form }, onEvent);
  },
  /** The shortlist report as a file download. */
  async downloadReport(id: string): Promise<void> {
    const response = await fetch(`/api/sessions/${encodeURIComponent(id)}/report`, withToken());
    if (!response.ok) throw await failure(response);
    const url = URL.createObjectURL(await response.blob());
    const link = Object.assign(document.createElement("a"), { href: url, download: `profile-match-${id}.md` });
    link.click();
    URL.revokeObjectURL(url);
  },
};
