// The session this page works on: kept in the URL (?session=...), so a reload reopens it.
import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type AgentEvent, type Snapshot, type Status } from "./api/client";

type Start = (onEvent: (event: AgentEvent) => void) => Promise<Snapshot>;

const message = (error: unknown) => (error instanceof Error ? error.message : String(error));

export function useSession() {
  const [status, setStatus] = useState<Status | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  // What the agent did during the current (or last) turn.
  const [activity, setActivity] = useState<AgentEvent[]>([]);
  const [busy, setBusy] = useState(false);
  // The message being answered, shown until the session comes back with it.
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // React runs effects twice in development (StrictMode); open one session, not two.
  const opening = useRef(false);

  useEffect(() => {
    if (opening.current) return;
    opening.current = true;
    api.status().then(setStatus, (e) => setError(message(e)));
    const wanted = new URLSearchParams(window.location.search).get("session");
    const open = wanted
      ? api.session(wanted).catch((e) => (e instanceof ApiError && e.status === 404 ? api.createSession() : Promise.reject(e)))
      : api.createSession();
    open.then(
      (opened) => {
        setSnapshot(opened);
        window.history.replaceState(null, "", `?session=${encodeURIComponent(opened.session_id)}`);
      },
      (e) => setError(message(e)),
    );
  }, []);

  const run = useCallback(
    async (said: string, start: Start) => {
      if (busy) return;
      setBusy(true);
      setPending(said);
      setError(null);
      setActivity([]);
      try {
        setSnapshot(await start((event) => setActivity((events) => [...events, event])));
      } catch (e) {
        setError(message(e));
      } finally {
        setBusy(false);
        setPending(null);
      }
    },
    [busy],
  );

  const id = snapshot?.session_id ?? "";
  return {
    status,
    snapshot,
    activity,
    busy,
    pending,
    error,
    send: (text: string) => run(text, (onEvent) => api.send(id, text, onEvent)),
    loadFolder: (path: string) => run(`Load the resumes in ${path}`, (onEvent) => api.loadFolder(id, path, onEvent)),
    upload: (files: File[]) => run(uploadMessage(files), (onEvent) => api.upload(id, files, onEvent)),
    downloadReport: () => api.downloadReport(id).catch((e) => setError(message(e))),
  };
}

/** An upload as the chat records it (``web/models.py: describe_command``). */
export function uploadMessage(files: File[]): string {
  return `Upload ${files.length} file${files.length === 1 ? "" : "s"}: ${files.map((file) => file.name).join(", ")}`;
}
