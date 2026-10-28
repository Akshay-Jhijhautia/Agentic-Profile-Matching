import { useCallback, useEffect, useState } from "react";
import type { SessionSummary } from "../api/client";
import { Drawer } from "./Drawer";
import { shortPath } from "./ResumesPanel";

const message = (error: unknown) => (error instanceof Error ? error.message : String(error));
const when = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });

interface Props {
  current: string; // the session this page has open
  load: () => Promise<SessionSummary[]>;
  onOpen: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => Promise<void>;
  onClose: () => void;
}

/** Saved screenings, newest first: open one to carry on, or delete it with its history. */
export function SessionsDrawer({ current, load, onOpen, onNew, onDelete, onClose }: Props) {
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => load().then(setSessions, (e) => setError(message(e))), [load]);
  useEffect(() => {
    void refresh();
  }, [refresh]);

  const remove = async (id: string) => {
    setError(null);
    try {
      await onDelete(id);
      setConfirming(null);
      await refresh();
    } catch (e) {
      setError(message(e));
    }
  };

  return (
    <Drawer
      title="Sessions"
      label="Sessions"
      meta={<p className="muted">Every screening is saved as it goes, with its role, results and conversation.</p>}
      onClose={onClose}
    >
      <div>
        <button type="button" className="secondary" onClick={onNew}>
          + New screening
        </button>
      </div>
      {error && (
        <div className="banner problem" role="alert">
          {error}
        </div>
      )}
      {sessions === null && !error && <p className="muted">Loading…</p>}
      {sessions?.length === 0 && <p className="muted">No saved sessions yet: they appear once you send a message.</p>}
      {sessions && sessions.length > 0 && (
        <ul className="sessions">
          {sessions.map((session) => (
            <li key={session.id} className={session.id === current ? "current" : undefined}>
              <button type="button" className="session-open" onClick={() => onOpen(session.id)}>
                <b>{session.role ?? "No role yet"}</b>
                <span className="muted">
                  {[session.resumes && shortPath(session.resumes), `${session.messages} messages`, when.format(new Date(session.updated))]
                    .filter(Boolean)
                    .join(" · ")}
                </span>
              </button>
              {session.id === current && <span className="tag neutral">open</span>}
              {confirming === session.id ? (
                <span className="confirm-delete">
                  <button type="button" className="danger" onClick={() => remove(session.id)}>
                    Delete
                  </button>
                  <button type="button" className="secondary" onClick={() => setConfirming(null)}>
                    Keep
                  </button>
                </span>
              ) : (
                <button
                  type="button"
                  className="secondary"
                  aria-label={`Delete ${session.role ?? session.id}`}
                  onClick={() => setConfirming(session.id)}
                >
                  Delete…
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </Drawer>
  );
}
