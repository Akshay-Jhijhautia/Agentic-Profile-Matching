import type { Status } from "../api/client";

interface Props {
  status: Status | null;
  busy: boolean; // a turn is running: the session cannot change
  onNew: () => void;
  onSessions: () => void;
  onSettings: () => void;
}

export function Header({ status, busy, onNew, onSessions, onSettings }: Props) {
  return (
    <header className="topbar">
      <div className="brand">
        Resume <span>Shortlister</span>
      </div>
      <div className="tagline">Explainable resume screening · it ranks and explains, the recruiter decides</div>
      <nav className="actions" aria-label="Sessions and settings">
        <button type="button" onClick={onNew} disabled={busy}>
          New screening
        </button>
        <button type="button" onClick={onSessions} disabled={busy}>
          Sessions
        </button>
        <button
          type="button"
          className="status"
          onClick={onSettings}
          title={status?.problem ?? "Settings"}
          aria-label={status ? `Settings: ${status.model}, ${status.ready ? "ready" : "not ready"}` : "Settings"}
        >
          {status && <span className={`dot ${status.ready ? "ready" : "problem"}`} />}
          {status ? status.model : "Settings"} <span aria-hidden="true">⚙</span>
        </button>
      </nav>
    </header>
  );
}
