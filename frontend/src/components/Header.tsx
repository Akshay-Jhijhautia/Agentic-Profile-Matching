import type { Status } from "../api/client";

export function Header({ status }: { status: Status | null }) {
  return (
    <header className="topbar">
      <div className="brand">
        Profile <span>Match</span>
      </div>
      <div className="tagline">Explainable resume screening · it ranks and explains, the recruiter decides</div>
      {status && (
        <div className="status" title={status.problem ?? "Ready"}>
          <span className={`dot ${status.ready ? "ready" : "problem"}`} aria-label={status.ready ? "Ready" : "Not ready"} />
          {status.model} · embeddings: {status.embedder}
        </div>
      )}
    </header>
  );
}
