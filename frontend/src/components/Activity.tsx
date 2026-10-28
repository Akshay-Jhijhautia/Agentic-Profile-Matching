import type { AgentEvent } from "../api/client";

/** What the agent is doing now, step by step; once done, folded into one line that opens. */
export function Activity({ events, busy }: { events: AgentEvent[]; busy: boolean }) {
  const steps = events.filter((event) => event.kind === "note" || event.kind === "tool_failed" || event.kind === "error");
  if (!busy && !steps.length) return null;
  const failed = steps.some((event) => event.kind === "error");
  const list = (
    <ol>
      {steps.map((event, index) => (
        <li key={index} className={event.kind}>
          {event.text}
        </li>
      ))}
      {busy && <li className="pending">…</li>}
    </ol>
  );
  if (busy) {
    return (
      <section className="card activity" aria-live="polite">
        <h3>Working…</h3>
        {list}
      </section>
    );
  }
  return (
    <details className="card activity" open={failed}>
      <summary>
        {failed ? "Stopped" : "Done"} · {steps.length} step{steps.length === 1 ? "" : "s"}
      </summary>
      {list}
    </details>
  );
}
