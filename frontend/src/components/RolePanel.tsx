import { useState, type FormEvent, type KeyboardEvent } from "react";

const EXAMPLES = ["React developer with 3+ years experience", "Senior Python backend engineer with Django and PostgreSQL"];

/** Step 2: the role, as a short request or a pasted job description. */
export function RolePanel({ busy, onSubmit }: { busy: boolean; onSubmit: (text: string) => void }) {
  const [text, setText] = useState("");

  const submit = (event?: FormEvent) => {
    event?.preventDefault();
    if (text.trim() && !busy) onSubmit(text.trim());
  };
  const shortcut = (event: KeyboardEvent) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) submit();
  };

  return (
    <section className="card">
      <h2>
        <span className="step">2</span> Role
      </h2>
      <form onSubmit={submit}>
        <label htmlFor="role" className="visually-hidden">
          Role or job description
        </label>
        <textarea
          id="role"
          rows={7}
          placeholder="Describe the role, or paste the job description…"
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={shortcut}
        />
        <div className="examples">
          {EXAMPLES.map((example) => (
            <button type="button" key={example} className="chip" onClick={() => setText(example)}>
              {example}
            </button>
          ))}
        </div>
        <button type="submit" className="primary" disabled={busy || !text.trim()}>
          {busy ? "Working…" : "Screen resumes"}
        </button>
      </form>
    </section>
  );
}
