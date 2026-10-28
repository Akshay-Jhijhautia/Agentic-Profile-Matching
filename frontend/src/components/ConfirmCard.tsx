import { useState, type FormEvent } from "react";
import { Markdown } from "./Markdown";

interface Props {
  questions: string[];
  busy: boolean;
  onConfirm: () => void;
  onChange: (text: string) => void;
}

/** A skill the first question left as nice-to-have, for the example ("make Docker a must-have"). */
export function exampleSkill(questions: string[]): string {
  return questions[0]?.match(/\*\*([^*,]+)[^*]*\*\* only nice-to-have/)?.[1] ?? "it";
}

/** Requirements the job description leaves open: confirm them, or say what to change. */
export function ConfirmCard({ questions, busy, onConfirm, onChange }: Props) {
  const [change, setChange] = useState("");
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (change.trim() && !busy) onChange(change.trim());
  };
  return (
    <section className="card confirm">
      <h3>Please check before screening</h3>
      <p>The job description leaves these requirements open:</p>
      <ul>
        {questions.map((question) => (
          <li key={question}>
            <Markdown text={question} />
          </li>
        ))}
      </ul>
      <div className="row">
        <button type="button" className="primary" disabled={busy} onClick={onConfirm}>
          Screen with these
        </button>
      </div>
      <form className="row" onSubmit={submit}>
        <label htmlFor="change" className="visually-hidden">
          What to change
        </label>
        <input
          id="change"
          placeholder={`Or say what to change, e.g. "make ${exampleSkill(questions)} a must-have"`}
          value={change}
          onChange={(event) => setChange(event.target.value)}
        />
        <button type="submit" className="secondary" disabled={busy || !change.trim()}>
          Apply
        </button>
      </form>
    </section>
  );
}
