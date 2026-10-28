import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent, type ReactNode } from "react";
import type { AgentEvent, Snapshot } from "../api/client";
import { Activity } from "./Activity";
import { Markdown } from "./Markdown";

type Message = Snapshot["messages"][number];

// A message longer than this (a pasted job description) shows its start until opened.
const LONG_MESSAGE = 280;

function UserMessage({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const long = text.length > LONG_MESSAGE;
  return (
    <div className="bubble user">
      <div className={long && !open ? "clamped" : undefined}>{text}</div>
      {long && (
        <button type="button" className="more" onClick={() => setOpen(!open)}>
          {open ? "Show less" : "Show all"}
        </button>
      )}
    </div>
  );
}

/** Follow-ups worth asking about this shortlist, with its own names and skills. */
export function suggestions(snapshot: Snapshot): string[] {
  const { shortlist, requirements } = snapshot;
  if (!requirements || snapshot.open_questions.length) return [];
  const found: string[] = [];
  if (shortlist.length >= 2) {
    found.push(`Compare the top ${Math.min(shortlist.length, 3)} side by side`);
    found.push(`Why is ${shortlist[0].name} ranked above ${shortlist[1].name}?`);
  }
  if (requirements.nice_to_have.length) found.push(`Make ${requirements.nice_to_have[0]} a must-have`);
  return found;
}

interface Props {
  messages: Message[];
  activity: AgentEvent[];
  busy: boolean;
  pending: string | null; // the message being answered, until the session comes back with it
  suggestions: string[];
  lastReply?: ReactNode; // shown instead of the agent's last message (e.g. the confirmation card)
  onSend: (text: string) => void;
}

/** The chat with the agent: questions, comparisons and changes to the role, and what it did for each. */
export function Conversation({ messages, activity, busy, pending, suggestions, lastReply, onSend }: Props) {
  const [text, setText] = useState("");
  const list = useRef<HTMLDivElement>(null);

  const shown = busy && pending ? [...messages, { role: "user" as const, text: pending }] : messages;
  // The current turn's steps follow the message that started it.
  const lastUser = shown.map((m) => m.role).lastIndexOf("user");
  const lastAgent = shown.map((m) => m.role).lastIndexOf("assistant");

  // The latest message in view, within the list only: the page stays where the reader is.
  useEffect(() => {
    if (list.current) list.current.scrollTop = list.current.scrollHeight;
  }, [shown.length, busy, activity.length]);

  const submit = (event?: FormEvent) => {
    event?.preventDefault();
    if (!text.trim() || busy) return;
    onSend(text.trim());
    setText("");
  };
  const keys = (event: KeyboardEvent) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <section className="card conversation" aria-label="Conversation">
      <h2>Conversation</h2>
      <div className="messages" ref={list}>
        {shown.map((message, index) => (
          <div key={index}>
            {index === lastAgent && lastReply && index > lastUser ? (
              lastReply
            ) : (
              message.role === "assistant" ? (
                <div className="bubble assistant">
                  <Markdown text={message.text} />
                </div>
              ) : (
                <UserMessage text={message.text} />
              )
            )}
            {index === lastUser && <Activity events={activity} busy={busy} />}
          </div>
        ))}
        {lastUser < 0 && <Activity events={activity} busy={busy} />}
      </div>
      {suggestions.length > 0 && !busy && (
        <div className="examples">
          {suggestions.map((suggestion) => (
            <button type="button" key={suggestion} className="chip" onClick={() => setText(suggestion)}>
              {suggestion}
            </button>
          ))}
        </div>
      )}
      <form className="composer" onSubmit={submit}>
        <label htmlFor="message" className="visually-hidden">
          Message
        </label>
        <textarea
          id="message"
          rows={2}
          placeholder="Ask about the candidates, or change the role (Enter sends, Shift+Enter for a new line)"
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={keys}
        />
        <button type="submit" className="primary" disabled={busy || !text.trim()}>
          Send
        </button>
      </form>
    </section>
  );
}
