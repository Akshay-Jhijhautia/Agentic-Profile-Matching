import { render, screen } from "@testing-library/react";
import type { AgentEvent } from "../api/client";
import { Activity } from "./Activity";

const note = (text: string): AgentEvent => ({ kind: "note", text, node: "" });

test("steps show while the agent works, and fold into one line when it is done", () => {
  const events = [note("Starting a new search…"), note("Screened 100 resumes; 10 pass every must-have.")];
  const { rerender, container } = render(<Activity events={events} busy />);
  expect(screen.getByText("Working…")).toBeTruthy();
  expect(container.querySelector("details")).toBeNull();

  rerender(<Activity events={events} busy={false} />);
  expect(screen.getByText("Done · 2 steps")).toBeTruthy();
  expect(container.querySelector("details")?.open).toBe(false);
});

test("a failed turn stays open so the error is seen", () => {
  const { container } = render(<Activity events={[note("Starting…"), { kind: "error", text: "Busy", node: "" }]} busy={false} />);
  expect(screen.getByText("Stopped · 2 steps")).toBeTruthy();
  expect(container.querySelector("details")?.open).toBe(true);
});
