import { render, screen } from "@testing-library/react";
import type { Snapshot } from "../api/client";
import { Results } from "./Results";

const snapshot: Snapshot = {
  session_id: "s1",
  messages: [],
  resumes: "applicants",
  requirements: { role_title: "React Developer", must_have: ["React (3+ yrs)"], nice_to_have: [], min_total_years: null, min_degree: null },
  open_questions: [],
  screened: 100,
  passing: 10,
  shortlist: [
    { rank: 1, id: "anika", name: "Anika Laurent", score: 90, retrieval: 75, deep: 100, must_haves: "✓ all met", passes_must_haves: true, decision: "interview", confidence: "high", borderline: false },
    { rank: 2, id: "rafael", name: "Rafael Bauer", score: 72, retrieval: 71, deep: 70, must_haves: "⚠ 1 unverified", passes_must_haves: true, decision: "not_now", confidence: "low", borderline: true },
  ],
  report: "# Shortlist",
  waiting: true,
};

test("the ranked table shows scores, must-haves and recommendations", () => {
  render(<Results snapshot={snapshot} onDownload={() => {}} />);
  expect(screen.getByText("React Developer")).toBeTruthy();
  expect(screen.getByText("100 resumes screened")).toBeTruthy();
  expect(screen.getByText("1 recommended for interview")).toBeTruthy();
  expect(screen.getByText("✓ Interview")).toBeTruthy();
  expect(screen.getByText("Not now")).toBeTruthy();
  expect(screen.getByText("borderline")).toBeTruthy();
  expect(screen.getByText("⚠ 1 unverified").className).toBe("warn");
});

test("nothing is shown before a screening", () => {
  const { container } = render(<Results snapshot={{ ...snapshot, requirements: null, screened: 0 }} onDownload={() => {}} />);
  expect(container.innerHTML).toBe("");
});
