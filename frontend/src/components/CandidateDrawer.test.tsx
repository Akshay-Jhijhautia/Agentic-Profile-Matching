import { fireEvent, render, screen } from "@testing-library/react";
import type { CandidateDetail, InterviewPlan } from "../api/client";
import { CandidateDrawer } from "./CandidateDrawer";

const detail: CandidateDetail = {
  id: "anika",
  name: "Anika Laurent",
  rank: 1,
  score: 90,
  retrieval: 75,
  deep: 100,
  resume: "anika_laurent.pdf",
  experience_years: 6,
  highest_degree: "Bachelor's",
  recommendation: { decision: "interview", confidence: "high", rule: "final 90 ≥ 75; no must-have missing", rationale: "Five years of React in production." },
  borderline: false,
  knocked_out: [],
  summary: "Strong React engineer.",
  score_components: [
    { component: "Semantic fit", value: "80%", weight: 0.2, points: 16 },
    { component: "Deep analysis", value: "100", weight: 0.6, points: 60 },
  ],
  rubric: "The deep-analysis score is computed from the verdicts below by a fixed rubric.",
  must_have_checks: [{ requirement: "React", status: "met", detail: "React (5 yrs)" }],
  evidence: [
    { requirement: "React", kind: "must", verdict: "strong", evidence: "Led the React migration at Initech", note: "" },
    { requirement: "GraphQL", kind: "nice", verdict: "missing", evidence: "", note: "" },
  ],
  strengths: ["Led a migration"],
  gaps: [],
  risk_flags: [{ flag: "Short last tenure", severity: "medium" }],
  improvements: [],
  retrieval_reasoning: "Strong match.",
  excerpts: ["[Skills] React, TypeScript"],
  report: "# Anika Laurent",
};

const plan: InterviewPlan = {
  candidate_id: "anika",
  focus_areas: { gaps: ["GraphQL"], unverified: [], strengths: ["React"] },
  questions: [{ question: "How did you plan the migration?", focus: "React", purpose: "assess_depth", strong_answer: "Incremental, measured." }],
};

const open = (overrides: Partial<Parameters<typeof CandidateDrawer>[0]> = {}) =>
  render(
    <CandidateDrawer
      load={() => Promise.resolve(detail)}
      prepareQuestions={() => Promise.resolve(plan)}
      onDownload={() => {}}
      onClose={() => {}}
      {...overrides}
    />,
  );

test("the detail shows the recommendation, the score and the evidence behind it", async () => {
  open();
  expect(await screen.findByRole("heading", { name: "Anika Laurent" })).toBeTruthy();
  expect(screen.getByText("Five years of React in production.")).toBeTruthy();
  expect(screen.getByText("Rule: final 90 ≥ 75; no must-have missing.")).toBeTruthy();
  expect(screen.getByText("+60.0")).toBeTruthy();
  expect(screen.getByText("Led the React migration at Initech").tagName).toBe("BLOCKQUOTE");
  expect(screen.getByText("Not shown on the resume.")).toBeTruthy();
  expect(screen.getByText("strong").className).toBe("ok");
  expect(screen.getByText("None found.")).toBeTruthy(); // no gaps
});

test("interview questions are written on request", async () => {
  const prepareQuestions = vi.fn(() => Promise.resolve(plan));
  open({ prepareQuestions });
  fireEvent.click(await screen.findByRole("button", { name: "Prepare interview questions" }));
  expect(await screen.findByText("How did you plan the migration?")).toBeTruthy();
  expect(screen.getByText("Probes depth", { exact: false })).toBeTruthy();
  expect(prepareQuestions).toHaveBeenCalledTimes(1);
});

test("a failure is shown, and the drawer closes", async () => {
  const onClose = vi.fn();
  open({ load: () => Promise.reject(new Error("No candidate matches 'x'")), onClose });
  expect((await screen.findByRole("alert")).textContent).toBe("No candidate matches 'x'");
  fireEvent.click(screen.getByRole("button", { name: "Close" }));
  expect(onClose).toHaveBeenCalled();
});
