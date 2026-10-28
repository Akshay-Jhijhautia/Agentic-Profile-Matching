import { render, screen } from "@testing-library/react";
import type { SettingsView } from "../api/client";
import { SettingsDrawer } from "./SettingsDrawer";

const settings: SettingsView = {
  version: "1.2.0",
  model: "OpenAI gpt-5.6-terra",
  api_key: "OPENAI_API_KEY",
  api_key_set: false,
  embeddings: "local:ibm-granite/granite-embedding-small-english-r2",
  shortlist_size: 10,
  screening_rounds: 3,
  interview_threshold: 75,
  data_home: "~/.resume-shortlister",
  reports_dir: "~/.resume-shortlister/reports",
  env_file: "~/.resume-shortlister/.env",
  stored_answers: { answers: 120, reused: 340, size_mb: 1.4 },
};

test("settings in effect, a missing key, and where to change them", async () => {
  render(<SettingsDrawer load={() => Promise.resolve(settings)} onClose={() => {}} />);
  expect((await screen.findByText("OPENAI_API_KEY is not set")).className).toBe("bad");
  expect(screen.getByText("(computed on this computer)", { exact: false })).toBeTruthy();
  expect(screen.getByText(/120 stored, reused 340 times/)).toBeTruthy();
  expect(screen.getByText("~/.resume-shortlister/.env")).toBeTruthy();
});

test("a provider needing no key, and answers not stored", async () => {
  render(
    <SettingsDrawer load={() => Promise.resolve({ ...settings, api_key: null, stored_answers: null })} onClose={() => {}} />,
  );
  expect(await screen.findByText("none needed")).toBeTruthy();
  expect(screen.getByText("not stored (LLM_CACHE=false)")).toBeTruthy();
});
