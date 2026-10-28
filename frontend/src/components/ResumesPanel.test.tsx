import { shortPath } from "./ResumesPanel";

test("long folder paths keep their last two folders", () => {
  expect(shortPath("~/Downloads/applicants")).toBe("~/Downloads/applicants");
  expect(shortPath("/Users/someone/Documents/Hiring/2026/backend-applicants")).toBe("…/2026/backend-applicants");
  expect(shortPath("your uploaded resumes")).toBe("your uploaded resumes");
});
