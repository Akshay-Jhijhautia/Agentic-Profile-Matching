import { fireEvent, render, screen } from "@testing-library/react";
import type { Snapshot } from "../api/client";
import { Conversation, suggestions } from "./Conversation";

const messages: Snapshot["messages"] = [
  { role: "user", text: "React developer with 3+ years" },
  { role: "assistant", text: "Screened 100 resumes: **3** meet every must-have." },
];
const props = { messages, activity: [], busy: false, pending: null, suggestions: [], onSend: () => {} };

test("the agent's replies render as Markdown, after the user's messages", () => {
  const { container } = render(<Conversation {...props} />);
  const bubbles = [...container.querySelectorAll(".bubble")];
  expect(bubbles.map((b) => b.className)).toEqual(["bubble user", "bubble assistant"]);
  expect(screen.getByText("3").tagName).toBe("STRONG");
});

test("while the agent works, the message and its steps show after the history", () => {
  const { container } = render(
    <Conversation {...props} busy pending="Compare the top 3" activity={[{ kind: "note", text: "Comparing…", node: "" }]} />,
  );
  const order = [...container.querySelectorAll(".bubble, .activity")].map((e) => e.textContent);
  expect(order.slice(-2)).toEqual(["Compare the top 3", expect.stringContaining("Comparing…")]);
  expect((screen.getByRole("button", { name: "Send" }) as HTMLButtonElement).disabled).toBe(true);
});

test("Enter sends, Shift+Enter does not, and a suggestion fills the box", () => {
  const onSend = vi.fn();
  render(<Conversation {...props} suggestions={["Make Docker a must-have"]} onSend={onSend} />);
  const box = screen.getByLabelText("Message") as HTMLTextAreaElement;
  fireEvent.click(screen.getByRole("button", { name: "Make Docker a must-have" }));
  expect(box.value).toBe("Make Docker a must-have");
  fireEvent.keyDown(box, { key: "Enter", shiftKey: true });
  expect(onSend).not.toHaveBeenCalled();
  fireEvent.keyDown(box, { key: "Enter" });
  expect(onSend).toHaveBeenCalledWith("Make Docker a must-have");
  expect(box.value).toBe("");
});

test("the confirmation card takes the place of the agent's question", () => {
  render(<Conversation {...props} lastReply={<p>Please check before screening</p>} />);
  expect(screen.getByText("Please check before screening")).toBeTruthy();
  expect(screen.queryByText("3")).toBeNull();
});

const snapshot = {
  open_questions: [],
  requirements: { role_title: "React Developer", must_have: ["React"], nice_to_have: ["Docker"], min_total_years: null, min_degree: null },
  shortlist: [{ name: "Anika Laurent" }, { name: "Rafael Bauer" }],
} as unknown as Snapshot;

test("suggestions use the shortlist's names and the role's skills", () => {
  expect(suggestions(snapshot)).toEqual([
    "Compare the top 2 side by side",
    "Why is Anika Laurent ranked above Rafael Bauer?",
    "Make Docker a must-have",
  ]);
  expect(suggestions({ ...snapshot, open_questions: ["Is Docker required?"] })).toEqual([]);
});

test("a long message (a pasted job description) shows its start until opened", () => {
  const jd = "Frontend Engineer\n" + "You will build accessible, fast web applications. ".repeat(10);
  const { container } = render(<Conversation {...props} messages={[{ role: "user", text: jd }]} />);
  expect(container.querySelector(".clamped")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Show all" }));
  expect(container.querySelector(".clamped")).toBeNull();
  expect(screen.getByRole("button", { name: "Show less" })).toBeTruthy();
});
