import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ConfirmCard, exampleSkill } from "./ConfirmCard";

const QUESTION = '"4+ years of iOS development in Swift": **Swift** must-have, **iOS** only nice-to-have';

test("the recruiter confirms the open requirements, or says what to change", async () => {
  const onConfirm = vi.fn();
  const onChange = vi.fn();
  render(<ConfirmCard questions={[QUESTION]} busy={false} onConfirm={onConfirm} onChange={onChange} />);
  expect(screen.getByText("Swift").tagName).toBe("STRONG");

  await userEvent.click(screen.getByRole("button", { name: "Screen with these" }));
  expect(onConfirm).toHaveBeenCalledOnce();

  await userEvent.type(screen.getByLabelText("What to change"), "make iOS a must-have");
  await userEvent.click(screen.getByRole("button", { name: "Apply" }));
  expect(onChange).toHaveBeenCalledWith("make iOS a must-have");
});

test("the example names a skill from the question", () => {
  expect(exampleSkill([QUESTION])).toBe("iOS");
  expect(exampleSkill(['"Deploying models with Docker and Kubernetes": **Docker, Kubernetes** only nice-to-have'])).toBe("Docker");
  expect(exampleSkill([])).toBe("it");
});
