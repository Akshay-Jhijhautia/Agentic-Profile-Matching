import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { SessionSummary } from "../api/client";
import { SessionsDrawer } from "./SessionsDrawer";

const saved: SessionSummary[] = [
  { id: "s2", updated: "2026-10-28T19:27:00+05:30", role: "Frontend Engineer", resumes: "/Users/me/hiring/2026/frontend/applicants", messages: 8 },
  { id: "s1", updated: "2026-10-27T10:00:00+05:30", role: null, resumes: null, messages: 2 },
];

const open = (overrides = {}) => {
  const props = {
    current: "s2",
    load: vi.fn(() => Promise.resolve(saved)),
    onOpen: vi.fn(),
    onNew: vi.fn(),
    onDelete: vi.fn(() => Promise.resolve()),
    onClose: vi.fn(),
    ...overrides,
  };
  render(<SessionsDrawer {...props} />);
  return props;
};

test("saved sessions are listed with their role and resumes, and open on a click", async () => {
  const props = open();
  fireEvent.click(await screen.findByText("No role yet"));
  expect(props.onOpen).toHaveBeenCalledWith("s1");
  expect(screen.getByText("open")).toBeTruthy(); // the session this page has open
  expect(screen.getByText(/…\/frontend\/applicants · 8 messages/)).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "+ New screening" }));
  expect(props.onNew).toHaveBeenCalled();
});

test("deleting asks first, then refreshes the list", async () => {
  const props = open();
  fireEvent.click(await screen.findByRole("button", { name: "Delete Frontend Engineer" }));
  expect(props.onDelete).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Delete" }));
  await waitFor(() => expect(props.load).toHaveBeenCalledTimes(2));
  expect(props.onDelete).toHaveBeenCalledWith("s2");
});

test("an empty list says when sessions appear", async () => {
  open({ load: () => Promise.resolve([]) });
  expect(await screen.findByText(/No saved sessions yet/)).toBeTruthy();
});
