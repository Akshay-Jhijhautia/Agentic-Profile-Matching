import { api, ApiError, pageToken, TOKEN_HEADER } from "./client";

function streamOf(text: string): Response {
  const body = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();
      // Two chunks, cut in the middle of an event.
      controller.enqueue(encoder.encode(text.slice(0, 25)));
      controller.enqueue(encoder.encode(text.slice(25)));
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

beforeEach(() => {
  document.head.innerHTML = '<meta name="resume-shortlister-token" content="secret">';
});

test("the page token is read from the meta tag and sent with every call", async () => {
  expect(pageToken()).toBe("secret");
  const fetchMock = vi.fn(async () => new Response(JSON.stringify({ ready: true }), { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  await api.status();
  const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
  expect((init.headers as Record<string, string>)[TOKEN_HEADER]).toBe("secret");
});

test("a turn reports each event, then resolves with the session", async () => {
  const sse =
    'event: event\ndata: {"kind": "note", "text": "Screened 12 resumes", "node": "search_resumes"}\n\n' +
    'event: snapshot\ndata: {"session_id": "s1", "screened": 12}\n\n';
  vi.stubGlobal("fetch", vi.fn(async () => streamOf(sse)));
  const events: string[] = [];
  const snapshot = await api.send("s1", "React developer", (event) => events.push(event.text));
  expect(events).toEqual(["Screened 12 resumes"]);
  expect(snapshot).toMatchObject({ session_id: "s1", screened: 12 });
});

test("an error response becomes an ApiError with the server's explanation", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "Busy" }), { status: 409 })));
  await expect(api.send("s1", "hi", () => {})).rejects.toEqual(new ApiError("Busy", 409));
});
