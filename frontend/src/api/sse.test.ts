import { SseParser } from "./sse";

test("messages split across chunks are put back together", () => {
  const parser = new SseParser();
  expect(parser.push('event: event\ndata: {"kind":')).toEqual([]);
  expect(parser.push(' "note"}\n\nevent: snapshot\ndata: {}\n\nevent: ev')).toEqual([
    { event: "event", data: '{"kind": "note"}' },
    { event: "snapshot", data: "{}" },
  ]);
  expect(parser.push("ent\ndata: 1\n\n")).toEqual([{ event: "event", data: "1" }]);
});

test("CRLF line endings and multi-line data are handled", () => {
  expect(new SseParser().push("event: x\r\ndata: a\r\ndata: b\r\n\r\n")).toEqual([{ event: "x", data: "a\nb" }]);
});
