// Server-Sent Events from a fetch() body. EventSource only does GET; turns are POSTs.

export interface SseMessage {
  event: string;
  data: string;
}

/** Feeds text chunks in, returns the complete messages so far; a message may span chunks. */
export class SseParser {
  private buffer = "";

  push(chunk: string): SseMessage[] {
    this.buffer += chunk.replace(/\r\n/g, "\n");
    const blocks = this.buffer.split("\n\n");
    this.buffer = blocks.pop() ?? "";
    return blocks.filter((block) => block.trim()).map(parseBlock);
  }
}

function parseBlock(block: string): SseMessage {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  return { event, data: data.join("\n") };
}
