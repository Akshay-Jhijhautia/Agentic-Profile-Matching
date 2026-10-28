import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** The agent's Markdown (answers, reports), with tables; links open in a new tab. */
export function Markdown({ text }: { text: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{ a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noreferrer" /> }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
