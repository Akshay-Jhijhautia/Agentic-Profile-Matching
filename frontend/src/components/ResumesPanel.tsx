import { useRef, useState, type DragEvent, type FormEvent } from "react";

const ACCEPTED = ".pdf,.docx,.txt,.md";

/** A long folder path as its last two folders ("…/hiring/applicants"); short ones as they are. */
export function shortPath(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return path.length <= 40 || parts.length <= 2 ? path : `…/${parts.slice(-2).join("/")}`;
}

interface Props {
  resumes: string | null;
  busy: boolean;
  onFiles: (files: File[]) => void;
  onFolder: (path: string) => void;
}

/** Step 1: resume files dropped or chosen, or a folder on this computer. */
export function ResumesPanel({ resumes, busy, onFiles, onFolder }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [folder, setFolder] = useState("");

  const drop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    const files = Array.from(event.dataTransfer.files);
    if (files.length && !busy) onFiles(files);
  };
  const submitFolder = (event: FormEvent) => {
    event.preventDefault();
    if (folder.trim() && !busy) onFolder(folder.trim());
  };

  return (
    <section className="card">
      <h2>
        <span className="step">1</span> Resumes
      </h2>
      {resumes && (
        <p className="loaded">
          Screening <b title={resumes}>{shortPath(resumes)}</b>
        </p>
      )}
      <div
        className={`dropzone ${dragging ? "dragging" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={drop}
      >
        <p>Drop resume files here</p>
        <p className="hint">PDF, DOCX, TXT or MD</p>
        <button type="button" className="secondary" disabled={busy} onClick={() => input.current?.click()}>
          Choose files
        </button>
        <input
          ref={input}
          type="file"
          multiple
          accept={ACCEPTED}
          hidden
          aria-label="Resume files"
          onChange={(event) => {
            const files = Array.from(event.target.files ?? []);
            event.target.value = "";
            if (files.length) onFiles(files);
          }}
        />
      </div>
      <form className="folder" onSubmit={submitFolder}>
        <label htmlFor="folder">Or a folder on this computer</label>
        <div className="row">
          <input
            id="folder"
            placeholder="~/Downloads/applicants"
            value={folder}
            onChange={(event) => setFolder(event.target.value)}
          />
          <button type="submit" className="secondary" disabled={busy || !folder.trim()}>
            Load
          </button>
        </div>
      </form>
    </section>
  );
}
