import { useMemo, useState } from "react";
import { api, saveFile } from "./api/client";
import { CandidateDrawer } from "./components/CandidateDrawer";
import { ConfirmCard } from "./components/ConfirmCard";
import { Conversation, suggestions } from "./components/Conversation";
import { Header } from "./components/Header";
import { ResumesPanel } from "./components/ResumesPanel";
import { Results } from "./components/Results";
import { RolePanel } from "./components/RolePanel";
import { useSession } from "./useSession";

function HowItWorks() {
  return (
    <section className="card how">
      <h2>How it works</h2>
      <ol>
        <li>
          <b>Every resume is scored</b>: semantic and keyword search, and the must-have checks.
        </li>
        <li>
          <b>The shortlist is reviewed in depth</b>: a verdict per requirement, backed by a quote from the resume.
        </li>
        <li>
          <b>A fixed rule recommends</b> whom to interview; the model only explains. You decide.
        </li>
      </ol>
    </section>
  );
}

export function App() {
  const session = useSession();
  const { status, snapshot, activity, busy, pending, error } = session;
  const [selected, setSelected] = useState<string | null>(null);
  const sessionId = snapshot?.session_id ?? "";
  const openQuestions = snapshot?.open_questions ?? [];
  const messages = snapshot?.messages ?? [];

  // Made once per candidate: the drawer loads again when they change.
  const drawer = useMemo(
    () =>
      selected && {
        load: () => api.candidate(sessionId, selected),
        prepareQuestions: () => api.questions(sessionId, selected),
      },
    [sessionId, selected],
  );

  return (
    <>
      <Header status={status} />
      {status && !status.ready && <div className="banner problem">{status.problem}</div>}
      {snapshot?.notice && <div className="banner">{snapshot.notice}</div>}
      {error && (
        <div className="banner problem" role="alert">
          {error}
        </div>
      )}
      <main className="layout">
        <aside>
          <ResumesPanel resumes={snapshot?.resumes ?? null} busy={busy || !snapshot} onFiles={session.upload} onFolder={session.loadFolder} />
          <RolePanel busy={busy || !snapshot} onSubmit={session.send} />
        </aside>
        <div className="main">
          {snapshot && <Results snapshot={snapshot} onDownload={session.downloadReport} onSelect={setSelected} />}
          {messages.length > 0 || busy ? (
            <Conversation
              messages={messages}
              activity={activity}
              busy={busy}
              pending={pending}
              suggestions={snapshot ? suggestions(snapshot) : []}
              lastReply={
                openQuestions.length > 0 && (
                  <ConfirmCard questions={openQuestions} busy={busy} onConfirm={() => session.send("yes")} onChange={session.send} />
                )
              }
              onSend={session.send}
            />
          ) : (
            <HowItWorks />
          )}
        </div>
      </main>
      {drawer && (
        <CandidateDrawer
          {...drawer}
          onDownload={(detail) => saveFile(new Blob([detail.report], { type: "text/markdown" }), `${detail.id}.md`)}
          onClose={() => setSelected(null)}
        />
      )}
      <footer>
        Resume Shortlister {status?.version} · runs on this computer · recommendations are a starting point; people make the
        hiring decisions
      </footer>
    </>
  );
}
