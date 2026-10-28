import { Activity } from "./components/Activity";
import { ConfirmCard } from "./components/ConfirmCard";
import { Header } from "./components/Header";
import { Markdown } from "./components/Markdown";
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
  const { status, snapshot, activity, busy, error } = session;
  const lastAgentMessage = [...(snapshot?.messages ?? [])].reverse().find((m) => m.role === "assistant");
  const openQuestions = snapshot?.open_questions ?? [];

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
          <Activity events={activity} busy={busy} />
          {openQuestions.length > 0 && !busy ? (
            <ConfirmCard
              questions={openQuestions}
              busy={busy}
              onConfirm={() => session.send("yes")}
              onChange={session.send}
            />
          ) : (
            lastAgentMessage &&
            !busy && (
              <section className="card agent">
                <Markdown text={lastAgentMessage.text} />
              </section>
            )
          )}
          {snapshot && <Results snapshot={snapshot} onDownload={session.downloadReport} />}
          {!snapshot?.screened && !busy && !openQuestions.length && <HowItWorks />}
        </div>
      </main>
      <footer>
        Resume Shortlister {status?.version} · runs on this computer · recommendations are a starting point; people make the
        hiring decisions
      </footer>
    </>
  );
}
