import type { CandidateRow, Snapshot } from "../api/client";

function Recommendation({ row }: { row: CandidateRow }) {
  if (!row.decision) return <span className="muted">–</span>;
  return (
    <span className="recommendation">
      <span className={`pill ${row.decision === "interview" ? "yes" : "no"}`}>
        {row.decision === "interview" ? "✓ Interview" : "Not now"}
      </span>
      <span className="muted">{row.confidence}</span>
      {row.borderline && <span className="tag">borderline</span>}
    </span>
  );
}

function MustHaves({ text }: { text: string }) {
  const tone = text.startsWith("✓") ? "ok" : text.startsWith("⚠") ? "warn" : "bad";
  return <span className={tone}>{text}</span>;
}

/** The ranked shortlist with every round's scores and the recommendation. */
export function Results({ snapshot, onDownload }: { snapshot: Snapshot; onDownload: () => void }) {
  const { requirements, shortlist } = snapshot;
  if (!requirements || !snapshot.screened) return null;
  const interviews = shortlist.filter((row) => row.decision === "interview").length;
  return (
    <section className="card results">
      <div className="results-head">
        <div>
          <h2>{requirements.role_title}</h2>
          <p className="requirements">
            <b>Must have:</b> {requirements.must_have.join(", ") || "none"} · <b>Nice to have:</b>{" "}
            {requirements.nice_to_have.join(", ") || "none"}
            {requirements.min_total_years ? ` · ${requirements.min_total_years}+ years` : ""}
            {requirements.min_degree ? ` · ${requirements.min_degree} degree` : ""}
          </p>
        </div>
        <button type="button" className="secondary" onClick={onDownload} disabled={!snapshot.report}>
          Download report
        </button>
      </div>
      <div className="chips">
        <span className="stat">{snapshot.screened} resumes screened</span>
        <span className="stat">{snapshot.passing} pass every must-have</span>
        {shortlist.some((row) => row.decision) && <span className="stat">{interviews} recommended for interview</span>}
      </div>
      {shortlist.length ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Candidate</th>
                <th>Score</th>
                <th>Retrieval</th>
                <th>Deep</th>
                <th>Must-haves</th>
                <th>Recommendation</th>
              </tr>
            </thead>
            <tbody>
              {shortlist.map((row) => (
                <tr key={row.id} className={row.decision === "not_now" ? "dim" : ""}>
                  <td className="muted">{row.rank}</td>
                  <td className="name">{row.name}</td>
                  <td className="score">{row.score}</td>
                  <td className="muted">{row.retrieval}</td>
                  <td className="muted">{row.deep ?? "–"}</td>
                  <td>
                    <MustHaves text={row.must_haves} />
                  </td>
                  <td>
                    <Recommendation row={row} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted">No candidate meets every must-have. Consider relaxing one in the role.</p>
      )}
    </section>
  );
}
