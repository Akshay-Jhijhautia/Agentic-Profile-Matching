import { useEffect, useState } from "react";
import type { CandidateDetail, InterviewPlan, InterviewQuestion } from "../api/client";
import { Drawer } from "./Drawer";
import { Decision } from "./Results";

const VERDICT_TONE = { strong: "ok", partial: "warn", missing: "bad" } as const;
const CHECK_TONE = { met: "ok", unverified: "warn", not_met: "bad" } as const;
const PURPOSE = {
  close_gap: "Closes a gap",
  verify_claim: "Verifies a claim",
  assess_depth: "Probes depth",
} satisfies Record<InterviewQuestion["purpose"], string>;

const message = (error: unknown) => (error instanceof Error ? error.message : String(error));

interface Props {
  load: () => Promise<CandidateDetail>;
  prepareQuestions: () => Promise<InterviewPlan>;
  onDownload: (detail: CandidateDetail) => void;
  onClose: () => void;
}

/** One candidate, opened from the shortlist: why they rank where they do, round by round, and
 * questions for their interview. Everything shown comes from the screening's own record. */
export function CandidateDrawer({ load, prepareQuestions, onDownload, onClose }: Props) {
  const [detail, setDetail] = useState<CandidateDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    load().then(setDetail, (e) => setError(message(e)));
  }, [load]);

  return (
    <Drawer
      title={detail?.name ?? "Loading…"}
      label={detail?.name ?? "Candidate"}
      meta={detail && <Meta detail={detail} />}
      footer={
        detail && (
          <button type="button" className="secondary" onClick={() => onDownload(detail)}>
            Download this report
          </button>
        )
      }
      onClose={onClose}
    >
      {error && (
        <div className="banner problem" role="alert">
          {error}
        </div>
      )}
      {detail && <Detail detail={detail} prepareQuestions={prepareQuestions} />}
    </Drawer>
  );
}

function Meta({ detail }: { detail: CandidateDetail }) {
  const facts = [
    detail.resume,
    detail.experience_years != null ? `${detail.experience_years} years experience` : null,
    detail.highest_degree ? `${detail.highest_degree} degree` : null,
  ].filter(Boolean);
  return (
    <div className="meta">
      <span className="score-badge" title="Final score">
        {detail.score}
      </span>
      <span>Rank {detail.rank}</span>
      <Decision
        decision={detail.recommendation?.decision ?? null}
        confidence={detail.recommendation?.confidence ?? null}
        borderline={detail.borderline}
      />
      <span className="muted">{facts.join(" · ")}</span>
    </div>
  );
}

function Detail({ detail, prepareQuestions }: { detail: CandidateDetail; prepareQuestions: Props["prepareQuestions"] }) {
  const { recommendation } = detail;
  return (
    <>
      {detail.knocked_out.length > 0 && (
        <p className="banner problem">Knocked out: fails {detail.knocked_out.join(", ")}.</p>
      )}
      {recommendation && (
        <section>
          <h3>Recommendation</h3>
          <p>{recommendation.rationale}</p>
          <p className="muted">Rule: {recommendation.rule}.</p>
        </section>
      )}
      {detail.summary && (
        <section>
          <h3>Deep review</h3>
          <p>{detail.summary}</p>
        </section>
      )}

      <section>
        <h3>How the score adds up</h3>
        <table className="score-table">
          <tbody>
            {detail.score_components.map((line) => (
              <tr key={line.component}>
                <td>{line.component}</td>
                <td className="muted">{line.value}</td>
                <td className="bar-cell">
                  <span className="bar" style={{ width: `${line.points}%` }} />
                </td>
                <td className="points">+{line.points.toFixed(1)}</td>
              </tr>
            ))}
            <tr className="total">
              <td>Final score</td>
              <td />
              <td />
              <td className="points">{detail.score}</td>
            </tr>
          </tbody>
        </table>
        {detail.rubric && <p className="hint">{detail.rubric}</p>}
      </section>

      {detail.evidence.length > 0 && (
        <section>
          <h3>Evidence per requirement</h3>
          <ul className="evidence">
            {detail.evidence.map((item) => (
              <li key={`${item.kind}:${item.requirement}`}>
                <div className="evidence-head">
                  <b>{item.requirement}</b>
                  <span className="tag neutral">{item.kind === "must" ? "must-have" : "nice-to-have"}</span>
                  <span className={VERDICT_TONE[item.verdict]}>{item.verdict}</span>
                </div>
                {item.evidence ? <blockquote>{item.evidence}</blockquote> : <p className="muted">Not shown on the resume.</p>}
                {item.note && <p className="hint">Guardrail: {item.note}</p>}
              </li>
            ))}
          </ul>
        </section>
      )}

      {detail.must_have_checks.length > 0 && (
        <section>
          <h3>Must-have checks</h3>
          <ul className="checks">
            {detail.must_have_checks.map((check) => (
              <li key={check.requirement}>
                <span className={CHECK_TONE[check.status]}>{check.status.replace("_", " ")}</span> {check.detail}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="two-columns">
        <List title="Strengths" items={detail.strengths} />
        <List title="Gaps" items={detail.gaps} />
      </section>

      {detail.risk_flags.length > 0 && (
        <section>
          <h3>Risk flags</h3>
          <ul>
            {detail.risk_flags.map((risk) => (
              <li key={risk.flag}>
                {risk.flag} <span className={`tag ${risk.severity === "high" ? "bad" : ""}`}>{risk.severity}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {detail.improvements.length > 0 && (
        <section>
          <h3>What would make this a clear interview</h3>
          <ul>
            {detail.improvements.map((item) => (
              <li key={item.gap}>
                {item.suggestion} <span className="muted">Verify: {item.how_to_verify}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <Questions key={detail.id} prepare={prepareQuestions} />

      <details className="retrieval">
        <summary>Round 1: retrieval score {detail.retrieval}</summary>
        <p>{detail.retrieval_reasoning}</p>
        {detail.excerpts.map((excerpt) => (
          <blockquote key={excerpt}>{excerpt}</blockquote>
        ))}
      </details>
    </>
  );
}

function List({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <h3>{title}</h3>
      {items.length ? (
        <ul>
          {items.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : (
        <p className="muted">None found.</p>
      )}
    </div>
  );
}

/** Interview questions, written on request (a model call). */
function Questions({ prepare }: { prepare: Props["prepareQuestions"] }) {
  const [plan, setPlan] = useState<InterviewPlan | null>(null);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = () => {
    setWorking(true);
    setError(null);
    prepare()
      .then(setPlan, (e) => setError(message(e)))
      .finally(() => setWorking(false));
  };

  return (
    <section className="questions">
      <h3>Interview questions</h3>
      {!plan && (
        <>
          <p className="muted">Questions aimed at this candidate's gaps, unverified claims and strengths.</p>
          <button type="button" className="secondary" disabled={working} onClick={run}>
            {working ? "Writing questions…" : "Prepare interview questions"}
          </button>
        </>
      )}
      {error && (
        <p className="bad" role="alert">
          {error}
        </p>
      )}
      {plan && (
        <ol>
          {plan.questions.map((question) => (
            <li key={question.question}>
              <p>{question.question}</p>
              <p className="hint">
                <span className="tag neutral">{question.focus}</span> {PURPOSE[question.purpose]}
              </p>
              <details>
                <summary>What a strong answer covers</summary>
                <p>{question.strong_answer}</p>
              </details>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
