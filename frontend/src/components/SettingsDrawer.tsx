import { useEffect, useState } from "react";
import type { SettingsView } from "../api/client";
import { Drawer } from "./Drawer";

const message = (error: unknown) => (error instanceof Error ? error.message : String(error));
const ROUNDS = {
  1: "retrieval only",
  2: "retrieval and the deep review",
  3: "retrieval, the deep review and interview recommendations",
} as Record<number, string>;

/** The settings in effect and where they come from. They are changed in a .env file, not
 * here: a page in the browser never handles API keys. */
export function SettingsDrawer({ load, onClose }: { load: () => Promise<SettingsView>; onClose: () => void }) {
  const [settings, setSettings] = useState<SettingsView | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    load().then(setSettings, (e) => setError(message(e)));
  }, [load]);

  return (
    <Drawer title="Settings" label="Settings" onClose={onClose}>
      {error && (
        <div className="banner problem" role="alert">
          {error}
        </div>
      )}
      {settings && <Details settings={settings} />}
    </Drawer>
  );
}

function Details({ settings }: { settings: SettingsView }) {
  const answers = settings.stored_answers;
  return (
    <>
      <section>
        <h3>Chat model</h3>
        <dl className="facts">
          <dt>Model</dt>
          <dd>{settings.model}</dd>
          <dt>API key</dt>
          <dd>
            {settings.api_key === null ? (
              "none needed"
            ) : settings.api_key_set ? (
              <span className="ok">{settings.api_key} is set</span>
            ) : (
              <span className="bad">{settings.api_key} is not set</span>
            )}
          </dd>
          <dt>Embeddings</dt>
          <dd>
            {settings.embeddings}
            {settings.embeddings.startsWith("local:") && <span className="muted"> (computed on this computer)</span>}
          </dd>
        </dl>
      </section>

      <section>
        <h3>Screening</h3>
        <dl className="facts">
          <dt>Shortlist</dt>
          <dd>the top {settings.shortlist_size} after the first round</dd>
          <dt>Rounds</dt>
          <dd>
            {settings.screening_rounds}: {ROUNDS[settings.screening_rounds]}
          </dd>
          <dt>Interview</dt>
          <dd>recommended from a final score of {settings.interview_threshold}, with no must-have missing and no high risk</dd>
          <dt>Blind</dt>
          <dd>the model never sees names, contact details or personal fields</dd>
        </dl>
      </section>

      <section>
        <h3>Your data</h3>
        <dl className="facts">
          <dt>Data folder</dt>
          <dd>
            <code>{settings.data_home}</code> <span className="muted">indexes, uploads, sessions</span>
          </dd>
          <dt>Reports</dt>
          <dd>
            <code>{settings.reports_dir}</code>
          </dd>
          <dt>Model answers</dt>
          <dd>
            {answers
              ? `${answers.answers} stored, reused ${answers.reused} times (${answers.size_mb} MB): the same question gets the same answer`
              : "not stored (LLM_CACHE=false)"}
          </dd>
        </dl>
      </section>

      <section>
        <h3>Changing settings</h3>
        <p>
          Put them in <code>{settings.env_file}</code> (or the environment), then restart <code>resume-shortlister web</code>.
          For example:
        </p>
        <pre>
          {"LLM_PROVIDER=anthropic\nANTHROPIC_API_KEY=...\nSHORTLIST_SIZE=15"}
        </pre>
        <p className="muted">
          Every setting is described in docs/configuration.md. Version {settings.version}.
        </p>
      </section>
    </>
  );
}
