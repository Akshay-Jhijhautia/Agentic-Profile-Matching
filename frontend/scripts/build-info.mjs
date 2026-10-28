// Records a fingerprint of the UI's source next to the build, so a Python test can tell
// when the committed build (resume_shortlister/web/static/) is older than the source.
// The same fingerprint is computed in tests/test_web_ui.py: keep the two in step.
import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const SOURCES = ["index.html", "package.json", "package-lock.json", "tsconfig.json", "vite.config.ts", "src"];

function files(path) {
  if (statSync(path).isDirectory()) return readdirSync(path).flatMap((name) => files(join(path, name)));
  return /\.test\.tsx?$/.test(path) ? [] : [path];
}

const all = SOURCES.flatMap((source) => files(join(root, source)))
  .map((path) => relative(root, path).split(sep).join("/"))
  .sort();
const hash = createHash("sha256");
for (const path of all) {
  hash.update(path + "\0");
  hash.update(readFileSync(join(root, path)));
  hash.update("\0");
}
const out = join(root, "..", "resume_shortlister", "web", "static", "build-info.json");
writeFileSync(out, JSON.stringify({ source_hash: hash.digest("hex"), files: all.length }, null, 2) + "\n");
console.log(`Fingerprinted ${all.length} source files into ${relative(join(root, ".."), out)}`);
