import { execFileSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));
const IGNORED_DIRS = new Set(['node_modules', 'dist', 'fixtures']);

function collectFiles(directory) {
  const scripts = [];
  const tests = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      if (IGNORED_DIRS.has(entry.name)) continue;
      const nested = collectFiles(join(directory, entry.name));
      scripts.push(...nested.scripts);
      tests.push(...nested.tests);
      continue;
    }
    if (!entry.isFile()) continue;
    if (entry.name.endsWith('.js') || entry.name.endsWith('.mjs')) {
      const rel = relative(root, join(directory, entry.name));
      scripts.push(rel);
      if (entry.name.endsWith('.test.mjs')) tests.push(rel);
    }
  }
  return { scripts, tests };
}

const { scripts, tests } = collectFiles(root);
scripts.sort();
tests.sort();

for (const file of scripts) {
  execFileSync(process.execPath, ['--check', file], { cwd: root, stdio: 'inherit' });
}
execFileSync(process.execPath, ['--test', ...tests], { cwd: root, stdio: 'inherit' });
