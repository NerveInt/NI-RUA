import { build } from 'esbuild';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const sourceDirectory = dirname(fileURLToPath(import.meta.url));
const { values } = parseArgs({ options: { outdir: { type: 'string', default: 'dist' } } });
const outputDirectory = resolve(sourceDirectory, values.outdir);
const result = await build({
  absWorkingDir: sourceDirectory,
  entryPoints: { app: 'app.js', library: 'library.js', 'console-bundle': 'console.js', 'calibration-bundle': 'calibration.js' },
  bundle: true,
  minify: true,
  outdir: outputDirectory,
  write: false,
});

// Keep stable script URLs; imported CSS uses distinct bundle names so source
// console.css/calibration.css cannot overwrite esbuild's imported styles.
for (const file of result.outputFiles) {
  file.path = file.path.replace(/(console|calibration)-bundle\.js$/, '$1.js');
}

// Hash the exact delivered bytes, including CSS emitted from bundled imports.
const assets = new Map(result.outputFiles.map((file) => [basename(file.path), file.contents]));
for (const name of ['styles.css', 'library.css', 'console.css', 'calibration.css']) assets.set(name, await readFile(resolve(sourceDirectory, name)));
const versions = new Map([...assets].map(([name, contents]) => [name, createHash('sha256').update(contents).digest('hex').slice(0, 16)]));
const digests = new Map([...assets].map(([name, contents]) => [name, createHash('sha256').update(contents).digest('hex')]));

function versionAssets(html) {
  const referenced = [];
  const rewritten = html.replace(/\b(src|href)=(['"])([^'"<>]+)\2/gi, (attribute, name, quote, reference) => {
    const url = new URL(reference, 'https://build.invalid/');
    if (url.origin !== 'https://build.invalid' || !/\.(?:js|css)$/i.test(url.pathname)) return attribute;
    const assetName = basename(url.pathname);
    const version = versions.get(assetName);
    if (!version) throw new Error(`HTML references an asset that was not built: ${reference}`);
    if (!referenced.includes(assetName)) referenced.push(assetName);
    url.searchParams.set('v', version);
    // Preserve /app.js for both the standalone gateway and the library's root
    // asset route; only the query changes. Source HTML remains untouched.
    const originalPath = reference.split(/[?#]/, 1)[0];
    return `${name}=${quote}${originalPath}${url.search}${url.hash}${quote}`;
  });
  return { html: rewritten, assets: referenced };
}

const pages = new Map();
const pageManifest = {};
for (const name of ['index.html', 'library.html', 'console.html', 'calibration.html']) {
  const rendered = versionAssets(await readFile(resolve(sourceDirectory, name), 'utf8'));
  pages.set(name, rendered.html);
  pageManifest[name] = { file: name, assets: rendered.assets };
}

const manifest = {
  version: 1,
  assets: Object.fromEntries(
    [...assets].map(([name, contents]) => [
      name,
      {
        file: name,
        version: versions.get(name),
        sha256: digests.get(name),
        bytes: contents.byteLength,
      },
    ]),
  ),
  pages: pageManifest,
};

await mkdir(outputDirectory, { recursive: true });
await Promise.all([
  ...[...assets, ...pages].map(([name, contents]) => writeFile(resolve(outputDirectory, name), contents)),
  writeFile(resolve(outputDirectory, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n'),
]);
console.log(`Built ${assets.size} assets, ${pages.size} pages, and manifest.json in ${outputDirectory}`);
