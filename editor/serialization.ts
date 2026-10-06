import { parse } from 'acorn';
import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';
import { validateDocument, type SimulationDocument } from './document.ts';
export const jsonSource = (document: SimulationDocument) =>
  JSON.stringify(validateDocument(document), null, 2) + '\n';
export const javascriptSource = (document: SimulationDocument) =>
  `export const simulation = ${jsonSource(document).trim()};\n`;
export function parseJavascript(source: string): SimulationDocument {
  const program = parse(source, { ecmaVersion: 'latest', sourceType: 'module' });
  if (program.body.length !== 1)
    throw Error('Configuration JS must contain only export const simulation = { ... };');
  const statement = program.body[0];
  if (
    statement.type !== 'ExportNamedDeclaration' ||
    statement.declaration?.type !== 'VariableDeclaration' ||
    statement.declaration.kind !== 'const' ||
    statement.declaration.declarations.length !== 1
  )
    throw Error('Expected export const simulation = { ... };');
  const declaration = statement.declaration.declarations[0];
  if (
    declaration.id.type !== 'Identifier' ||
    declaration.id.name !== 'simulation' ||
    declaration.init?.type !== 'ObjectExpression'
  )
    throw Error('Expected a simulation object literal.');
  // JSON.parse deliberately rejects methods, calls, getters, spreads and computed values.
  return validateDocument(JSON.parse(source.slice(declaration.init.start, declaration.init.end)));
}
export function importDocument(name: string, bytes: Uint8Array): SimulationDocument {
  if (bytes.length > 50 * 1024 * 1024) throw Error('This file exceeds the 50 MB import limit.');
  const extension = name.split('.').pop()?.toLowerCase();
  if (extension === 'zip') {
    let total = 0,
      count = 0;
    const files = unzipSync(bytes, {
      filter(file) {
        total += file.originalSize;
        count++;
        if (
          count > 30 ||
          total > 80 * 1024 * 1024 ||
          file.name.split(/[\\/]/).some((p) => p === '..') ||
          file.name.startsWith('/') ||
          file.name.includes('\\')
        )
          throw Error('Invalid or oversized application archive.');
        return (
          file.name === 'simulation.js' ||
          file.name === 'index.html' ||
          file.name === 'app.js' ||
          file.name === 'runtime.js'
        );
      },
    });
    for (const required of ['simulation.js', 'index.html', 'app.js', 'runtime.js'])
      if (!files[required]) throw Error(`Application archive is missing ${required}.`);
    if (files['simulation.js'].length > 2 * 1024 * 1024)
      throw Error('Configuration exceeds the 2 MB limit.');
    return parseJavascript(strFromU8(files['simulation.js']));
  }
  const source = strFromU8(bytes);
  if (source.length > 2 * 1024 * 1024) throw Error('Configuration exceeds the 2 MB limit.');
  if (extension === 'json') return validateDocument(JSON.parse(source));
  if (extension === 'js') return parseJavascript(source);
  throw Error('Choose an exported application ZIP, simulation.js, or simulation.json.');
}
export const appSource = `import { startSimulation, demoActions } from './runtime.js';
import { simulation } from './simulation.js';

const container = document.querySelector('#viewport');
try {
  const runtime = await startSimulation(container, simulation);
  document.querySelector('#status').textContent = simulation.name;
  document.querySelector('#pause').onclick = (event) => {
    runtime.paused = !runtime.paused;
    event.currentTarget.textContent = runtime.paused ? 'Play' : 'Pause';
  };
  document.querySelector('#restart').onclick = () => runtime.restart();
  const actions = demoActions(simulation);
  document.querySelector('#effect-actions').hidden = !actions.launch && !actions.burst && !actions.continuous;
  const trigger = document.querySelector('#trigger');
  trigger.hidden = !actions.burst;
  trigger.onclick = () => { if (actions.resetOnBurst) runtime.restart(); runtime.trigger(); resume(); };
  const launch = document.querySelector('#launch');
  launch.hidden = !actions.launch;
  launch.onclick = () => { runtime.launch(); resume(); };
  document.querySelector('#reset-scene').hidden = !actions.launch && !actions.burst;
  document.querySelector('#reset-scene').onclick = () => runtime.restart();
  const emit = document.querySelector('#emit');
  emit.hidden = !actions.continuous;
  let emitting = actions.emitting;
  const label = () => {
    emit.textContent = emitting ? actions.stop || 'Stop emitters' : actions.start || 'Start emitters';
    emit.setAttribute('aria-pressed', String(emitting));
  };
  function resume() { runtime.paused = false; document.querySelector('#pause').textContent = 'Pause'; }
  label();
  emit.onclick = () => {
    emitting = !emitting;
    for (const source of simulation.emitters)
      if (source.mode === 'continuous') runtime.setEmitting(source.id, emitting);
    if (emitting) resume();
    label();
  };
  const lights = document.querySelector('#lights');
  const updateLights = () => {
    lights.setAttribute('aria-pressed', String(simulation.scene.sky));
    document.querySelector('#light-label').textContent = simulation.scene.sky ? 'Lights on' : 'Lights off';
  };
  updateLights();
  lights.onclick = async () => {
    simulation.scene.sky = !simulation.scene.sky;
    updateLights();
    try { await runtime.apply(simulation); }
    catch (error) { document.querySelector('#status').textContent = error.message; }
  };
  addEventListener('pagehide', () => runtime.dispose(), { once: true });
} catch (error) {
  document.querySelector('#status').textContent = error.message;
}
`;
export const htmlSource = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Fire Pro simulation</title>
  <style>
    body { margin:0; height:100dvh; display:grid; grid-template-rows:auto minmax(0,1fr); background:#141517; color:#eee; font:14px system-ui; }
    #viewport { grid-row:2; position:relative; min-height:0; min-width:0; }
    header { grid-row:1; display:grid; grid-template-columns:minmax(0,1fr) auto minmax(0,1fr); align-items:center; gap:12px; padding:8px 12px; border-bottom:1px solid #ffffff20; background:#1c1d20; }
    #status { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:12px; }
    #playback { justify-self:end; display:flex; gap:6px; }
    button { color:inherit; background:#26272c; border:1px solid #555; border-radius:6px; padding:8px 14px; cursor:pointer; }
    #demo-actions { display:flex; justify-content:center; align-items:center; gap:8px; }
    #effect-actions { display:flex; gap:6px; }
    #lights { display:flex; align-items:center; justify-content:center; gap:9px; min-width:110px; font-weight:600; }
    #lights[aria-pressed=true] { background:linear-gradient(135deg,#57412c,#342b24); color:#ffe1a3; border-color:#bf9358; }
    .light-icon { display:inline-flex; transition:transform .32s ease,color .32s ease; transform:rotate(-18deg) scale(.9); }
    #lights[aria-pressed=true] .light-icon { transform:rotate(0) scale(1); filter:drop-shadow(0 0 5px #ffb94777); }
    button:focus-visible { outline:2px solid #ed925e; outline-offset:3px; }
    @media(max-width:720px) { header { grid-template-columns:minmax(0,1fr) auto; gap:8px; } #demo-actions { grid-row:2; grid-column:1 / -1; padding-top:8px; border-top:1px solid #ffffff12; } #playback { grid-row:1; grid-column:2; } }
    @media(prefers-reduced-motion:reduce) { .light-icon { transition:none; } }
    #demo-actions button { min-height:36px; cursor:pointer; }
    #trigger, #launch, #emit[aria-pressed=false] { background:#ee935b; color:#181a20; border-color:#ee935b; }
    [hidden] { display:none !important; }
  </style>
</head>
<body>
  <div id="viewport"></div>
  <header><span id="status">Loading simulation…</span>
  <div id="demo-actions" role="group" aria-label="Demo controls">
    <div id="effect-actions"><button id="trigger">Detonate</button><button id="launch">Launch</button><button id="emit"></button><button id="reset-scene">Reset scene</button></div>
    <div><button id="lights" aria-label="Scene lights" aria-pressed="false"><span class="light-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M9 18h6m-5 4h4m-5-8a6 6 0 1 1 6 0c-1 1-1 2-1 4h-4c0-2 0-3-1-4Z"/></svg></span><span id="light-label">Lights off</span></button></div>
  </div>
  <div id="playback"><button id="pause">Pause</button><button id="restart">Restart</button></div></header>
  <script type="module" src="./app.js"></script>
</body>
</html>
`;
export async function exportApplication(document: SimulationDocument) {
  const response = await fetch('/export-runtime/runtime.js');
  if (!response.ok)
    throw Error('Export runtime is unavailable. Rebuild the application and try again.');
  const notices = await import('../THIRD_PARTY_NOTICES.md?raw');
  return zipSync(
    {
      'index.html': strToU8(htmlSource),
      'app.js': strToU8(appSource),
      'simulation.js': strToU8(javascriptSource(document)),
      'runtime.js': new Uint8Array(await response.arrayBuffer()),
      'README.txt': strToU8(
        'Serve this folder with a static HTTP server, then open index.html in a WebGPU browser.\nFor example: npx serve .\nNo dependency installation or source repository is required.\nImport the complete ZIP into Fire Pro to edit the authored configuration.\nChanges to app.js are not converted into editor settings.\n',
      ),
      'THIRD_PARTY_NOTICES.md': strToU8(notices.default),
    },
    { level: 6 },
  );
}
export function download(name: string, data: string | Uint8Array) {
  const blob = new Blob([typeof data === 'string' ? data : new Uint8Array(data)], {
    type: name.endsWith('.zip') ? 'application/zip' : 'text/plain',
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
