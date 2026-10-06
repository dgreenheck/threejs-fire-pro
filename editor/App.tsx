import { useCallback, useEffect, useRef, useState } from 'react';
import { MotionConfig } from 'motion/react';
import { Tooltip } from 'radix-ui';
import { Group, Panel, Separator, usePanelRef } from 'react-resizable-panels';
import {
  Flame,
  Plus,
  Upload,
  Download,
  Play,
  Pause,
  RotateCcw,
  Undo2,
  Redo2,
  Move,
  Rotate3D,
  Code2,
  Check,
  X,
  MoreHorizontal,
  PanelLeft,
  PanelRight,
  Box,
  Boxes,
  Eye,
  EyeOff,
  Lightbulb,
  LightbulbOff,
} from 'lucide-react';
import {
  cloneDocument,
  newDocument,
  newSource,
  newForce,
  transform,
  uid,
  validateDocument,
  SIMULATION_ID,
  type SimulationDocument,
} from './document';
import { presets, presetImage } from './presets';
import { loadDraft, loadPresets, saveDraft, savePreset, deletePreset } from './storage';
import { ObjectTree } from './components/Tree';
import { Inspector } from './components/Inspector';
import { ChoiceMenu, Modal, Menu } from './components/controls';
import { DEBUG_SCALES, type DebugField } from '../src/library/options';
import type { SimulationRuntime } from './runtime';
import {
  javascriptSource,
  jsonSource,
  htmlSource,
  appSource,
  importDocument,
  exportApplication,
  download,
} from './serialization';
import '@fontsource-variable/dm-sans';
import './style.css';
import { DemoActions } from './components/DemoActions';
import { PerformanceStats } from './components/PerformanceStats';
import {
  loadPreviewPreferences,
  applyPreviewPreferences,
  rememberPreviewPreferences,
} from './preview-preferences';

/** The preview's views: the final image, or one simulation field. */
const FIELD_VIEWS: readonly { value: DebugField; label: string }[] = [
  { value: 'beauty', label: 'Final image' },
  { value: 'smoke', label: 'Smoke' },
  { value: 'heat', label: 'Heat' },
  { value: 'lifetime', label: 'Flame lifetime' },
  { value: 'flame', label: 'Flame glow' },
  { value: 'fuel', label: 'Fuel' },
  { value: 'velocity', label: 'Velocity' },
  { value: 'vorticity', label: 'Vorticity' },
  { value: 'expansion', label: 'Expansion' },
  { value: 'pressure', label: 'Pressure' },
  { value: 'divergence', label: 'Divergence' },
];
/** The current view's heat map colors and range. */
function ViewLegend({ field }: { field: Exclude<DebugField, 'beauty'> }) {
  const { range, signed, unit } = DEBUG_SCALES[field];
  const label = FIELD_VIEWS.find((view) => view.value === field)!.label;
  const value = (n: number) => `${n}${unit ? ` ${unit}` : ''}`;
  return (
    <div className="view-legend" role="img" aria-label={`${label} color scale`}>
      <span className="view-legend-title">{label}</span>
      <span className={`view-legend-bar ${signed ? 'signed' : ''}`} />
      <span className="view-legend-scale">
        <span>{signed ? value(-range) : '0'}</span>
        {signed && <span>0</span>}
        <span>{value(range)}</span>
      </span>
    </div>
  );
}
export function App() {
  const [savedPreferences] = useState(loadPreviewPreferences);
  const preferences = useRef(savedPreferences);
  const [document, setDocument] = useState<SimulationDocument | null>(null);
  const current = useRef(document);
  current.current = document;
  const [personal, setPersonal] = useState<SimulationDocument[]>([]);
  const [selected, setSelected] = useState('scene'),
    [playing, setPlaying] = useState(true),
    [status, setStatus] = useState('Saved'),
    [error, setError] = useState(''),
    [notice, setNotice] = useState('');
  const [retry, setRetry] = useState(0);
  const [loading, setLoading] = useState(false),
    [mode, setMode] = useState<'translate' | 'rotate' | null>(null),
    [gizmosVisible, setGizmosVisible] = useState(false),
    [bricksVisible, setBricksVisible] = useState(false),
    [field, setField] = useState<DebugField>('beauty'),
    [gesture, setGesture] = useState(false),
    [, setHistoryVersion] = useState(0);
  const presetPanel = usePanelRef();
  const [presetsCollapsed, setPresetsCollapsed] = useState(false);
  const [mobilePanel, setMobilePanel] = useState<'presets' | 'properties' | null>(null);
  const [exportOpen, setExportOpen] = useState(false),
    [format, setFormat] = useState('app'),
    [exportFile, setExportFile] = useState('index.html'),
    [exporting, setExporting] = useState(false),
    [copied, setCopied] = useState(false);
  const [imported, setImported] = useState<SimulationDocument | null>(null),
    [importName, setImportName] = useState('');
  const [saveOpen, setSaveOpen] = useState(false),
    [presetName, setPresetName] = useState('');
  const input = useRef<HTMLInputElement>(null),
    viewport = useRef<HTMLDivElement>(null),
    runtime = useRef<SimulationRuntime | null>(null);
  const past = useRef<SimulationDocument[]>([]),
    future = useRef<SimulationDocument[]>([]),
    baseline = useRef<SimulationDocument | null>(null);
  const saveRevision = useRef(0);
  const latestSelection = useRef({ selected, mode, gizmosVisible });
  latestSelection.current = { selected, mode, gizmosVisible };
  const report = (value: unknown) =>
    setError(value instanceof Error ? value.message : String(value));
  /** Reload the saved presets. Only the first load reports ones it can't open. */
  const refreshPresets = () => loadPresets().then(({ presets }) => setPersonal(presets));
  useEffect(() => {
    let active = true;
    loadPresets()
      .then(({ presets, obsolete }) => {
        if (!active) return;
        setPersonal(presets);
        if (obsolete.length)
          setError(`Saved presets from an older version can't be opened: ${obsolete.join(', ')}.`);
      })
      .catch(report);
    if (!current.current) {
      loadDraft()
        .catch((e) => {
          report(e);
          return null;
        })
        .then((draft) => {
          if (!active || current.current) return;
          const next = applyPreviewPreferences(draft ?? newDocument(), preferences.current);
          current.current = next;
          setDocument(next);
          setSelected(SIMULATION_ID);
        });
    }
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    if (!document) return;
    setStatus('Saving…');
    const revision = ++saveRevision.current;
    const timer = setTimeout(() => {
      saveDraft(document)
        .then(() => {
          if (revision === saveRevision.current) {
            setStatus('Saved');
          }
        })
        .catch((e) => {
          setStatus('Save failed');
          report(e);
        });
    }, 500);
    return () => clearTimeout(timer);
  }, [document]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(''), 4000);
    return () => clearTimeout(timer);
  }, [notice]);
  const commit = useCallback((fn: (doc: SimulationDocument) => void, record = true) => {
    if (!current.current) return;
    try {
      const next = structuredClone(current.current);
      fn(next);
      const valid = validateDocument(next);
      if (JSON.stringify(valid) === JSON.stringify(current.current)) return;
      if (!baseline.current && record) {
        past.current.push(current.current);
        past.current = past.current.slice(-80);
        future.current = [];
      }
      rememberPreviewPreferences(current.current, valid, preferences.current);
      current.current = valid;
      setDocument(valid);
      setHistoryVersion((v) => v + 1);
      setError('');
    } catch (e) {
      report(e);
    }
  }, []);
  const begin = () => {
    if (!baseline.current) baseline.current = current.current;
    setGesture(true);
  };
  const end = useCallback(() => {
    if (baseline.current && baseline.current !== current.current) {
      past.current.push(baseline.current);
      future.current = [];
      setHistoryVersion((v) => v + 1);
    }
    baseline.current = null;
    setGesture(false);
  }, []);
  const undo = useCallback(() => {
    if (!past.current.length || !current.current) return;
    future.current.push(current.current);
    const prev = past.current.pop()!;
    rememberPreviewPreferences(current.current, prev, preferences.current);
    current.current = prev;
    setDocument(prev);
    setSelected('scene');
    setHistoryVersion((v) => v + 1);
  }, []);
  const redo = useCallback(() => {
    if (!future.current.length || !current.current) return;
    past.current.push(current.current);
    const next = future.current.pop()!;
    rememberPreviewPreferences(current.current, next, preferences.current);
    current.current = next;
    setDocument(next);
    setSelected('scene');
    setHistoryVersion((v) => v + 1);
  }, []);
  useEffect(() => {
    const listener = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).matches('input,textarea,[contenteditable=true]')) return;
      if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
        e.preventDefault();
        e.shiftKey ? redo() : undo();
      }
    };
    window.addEventListener('keydown', listener);
    return () => window.removeEventListener('keydown', listener);
  }, [undo, redo]);
  const open = (d: SimulationDocument) => {
    end();
    applyPreviewPreferences(d, preferences.current);
    setDocument(d);
    current.current = d;
    setSelected(SIMULATION_ID);
    setError('');
    setPlaying(true);
    past.current = [];
    future.current = [];
    setHistoryVersion((v) => v + 1);
    setMobilePanel(null);
  };
  const recipe = document?.scene.recipe,
    documentId = document?.id;
  useEffect(() => {
    if (!document || !viewport.current) return;
    let stopped = false;
    let instance: SimulationRuntime | undefined;
    setLoading(true);
    setError('');
    import('./runtime')
      .then(async ({ SimulationRuntime }) => {
        if (stopped) return;
        instance = new SimulationRuntime(viewport.current!, current.current!, {
          editor: true,
          select: setSelected,
          error: report,
          transform: (id, value) =>
            commit((d) => {
              const target = [...d.emitters, ...d.colliders].find((x) => x.id === id);
              if (target) Object.assign(target, value);
            }),
          camera: (position, target) =>
            commit((d) => {
              d.camera = { position, target };
            }, false),
        });
        runtime.current = instance;
        const selection = latestSelection.current;
        instance.select(selection.selected, selection.mode, selection.gizmosVisible);
        await instance.initialize();
        if (stopped) return;
        setLoading(false);
        instance.select(latestSelection.current.selected);
      })
      .catch((e) => {
        if (!stopped) {
          report(e);
          setLoading(false);
        }
      });
    return () => {
      stopped = true;
      instance?.dispose();
      runtime.current = null;
    };
  }, [documentId, recipe, commit, retry]);
  useEffect(() => {
    if (runtime.current && document) {
      runtime.current.apply(document, !gesture).catch(report);
    }
  }, [document, gesture]);
  useEffect(() => {
    runtime.current?.select(selected, mode, gizmosVisible);
  }, [selected, mode, gizmosVisible, loading]);
  useEffect(() => {
    runtime.current?.showBricks(bricksVisible);
  }, [bricksVisible, loading]);
  useEffect(() => {
    runtime.current?.showField(field);
  }, [field, loading]);
  useEffect(() => {
    if (runtime.current) runtime.current.paused = !playing;
  }, [playing, loading]);
  useEffect(() => {
    Object.assign(window, {
      __FIRE_EDITOR__: { document, runtime: runtime.current, error, loading },
    });
  }, [document, error, loading]);
  const add = (kind: string) => {
    let id = '';
    commit((d) => {
      if (['Wind', 'Turbulence', 'Vortex', 'Radial'].includes(kind)) {
        const f = newForce(kind.toLowerCase() as any);
        if (d.emitters.some((e) => e.id === selected && e.mode !== 'burst')) f.targets = [selected];
        id = f.id;
        d.forces.push(f);
      } else if (kind === 'Collider') {
        const c = {
          id: uid(),
          name: 'Collider',
          shape: 'box' as const,
          size: [1, 1, 1] as [number, number, number],
          radius: 0.5,
          ...transform(),
        };
        id = c.id;
        d.colliders.push(c);
      } else {
        const e = newSource();
        id = e.id;
        d.emitters.push(e);
      }
    });
    if (id) setSelected(id);
  };
  const rename = (id: string, name: string) =>
    commit((d) => {
      const item = [...d.emitters, ...d.forces, ...d.colliders].find((e) => e.id === id);
      if (item) item.name = name;
    });
  const remove = (id: string) => {
    if (id === SIMULATION_ID) return;
    commit((d) => {
      d.emitters = d.emitters.filter((e) => e.id !== id);
      d.forces = d.forces.filter((f) => f.id !== id);
      d.colliders = d.colliders.filter((c) => c.id !== id);
      for (const f of d.forces) if (f.targets) f.targets = f.targets.filter((t) => t !== id);
    });
    setSelected('scene');
    setNotice('Object deleted. Undo to restore it.');
  };
  const duplicate = (id: string) => {
    let next = '';
    commit((d) => {
      for (const key of ['emitters', 'forces', 'colliders'] as const) {
        const item = d[key].find((e) => e.id === id);
        if (item) {
          const copy = { ...structuredClone(item), id: uid(), name: item.name + ' copy' };
          (d[key] as any[]).push(copy);
          next = copy.id;
          return;
        }
      }
    });
    if (next) setSelected(next);
  };
  /** Move objects within their list; `parent` is the tree group they belong to. */
  const reorder = (ids: string[], parent: string, index: number) =>
    commit((d) => {
      const key = parent.split('/')[1];
      if (key !== 'emitters' && key !== 'forces' && key !== 'colliders') return;
      const list: { id: string }[] = d[key];
      const moving = list.filter((x) => ids.includes(x.id));
      const rest = list.filter((x) => !ids.includes(x.id));
      const before = list.slice(0, index).filter((x) => !ids.includes(x.id)).length;
      rest.splice(before, 0, ...moving);
      list.splice(0, list.length, ...rest);
    });
  const importFile = async (file: File) => {
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      const value = importDocument(file.name, bytes);
      setImported(value);
      setImportName(file.name);
      setError('');
    } catch (e) {
      report(e);
    }
  };
  const preview = document
    ? format === 'json'
      ? jsonSource(document)
      : format === 'js'
        ? javascriptSource(document)
        : exportFile === 'index.html'
          ? htmlSource
          : exportFile === 'app.js'
            ? appSource
            : javascriptSource(document)
    : '';
  const exportNow = async () => {
    if (!document) return;
    setExporting(true);
    try {
      const slug = document.name.toLowerCase().replace(/[^a-z0-9]+/g, '-') || 'simulation';
      if (format === 'app') download(`${slug}.zip`, await exportApplication(document));
      else download(format === 'js' ? 'simulation.js' : 'simulation.json', preview);
      setNotice('Export downloaded.');
    } catch (e) {
      report(e);
    } finally {
      setExporting(false);
    }
  };
  const savePersonal = async () => {
    if (!document) return;
    try {
      const value = validateDocument({ ...document, name: presetName });
      await savePreset(value);
      await refreshPresets();
      setSaveOpen(false);
      commit((d) => {
        d.name = presetName;
      });
      setNotice('Preset saved to this browser.');
    } catch (e) {
      report(e);
    }
  };
  return (
    <MotionConfig reducedMotion="user">
      <Tooltip.Provider delayDuration={250} skipDelayDuration={400}>
        <div
          className="fire-app"
          onDragOver={(e) => {
            if (e.dataTransfer.types.includes('Files')) e.preventDefault();
          }}
          onDrop={(e) => {
            if (e.dataTransfer.files.length) {
              e.preventDefault();
              void importFile(e.dataTransfer.files[0]);
            }
          }}
        >
          <input
            ref={input}
            type="file"
            accept=".zip,.js,.json"
            hidden
            onChange={(e) => {
              if (e.target.files?.[0]) void importFile(e.target.files[0]);
              e.target.value = '';
            }}
          />
          {!document ? (
            <div className="editor-startup" role="status">
              Opening editor…
            </div>
          ) : (
            <div className="editor-shell">
              <header className="editor-header">
                <div className="document-heading">
                  <span className="document-name">{document.name}</span>
                  <span className={`save-state ${status === 'Save failed' ? 'danger' : ''}`}>
                    {status === 'Saved' && <Check size={11} />} {status}
                  </span>
                </div>
                <div className="header-actions">
                  <button
                    className="button subtle"
                    aria-label="Import"
                    onClick={() => input.current?.click()}
                  >
                    <Upload size={14} />
                    <span>Import</span>
                  </button>
                  <button
                    className="button subtle"
                    onClick={() => {
                      setPresetName(document.name);
                      setSaveOpen(true);
                    }}
                  >
                    Save preset
                  </button>
                  <span className="mobile-header-menu">
                    <Menu
                      label="Preset actions"
                      icon={<MoreHorizontal size={16} />}
                      items={[
                        {
                          label: 'Save preset',
                          action: () => {
                            setPresetName(document.name);
                            setSaveOpen(true);
                          },
                        },
                      ]}
                    />
                  </span>
                  <button className="button primary" onClick={() => setExportOpen(true)}>
                    <Code2 size={15} />
                    <span>Export code</span>
                  </button>
                </div>
              </header>
              <div className="mobile-tools">
                <button
                  onClick={() => {
                    setPresetsCollapsed(false);
                    setMobilePanel(mobilePanel === 'presets' ? null : 'presets');
                  }}
                >
                  <PanelLeft size={16} />
                  Presets
                </button>
                <button
                  onClick={() => setMobilePanel(mobilePanel === 'properties' ? null : 'properties')}
                >
                  <PanelRight size={16} />
                  Properties
                </button>
              </div>
              <div className={`editor-workspace ${mobilePanel ? 'mobile-' + mobilePanel : ''}`}>
                <Group
                  orientation="horizontal"
                  className="workspace-panels"
                  defaultLayout={readLayout()}
                  onLayoutChanged={(layout) => {
                    try {
                      localStorage.setItem('fire-pro-workspace-layout', JSON.stringify(layout));
                    } catch {}
                  }}
                >
                  <Panel
                    id="presets"
                    defaultSize="220px"
                    minSize="180px"
                    maxSize="320px"
                    collapsedSize="48px"
                    collapsible
                    panelRef={presetPanel}
                    onResize={(size) => {
                      if (matchMedia('(min-width: 821px)').matches)
                        setPresetsCollapsed(size.inPixels < 100);
                    }}
                    className={`presets-pane ${presetsCollapsed ? 'collapsed' : ''}`}
                  >
                    <aside className="preset-browser" aria-label="Presets">
                      <div className="preset-browser-heading">
                        {!presetsCollapsed && <strong>Presets</strong>}
                        <button
                          aria-label={presetsCollapsed ? 'Expand presets' : 'Collapse presets'}
                          aria-expanded={!presetsCollapsed}
                          onClick={() => {
                            if (matchMedia('(max-width: 820px)').matches) {
                              setMobilePanel(null);
                              return;
                            }
                            if (presetPanel.current?.isCollapsed()) presetPanel.current.expand();
                            else presetPanel.current?.collapse();
                          }}
                        >
                          <PanelLeft size={17} />
                        </button>
                      </div>
                      {!presetsCollapsed && (
                        <div className="preset-browser-scroll">
                          <button
                            className="preset-browser-item new-preset"
                            onClick={() => open(newDocument())}
                          >
                            <span className="new-preset-image">
                              <Plus size={22} />
                            </span>
                            <span>New simulation</span>
                          </button>
                          <span className="preset-collection-label">Built-in</span>
                          {presets.map((p) => (
                            <button
                              className="preset-browser-item"
                              key={p.document.id}
                              onClick={() => open(cloneDocument(p.document))}
                            >
                              <img src={p.image} alt="" loading="lazy" />
                              <span>{p.document.name}</span>
                            </button>
                          ))}
                          {personal.length > 0 && (
                            <>
                              <span className="preset-collection-label">Your presets</span>
                              {personal.map((p) => (
                                <div className="personal-preset" key={p.id}>
                                  <button
                                    className="preset-browser-item"
                                    onClick={() => open(structuredClone(p))}
                                  >
                                    {presetImage(p) ? (
                                      <img src={presetImage(p)} alt="" loading="lazy" />
                                    ) : (
                                      <Flame size={18} />
                                    )}
                                    <span>{p.name}</span>
                                  </button>
                                  <Menu
                                    label={`Actions for ${p.name}`}
                                    icon={<MoreHorizontal size={17} />}
                                    items={[
                                      {
                                        label: 'Rename',
                                        action: () => {
                                          open(structuredClone(p));
                                          setPresetName(p.name);
                                          setSaveOpen(true);
                                        },
                                      },
                                      {
                                        label: 'Duplicate',
                                        action: () => {
                                          const copy = cloneDocument(p);
                                          copy.name += ' copy';
                                          savePreset(copy)
                                            .then(() => refreshPresets())
                                            .catch(report);
                                        },
                                      },
                                      {
                                        label: 'Delete',
                                        danger: true,
                                        action: () => {
                                          deletePreset(p.id)
                                            .then(() => refreshPresets())
                                            .catch(report);
                                          setNotice('Preset deleted.');
                                          deletedPreset.current = p;
                                        },
                                      },
                                    ]}
                                  />
                                </div>
                              ))}
                            </>
                          )}
                        </div>
                      )}
                    </aside>
                  </Panel>
                  <Separator className="panel-separator" />
                  <Panel id="viewport" minSize="240px" className="viewport-pane">
                    <section className="viewport-section">
                      <div ref={viewport} className="viewport" />
                      <div className="viewport-stats">
                        <PerformanceStats runtime={runtime} />
                      </div>
                      <div className="viewport-toolbar">
                        <div
                          className="viewport-tools"
                          role="toolbar"
                          aria-label="Preview controls"
                        >
                          <button
                            className="play-button"
                            title={playing ? 'Pause' : 'Play'}
                            aria-label={playing ? 'Pause' : 'Play'}
                            onClick={() => setPlaying(!playing)}
                          >
                            {playing ? <Pause size={14} /> : <Play size={14} />}
                          </button>
                          <button
                            aria-label="Restart"
                            title="Restart"
                            onClick={() => runtime.current?.restart()}
                          >
                            <RotateCcw size={14} />
                          </button>
                          <span className="toolbar-divider" />
                          <button
                            className="icon-button"
                            aria-label="Undo"
                            title="Undo"
                            disabled={!past.current.length}
                            onClick={undo}
                          >
                            <Undo2 size={15} />
                          </button>
                          <button
                            className="icon-button"
                            aria-label="Redo"
                            title="Redo"
                            disabled={!future.current.length}
                            onClick={redo}
                          >
                            <Redo2 size={15} />
                          </button>
                          <span className="toolbar-divider" />
                          <button
                            aria-label="Move tool"
                            aria-pressed={mode === 'translate'}
                            onClick={() => setMode(mode === 'translate' ? null : 'translate')}
                          >
                            <Move size={16} />
                          </button>
                          <button
                            aria-label="Rotate tool"
                            aria-pressed={mode === 'rotate'}
                            onClick={() => setMode(mode === 'rotate' ? null : 'rotate')}
                          >
                            <Rotate3D size={16} />
                          </button>
                          <button
                            aria-label="Preview gizmos"
                            aria-pressed={gizmosVisible}
                            title={gizmosVisible ? 'Hide preview gizmos' : 'Show preview gizmos'}
                            onClick={() => setGizmosVisible(!gizmosVisible)}
                          >
                            {gizmosVisible ? <Eye size={16} /> : <EyeOff size={16} />}
                          </button>
                          <button
                            aria-label="Simulation bricks"
                            aria-pressed={bricksVisible}
                            title={
                              bricksVisible
                                ? 'Hide simulation bricks'
                                : 'Show the simulation bricks the solver computes'
                            }
                            onClick={() => setBricksVisible(!bricksVisible)}
                          >
                            <Boxes size={16} />
                          </button>
                        </div>
                        <div className="preview-actions-slot">
                          <DemoActions
                            document={document}
                            disabled={loading || Boolean(error)}
                            emit={(active) => {
                              commit((d) => {
                                for (const e of d.emitters)
                                  if (e.mode === 'continuous') e.options.active = active;
                              });
                              if (active) setPlaying(true);
                            }}
                            launch={() => {
                              runtime.current?.launch();
                              setPlaying(true);
                            }}
                            detonate={(reset) => {
                              if (reset) runtime.current?.restart();
                              runtime.current?.trigger();
                              setPlaying(true);
                            }}
                            reset={() => runtime.current?.restart()}
                          />
                        </div>
                        <div className="preview-settings">
                          <button
                            className="sky-toggle"
                            role="switch"
                            aria-checked={document.scene.sky}
                            aria-label="Lights"
                            title="Turns the sky, sun and fill lights on or off."
                            onClick={() => commit((d) => void (d.scene.sky = !d.scene.sky))}
                          >
                            {document.scene.sky ? (
                              <Lightbulb size={15} />
                            ) : (
                              <LightbulbOff size={15} />
                            )}
                            <span>{document.scene.sky ? 'Lights On' : 'Lights Off'}</span>
                          </button>
                          <ChoiceMenu
                            label="View"
                            value={field}
                            options={FIELD_VIEWS}
                            onChange={setField}
                          />
                        </div>
                      </div>
                      {loading && (
                        <div className="viewport-state">
                          <span className="loader" />
                          <span>Lighting the scene…</span>
                        </div>
                      )}
                      {!loading && !document.emitters.length && (
                        <div className="viewport-state">
                          <Flame size={26} />
                          <h3>Start with a spark</h3>
                          <button className="button primary" onClick={() => add('Emitter')}>
                            <Plus size={15} />
                            Add emitter
                          </button>
                        </div>
                      )}
                      {field !== 'beauty' && <ViewLegend field={field} />}
                      {gizmosVisible && <div className="viewport-axis">Y ↑</div>}
                    </section>
                  </Panel>
                  <Separator className="panel-separator" />
                  <Panel
                    id="properties"
                    defaultSize="330px"
                    minSize="260px"
                    maxSize="440px"
                    className="inspector-pane"
                  >
                    <Group orientation="vertical" className="property-panels">
                      <Panel
                        id="hierarchy"
                        defaultSize="38%"
                        minSize="140px"
                        maxSize="55%"
                        className="tree-pane"
                      >
                        <ObjectTree
                          document={document}
                          selected={selected}
                          select={setSelected}
                          rename={rename}
                          remove={remove}
                          duplicate={duplicate}
                          reorder={reorder}
                          add={add}
                        />
                      </Panel>
                      <Separator className="panel-separator horizontal-separator" />
                      <Panel id="inspector" minSize="220px">
                        <Inspector
                          document={document}
                          selected={selected}
                          edit={commit}
                          select={setSelected}
                          begin={begin}
                          end={end}
                          trigger={(id) => runtime.current?.trigger(id)}
                        />
                      </Panel>
                    </Group>
                  </Panel>
                </Group>
              </div>
            </div>
          )}
          {error && (
            <div className="error-banner" role="alert">
              <span>{error}</span>
              {document && (
                <button className="button subtle" onClick={() => setRetry((v) => v + 1)}>
                  Retry preview
                </button>
              )}
              <button
                className="icon-button"
                aria-label="Dismiss error"
                onClick={() => setError('')}
              >
                <X size={15} />
              </button>
            </div>
          )}
          {notice && (
            <div className="toast" role="status">
              <Check size={15} />
              {notice}
              {deletedPreset.current && notice === 'Preset deleted.' && (
                <button
                  onClick={() => {
                    savePreset(deletedPreset.current!)
                      .then(() => refreshPresets())
                      .catch(report);
                    deletedPreset.current = null;
                    setNotice('Preset restored.');
                  }}
                >
                  Undo
                </button>
              )}
            </div>
          )}
          <Modal
            title="Export code"
            description="Take the complete simulation into your own scene."
            open={exportOpen}
            onClose={() => setExportOpen(false)}
            wide
          >
            <div className="export-formats">
              {[
                ['app', 'Full app', 'HTML + JavaScript'],
                ['js', 'JavaScript', 'Configuration module'],
                ['json', 'JSON', 'Portable configuration'],
              ].map(([value, title, caption]) => (
                <button
                  key={value}
                  aria-pressed={format === value}
                  onClick={() => setFormat(value)}
                >
                  <Code2 size={18} />
                  <strong>{title}</strong>
                  <small>{caption}</small>
                </button>
              ))}
            </div>
            {format === 'app' && (
              <div className="file-tabs">
                {['index.html', 'app.js', 'simulation.js'].map((file) => (
                  <button
                    key={file}
                    aria-pressed={exportFile === file}
                    onClick={() => setExportFile(file)}
                  >
                    {file}
                  </button>
                ))}
              </div>
            )}
            <pre className="export-preview">
              <code>{preview}</code>
            </pre>
            <p className="export-note">
              {format === 'app'
                ? 'Includes the runtime and assets. Serve the folder over HTTP. Re-import the ZIP to restore configuration; custom app.js edits are not converted to editor settings.'
                : 'Includes the simulation settings, every emitter, force target, and scene setting. Import this file to keep editing.'}
            </p>
            <div className="dialog-actions">
              <button
                className="button"
                onClick={() => {
                  navigator.clipboard
                    .writeText(preview)
                    .then(() => {
                      setCopied(true);
                      setTimeout(() => setCopied(false), 1400);
                    })
                    .catch(report);
                }}
              >
                {copied ? <Check size={14} /> : <Code2 size={14} />}{' '}
                {copied ? 'Copied' : 'Copy code'}
              </button>
              <button
                className="button primary"
                disabled={exporting}
                onClick={() => void exportNow()}
              >
                <Download size={15} />
                {exporting
                  ? 'Preparing…'
                  : format === 'app'
                    ? 'Download app ZIP'
                    : 'Download configuration'}
              </button>
            </div>
          </Modal>
          <Modal
            title="Open simulation"
            description="Your current simulation will be kept in Your presets."
            open={Boolean(imported)}
            onClose={() => setImported(null)}
          >
            <div className="import-review">
              <Box size={28} />
              <h3>{imported?.name}</h3>
              <p>{importName}</p>
              <span className="import-valid">
                <Check size={14} /> Configuration validated
              </span>
            </div>
            <div className="dialog-actions">
              <button className="button" onClick={() => setImported(null)}>
                Cancel
              </button>
              <button
                className="button primary"
                onClick={async () => {
                  try {
                    if (current.current) {
                      await saveDraft(current.current);
                      await savePreset(current.current);
                      await refreshPresets();
                    }
                    open({ ...imported!, id: uid() });
                    setImported(null);
                  } catch (e) {
                    report(e);
                  }
                }}
              >
                Open simulation
              </button>
            </div>
          </Modal>
          <Modal
            title="Save preset"
            description="Keep this simulation in your personal preset collection."
            open={saveOpen}
            onClose={() => setSaveOpen(false)}
          >
            <label className="text-field">
              Preset name
              <input
                autoFocus
                value={presetName}
                onChange={(e) => setPresetName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') void savePersonal();
                }}
              />
            </label>
            <div className="dialog-actions">
              <button className="button" onClick={() => setSaveOpen(false)}>
                Cancel
              </button>
              <button
                className="button primary"
                disabled={!presetName.trim()}
                onClick={() => void savePersonal()}
              >
                Save preset
              </button>
            </div>
          </Modal>
        </div>
      </Tooltip.Provider>
    </MotionConfig>
  );
  function readLayout() {
    try {
      return JSON.parse(localStorage.getItem('fire-pro-workspace-layout') ?? 'null') ?? undefined;
    } catch {
      return undefined;
    }
  }
}
const deletedPreset = { current: null as SimulationDocument | null };
