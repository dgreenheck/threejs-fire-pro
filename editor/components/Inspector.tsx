import { useEffect, useState, type ComponentType, type ReactNode } from 'react';
import {
  Camera,
  ArrowUpRight,
  Box,
  CloudFog,
  Flame,
  Fuel,
  Lightbulb,
  Mountain,
  SlidersHorizontal,
  Shield,
  Sparkles,
  Wind,
} from 'lucide-react';
import {
  Section,
  NumberField,
  VectorField,
  Toggle,
  Choice,
  ColorField,
  ScrubField,
  TargetCheck,
  Menu,
} from './controls';
import {
  DOCUMENT_LIMITS,
  newForce,
  type SimulationDocument,
  type ForceDocument,
} from '../document';
import { LIMITS } from '../../src/library/options';
import { HINTS } from '../hints';
type TabIcon = ComponentType<{ size?: number }>;
interface Props {
  document: SimulationDocument;
  selected: string;
  edit: (fn: (d: SimulationDocument) => void) => void;
  select: (id: string) => void;
  begin: () => void;
  end: () => void;
  trigger: (id: string) => void;
}
const names: Record<string, string> = {
  heatRate: 'Heat',
  ignitionHeat: 'Ignition heat',
  burnRate: 'Burn speed',
  fuelRate: 'Fuel',
  smokeRate: 'Smoke',
  flame: 'Flame',
  speed: 'Speed',
  strength: 'Strength',
  scale: 'Scale',
  radius: 'Radius',
  lift: 'Lift',
  inward: 'Inward pull',
  raySteps: 'Ray steps',
  smokeWeight: 'Smoke weight',
  dissipation: 'Dissipation',
  expansionRate: 'Expansion',
  brightness: 'Brightness',
  opacity: 'Opacity',
  temperature: 'Temperature',
  sootGlow: 'Soot glow',
  shadowDensity: 'Shadow density',
  illuminateScene: 'Illuminate scene',
  outwardSpeed: 'Outward speed',
  period: 'Variation period',
  maxVoxels: 'Max voxels',
  cutoff: 'Cutoff',
  ground: 'Ground',
};
const flameLabels: Record<string, string> = {
  lifespan: 'Duration',
  heatRate: 'Heat rate',
  smokeRate: 'Smoke rate',
  cooling: 'Heat decay',
  temperature: 'Color temp.',
  sootGlow: 'Smoke glow',
};
const flameUnits: Record<string, string> = {
  lifespan: 's',
  heatRate: '/s',
  smokeRate: '/s',
  expansionRate: '/s',
  cooling: '/s',
  temperature: 'K',
};
const flameBehavior = ['lifespan', 'heatRate', 'smokeRate', 'expansionRate', 'cooling'] as const;
const flameAppearance = ['color', 'brightness', 'opacity', 'temperature', 'sootGlow'] as const;
const filterNames: Record<string, string> = {
  trilinear: 'Trilinear',
  quadratic: 'Quadratic B-spline',
  cubic: 'Cubic B-spline',
};
const resolutionNames: Record<number, string> = {
  1: 'Full resolution',
  2: 'Half resolution',
  4: 'Quarter resolution',
};
type Range = readonly [number, number];
type SimulationGroup = 'grid' | 'flame' | 'smoke' | 'motion' | 'fuel' | 'lighting' | 'rendering';
type Tab = 'scene' | 'simulation' | 'quality' | Exclude<SimulationGroup, 'grid'> | 'object';
const simulationTabs: { id: Tab; label: string; icon: TabIcon }[] = [
  { id: 'simulation', label: 'Simulation', icon: Box },
  { id: 'motion', label: 'Motion', icon: Wind },
  { id: 'flame', label: 'Flame', icon: Flame },
  { id: 'smoke', label: 'Smoke', icon: CloudFog },
  { id: 'fuel', label: 'Fuel', icon: Fuel },
  { id: 'lighting', label: 'Lighting', icon: Lightbulb },
  { id: 'rendering', label: 'Rendering', icon: Camera },
  { id: 'quality', label: 'Quality', icon: SlidersHorizontal },
];
/** Settings that only take whole numbers. */
const integerKeys = new Set(['maxVoxels', 'raySteps']);
/** Keys whose whole valid range suits a slider despite being wide. */
const sliderKeys = new Set(['temperature', 'raySteps']);
/** Keys that take a number box with arrows despite a narrow range: the amounts, rates and
 * strengths of the smoke, flame, motion, fuel and lighting. */
const numberKeys = new Set([
  'dissipation',
  'density',
  'shadowDensity',
  'lifespan',
  'heatRate',
  'smokeRate',
  'expansionRate',
  'cooling',
  'brightness',
  'opacity',
  'sootGlow',
  'buoyancy',
  'smokeWeight',
  'damping',
  'vorticity',
  'ignitionHeat',
  'burnRate',
  'intensity',
]);
/** Wide ranges such as rates and strengths use a drag-and-step number box, not a slider. */
const wide = (key: string, [min, max]: Range) =>
  numberKeys.has(key) || (max - min > 25 && !sliderKeys.has(key));
export function Inspector({ document, selected, edit, select, begin, end, trigger }: Props) {
  const source = document.emitters.find((e) => e.id === selected),
    force = document.forces.find((f) => f.id === selected),
    collider = document.colliders.find((c) => c.id === selected);
  const object = source ?? force ?? collider;
  const kind = source ? 'Emitter' : force ? 'Force' : 'Collider';
  const settings = document.simulation;
  const [tab, setTab] = useState<Tab>(() =>
    object ? 'object' : selected !== 'scene' ? 'simulation' : 'scene',
  );
  useEffect(() => {
    if (object) setTab('object');
    else if (selected === 'scene') setTab('scene');
    else
      setTab((current) => (current === 'object' || current === 'scene' ? 'simulation' : current));
  }, [selected, Boolean(object)]);
  const active: Tab = tab === 'object' && !object ? 'simulation' : tab;

  const assign = (target: any, path: string[], value: unknown) => {
    for (const key of path.slice(0, -1)) target = target[key];
    target[path.at(-1)!] = value;
  };
  const patchScene = (path: string[], value: unknown) => edit((d) => assign(d.scene, path, value));
  const patchSimulation = (path: string[], value: unknown) =>
    edit((d) => assign(d.simulation, path, value));
  const patchObject = (path: string[], value: unknown) =>
    edit((d) =>
      assign(
        [...d.emitters, ...d.forces, ...d.colliders].find((item) => item.id === selected),
        path,
        value,
      ),
    );

  /** Rows for every field of a settings object, with each field's valid range and what it
   * does (HINTS, by its path unless `hint` finds it). */
  function fields(
    value: Record<string, any>,
    path: string[],
    patch: (path: string[], value: unknown) => void,
    limits: (key: string) => Range | undefined,
    hint: (path: string[]) => string | undefined = (p) => HINTS[p.join('.')],
  ): ReactNode {
    return Object.entries(value)
      .filter(([key]) => !['type', 'active'].includes(key))
      .map(([key, item]) => {
        const label =
          (path[0] === 'flame' ? flameLabels[key] : undefined) ??
          names[key] ??
          key[0].toUpperCase() + key.slice(1);
        const next = [...path, key];
        if (key === 'filter')
          return (
            <Choice
              key={key}
              label="Filtering"
              hint={hint(next)}
              value={filterNames[String(item)]}
              options={Object.values(filterNames)}
              onChange={(label) =>
                patch(
                  next,
                  Object.keys(filterNames).find((key) => filterNames[key] === label)!,
                )
              }
            />
          );
        if (typeof item === 'number') {
          const range = limits(key);
          if (!range) throw new Error(`No valid range for ${next.join('.')}.`);
          const Field = wide(key, range) ? ScrubField : NumberField;
          return (
            <Field
              key={key}
              label={label}
              unit={path[0] === 'flame' ? flameUnits[key] : undefined}
              hint={hint(next)}
              value={item}
              min={range[0]}
              max={range[1]}
              {...(key === 'raySteps' ? { step: 8 } : {})}
              integer={integerKeys.has(key)}
              onChange={(n) => patch(next, n)}
              begin={begin}
              end={end}
            />
          );
        }
        if (typeof item === 'boolean')
          return (
            <Toggle
              key={key}
              label={label}
              hint={hint(next)}
              value={item}
              onChange={(v) => patch(next, v)}
            />
          );
        if (Array.isArray(item) && item.length === 3)
          return (
            <VectorField
              key={key}
              label={label}
              hint={hint(next)}
              value={item}
              onChange={(v) => patch(next, v)}
              begin={begin}
              end={end}
            />
          );
        if (typeof item === 'string' && item.startsWith('#'))
          return (
            <ColorField
              key={key}
              label={label}
              hint={hint(next)}
              value={item}
              onChange={(v) => patch(next, v)}
            />
          );
        if (item && typeof item === 'object')
          return (
            <div className="property-subsection" key={key}>
              <div className="subsection-label">{label}</div>
              {fields(item, next, patch, limits, hint)}
            </div>
          );
        return null;
      });
  }
  const simulationFields = (group: SimulationGroup, value: Record<string, unknown>) =>
    fields(value, [group], patchSimulation, (key) => (LIMITS[group] as Record<string, Range>)[key]);

  function sceneTab() {
    const s = document.scene;
    const scene = (section: keyof typeof DOCUMENT_LIMITS) => (key: string) =>
      (DOCUMENT_LIMITS[section] as Record<string, Range>)[key];
    return (
      <>
        <p className="help scene-note">
          The world around the fire. These settings don’t change the simulation.
        </p>
        <Section title="Lights">
          <Toggle
            label="Sky and sun"
            hint={HINTS.sky}
            value={s.sky}
            onChange={(v) => patchScene(['sky'], v)}
          />
        </Section>
        <Section title="Floor">{fields(s.floor, ['floor'], patchScene, scene('floor'))}</Section>
      </>
    );
  }

  function simulationTab() {
    return (
      <>
        <Section title="Grid">
          <ScrubField
            label="Voxel size"
            hint={HINTS.voxelSize}
            unit="m"
            value={settings.voxelSize}
            min={LIMITS.simulation.voxelSize[0]}
            max={LIMITS.simulation.voxelSize[1]}
            precision={0.0001}
            onChange={(n) => patchSimulation(['voxelSize'], n)}
            begin={begin}
            end={end}
          />
          {simulationFields('grid', settings.grid)}
          <p className="help">
            The simulation computes cells near the emitters and the flame and smoke above the
            cutoff, wherever they are. Past the voxel budget, the rest stays empty. The ground is
            solid below y = 0.
          </p>
        </Section>
        {!document.emitters.length && (
          <p className="empty-note">Add an emitter to bring this simulation to life.</p>
        )}
      </>
    );
  }

  function groupTab(group: Exclude<SimulationGroup, 'grid'>, title: string) {
    const options = settings;
    const value: Record<string, unknown> =
      group === 'fuel' && !options.fuel.enabled ? { enabled: false } : { ...options[group] };
    return (
      <Section title={title}>
        {simulationFields(group, value)}
        {group === 'fuel' && (
          <p className="help">
            The flow carries fuel, which burns as flame where it is hot enough: the Flame settings
            give its heat, smoke and expansion. Stopping emission preserves existing fuel; disabling
            fuel clears it and keeps fire and smoke.
          </p>
        )}
      </Section>
    );
  }

  function flameFields(keys: readonly (keyof typeof settings.flame)[]) {
    return simulationFields(
      'flame',
      Object.fromEntries(keys.map((key) => [key, settings.flame[key]])),
    );
  }

  function flameTab() {
    const cooling = settings.flame.cooling;
    return (
      <Section title="Flame behavior">
        <p className="help">
          Fresh flame fades as it moves. Duration is its lifetime after emission, unless an emitter
          or burning fuel replenishes it.
        </p>
        {flameFields(flameBehavior)}
        <p className="help">
          Heat drives buoyancy and fuel ignition; smoke adds to the carried smoke field. These are
          relative amounts produced per second, varying as flame ages. Emitters can also add heat
          and smoke directly.
        </p>
        <p className="help">
          {cooling > 0
            ? `Without new heating, heat halves in ${(Math.LN2 / cooling).toFixed(2)} s.`
            : 'Heat decay is off: stored heat does not cool.'}{' '}
          Higher expansion pushes gas outward more strongly; 0 turns that contribution off.
        </p>
        <p className="help">Flame color, brightness and opacity are in Rendering.</p>
      </Section>
    );
  }

  function renderingTab() {
    return (
      <>
        <Section title="Flame appearance">
          {flameFields(flameAppearance)}
          <p className="help">
            Color temperature changes the glow’s color. It does not heat the gas. Smoke glow
            controls light emitted by hot smoke.
          </p>
        </Section>
      </>
    );
  }

  function qualityTab() {
    return (
      <>
        <Section title="Simulation quality">
          <Toggle
            label="Fine Detail"
            hint="Preserve fine flame and smoke detail during transport. Turn off for faster simulation with softer detail. Changing this restarts the simulation."
            value={settings.scalarMacCormack}
            onChange={(value) => patchSimulation(['scalarMacCormack'], value)}
          />
          <Choice
            label="Brick size"
            hint="Fine cells along each brick side. Independent of velocity resolution. Changing this restarts the simulation."
            value={String(settings.brickSize)}
            options={['8', '16', '32']}
            onChange={(value) => patchSimulation(['brickSize'], Number(value))}
          />
          <p className="help">
            {settings.brickSize} × {settings.brickSize} × {settings.brickSize} fine cells per brick.
            Smaller bricks follow sparse detail more closely; larger bricks need fewer allocations.
          </p>
          <Choice
            label="Velocity grid"
            hint={HINTS.velocityGrid}
            value={resolutionNames[settings.velocityDivisor]}
            options={Object.values(resolutionNames)}
            onChange={(label) =>
              patchSimulation(
                ['velocityDivisor'],
                Number(
                  Object.keys(resolutionNames).find(
                    (key) => resolutionNames[Number(key)] === label,
                  ),
                ),
              )
            }
          />
          <p className="help">
            Velocity and pressure use{' '}
            {Number((settings.voxelSize * settings.velocityDivisor).toFixed(4))} m cells. Flame and
            heat keep the voxel size. Coarser motion uses fewer cells but loses detail. Brick size stays unchanged.
          </p>
          <Choice
            label="Smoke grid"
            hint={HINTS.smokeGrid}
            value={resolutionNames[settings.smokeDivisor]}
            options={Object.values(resolutionNames)}
            onChange={(label) =>
              patchSimulation(
                ['smokeDivisor'],
                Number(
                  Object.keys(resolutionNames).find(
                    (key) => resolutionNames[Number(key)] === label,
                  ),
                ),
              )
            }
          />
          <p className="help">
            Smoke is simulated and transported in{' '}
            {Number((settings.voxelSize * settings.smokeDivisor).toFixed(4))} m cells. Half uses 8×
            fewer smoke cells; quarter uses 64× fewer, with softer detail. Changing this restarts
            the simulation.
          </p>
        </Section>
        <Section title="Render quality">
          {simulationFields('rendering', {
            filter: settings.rendering.filter,
            raySteps: settings.rendering.raySteps,
          })}
          <Choice
            label="Lighting resolution"
            hint={HINTS['rendering.lightingDivisor']}
            value={resolutionNames[settings.rendering.lightingDivisor]}
            options={Object.values(resolutionNames)}
            onChange={(label) =>
              patchSimulation(
                ['rendering', 'lightingDivisor'],
                Number(
                  Object.keys(resolutionNames).find((key) => resolutionNames[Number(key)] === label),
                ),
              )
            }
          />
          <p className="help">
            Quarter resolution uses 64× fewer interior lighting cells, with softer shadows.
            Changes apply immediately without restarting the simulation.
          </p>
          <Choice
            label="Render resolution"
            hint={HINTS['rendering.halfResolution']}
            value={settings.rendering.halfResolution ? 'Half resolution' : 'Full resolution'}
            options={['Full resolution', 'Half resolution']}
            onChange={(label) =>
              patchSimulation(['rendering', 'halfResolution'], label === 'Half resolution')
            }
          />
          <p className="help">
            Trilinear is fastest; quadratic and cubic smooth the grid. Ray steps limits samples per
            ray. Half resolution renders the volume at half width and height for a faster, softer
            image.
          </p>
        </Section>
      </>
    );
  }

  function setEmission(channel: string, amount: number) {
    edit((d) => {
      const e = d.emitters.find((e) => e.id === source!.id)!;
      const burst = e.mode === 'burst';
      const block = (burst ? e.burst.charge : e.options.emission) as Record<string, number>;
      const key = channel === 'flame' || burst ? channel : `${channel}Rate`;
      block[key] = amount;
      if (amount > 0 && channel === 'fuel') d.simulation.fuel.enabled = true;
    });
  }
  /** One always-visible number per channel; zero means the channel is off. */
  function emissionControls() {
    const burst = source!.mode === 'burst';
    const values = (burst ? source!.burst.charge : source!.options.emission) as Record<
      string,
      number
    >;
    return ['flame', 'heat', 'smoke', 'fuel'].map((channel) => {
      const key = channel === 'flame' || burst ? channel : `${channel}Rate`;
      const value = values[key] ?? 0;
      const fraction = channel === 'flame';
      const range: Range = burst
        ? fraction
          ? LIMITS.explosion.flame
          : LIMITS.explosion.amount
        : fraction
          ? LIMITS.emitter.flame
          : LIMITS.emitter.rate;
      return (
        <ScrubField
          key={channel}
          label={channel[0].toUpperCase() + channel.slice(1)}
          hint={HINTS[`${burst ? 'charge' : 'emission'}.${channel}`]}
          unit={burst || fraction ? '' : 'per s'}
          value={value}
          off={value === 0}
          min={range[0]}
          max={range[1]}
          begin={begin}
          end={end}
          onChange={(n) => setEmission(channel, n)}
        />
      );
    });
  }
  const targeting = (f: ForceDocument, ids: string[] | null) =>
    edit((d) => {
      d.forces.find((x) => x.id === f.id)!.targets = ids;
    });

  function objectTab() {
    return (
      <>
        {(source ?? collider) && (
          <Section title="Transform">
            <VectorField
              begin={begin}
              end={end}
              label="Position"
              hint={HINTS.position}
              value={(source ?? collider)!.position}
              onChange={(v) => patchObject(['position'], v)}
            />
            <VectorField
              begin={begin}
              end={end}
              label="Rotation"
              hint={HINTS.rotation}
              value={(source ?? collider)!.rotation}
              onChange={(v) => patchObject(['rotation'], v)}
              rotation
            />
          </Section>
        )}
        {source && emitterSections()}
        {force && forceSections(force)}
        {collider && (
          <Section title="Shape">
            <Choice
              label="Shape"
              hint={HINTS.colliderShape}
              value={collider.shape}
              options={['sphere', 'box']}
              onChange={(v) => patchObject(['shape'], v)}
            />
            {collider.shape === 'box' ? (
              <VectorField
                begin={begin}
                end={end}
                label="Size"
                hint={HINTS.colliderSize}
                value={collider.size}
                min={LIMITS.collider.size[0]}
                max={LIMITS.collider.size[1]}
                onChange={(v) => patchObject(['size'], v)}
              />
            ) : (
              <ScrubField
                label="Radius"
                hint={HINTS.colliderRadius}
                value={collider.radius}
                min={LIMITS.collider.radius[0]}
                max={LIMITS.collider.radius[1]}
                onChange={(v) => patchObject(['radius'], v)}
                begin={begin}
                end={end}
              />
            )}
          </Section>
        )}
      </>
    );
  }

  function emitterSections() {
    const e = source!;
    return (
      <>
        <Section title="Emission">
          <Choice
            label="Mode"
            hint={HINTS.mode}
            value={e.mode === 'burst' ? 'Burst' : 'Continuous'}
            options={['Continuous', 'Burst']}
            onChange={(value) =>
              edit((d) => {
                const item = d.emitters.find((item) => item.id === e.id)!;
                item.mode = value === 'Burst' ? 'burst' : 'continuous';
                if (item.mode === 'burst')
                  d.forces.forEach((f) => {
                    if (f.targets) f.targets = f.targets.filter((id) => id !== item.id);
                  });
              })
            }
          />
          {e.mode === 'continuous' && (
            <Toggle
              label="Emitting"
              hint={HINTS.emitting}
              value={e.options.active !== false}
              onChange={(v) => patchObject(['options', 'active'], v)}
            />
          )}
          {e.mode === 'burst' && <div className="subsection-label">Amount per burst</div>}
          {emissionControls()}
          <p className="help">
            {e.mode === 'burst'
              ? 'Each trigger adds these amounts once. The presets use heat 2.5–4, smoke 0.05–0.6 and fuel 0.6–0.85.'
              : 'Flame sets how young new flame is; heat, smoke and fuel are added every second. The presets use heat 1–3 for small flames and 12–22 for jets, smoke up to 0.15 with fire and about 20 for a smoke plume, and fuel about 10.'}{' '}
            Drag a value or use the arrows; 0 turns a channel off.
            {!settings.fuel.enabled && ' Fuel above 0 turns on combustion.'}
          </p>
        </Section>
        {e.mode === 'burst' && (
          <Section title="Burst">
            {fields(
              {
                radius: e.burst.radius,
                outwardSpeed: e.burst.outwardSpeed,
                variation: e.burst.variation,
              },
              ['burst'],
              patchObject,
              (key) => (LIMITS.explosion as Record<string, Range>)[key],
            )}
            <button
              className="button primary full"
              title="Release one burst now."
              onClick={() => trigger(e.id)}
            >
              Trigger burst
            </button>
          </Section>
        )}
        {e.mode !== 'burst' && (
          <>
            <Section title="Shape">
              <Choice
                label="Shape"
                hint={HINTS.emitterShape}
                value={e.shape}
                options={
                  document.scene.recipe === 'flaming-letters'
                    ? ['sphere', 'box', 'disk', 'torus', 'letters']
                    : ['sphere', 'box', 'disk', 'torus']
                }
                onChange={(v) => patchObject(['shape'], v)}
              />
              {e.shape !== 'letters' && e.shape !== 'box' && (
                <NumberField
                  label="Radius"
                  hint={HINTS.emitterRadius}
                  value={e.radius}
                  min={LIMITS.emitter.radius[0]}
                  max={LIMITS.emitter.radius[1]}
                  onChange={(v) => patchObject(['radius'], v)}
                  begin={begin}
                  end={end}
                />
              )}
              {e.shape === 'box' && (
                <VectorField
                  begin={begin}
                  end={end}
                  label="Size"
                  hint={HINTS.emitterSize}
                  value={e.size}
                  min={DOCUMENT_LIMITS.emitter.size[0]}
                  max={DOCUMENT_LIMITS.emitter.size[1]}
                  onChange={(v) => patchObject(['size'], v)}
                />
              )}
            </Section>
            <Section title="Launch velocity">
              <Toggle
                label="Launch velocity"
                hint={HINTS.launchVelocity}
                value={e.options.velocity !== null}
                onChange={(v) =>
                  patchObject(
                    ['options', 'velocity'],
                    v ? { direction: [0, 1, 0], speed: 1 } : null,
                  )
                }
              />
              {e.options.velocity &&
                fields(
                  e.options.velocity,
                  ['options', 'velocity'],
                  patchObject,
                  (key) => (key === 'speed' ? LIMITS.emitter.speed : undefined),
                  (path) => HINTS[`velocity.${path.at(-1)}`],
                )}
            </Section>
            <Section title="Affecting forces">
              {document.forces
                .filter((f) => f.targets === null || f.targets.includes(e.id))
                .map((f) => (
                  <div className="force-assignment" key={f.id}>
                    <button title={`Select ${f.name}.`} onClick={() => select(f.id)}>
                      <Wind size={15} />
                      <span>
                        {f.name}
                        <small>{f.targets === null ? 'Everywhere' : 'Emitter region'}</small>
                      </span>
                      <ArrowUpRight size={13} />
                    </button>
                    {f.targets !== null && (
                      <button
                        className="remove-target"
                        aria-label={`Remove ${f.name} from ${e.name}`}
                        title={`Stop ${f.name} acting on ${e.name}.`}
                        onClick={() =>
                          targeting(
                            f,
                            f.targets!.filter((id) => id !== e.id),
                          )
                        }
                      >
                        ×
                      </button>
                    )}
                  </div>
                ))}
              {document.forces.some((f) => f.targets !== null && !f.targets.includes(e.id)) && (
                <Menu
                  label="Add existing force"
                  items={document.forces
                    .filter((f) => f.targets !== null && !f.targets.includes(e.id))
                    .map((f) => ({
                      label: f.name,
                      action: () => targeting(f, [...f.targets!, e.id]),
                    }))}
                />
              )}
            </Section>
          </>
        )}
      </>
    );
  }

  function forceSections(f: ForceDocument) {
    return (
      <>
        <Section title="Field">
          <Toggle
            label="Enabled"
            hint={HINTS.forceEnabled}
            value={f.options.active !== false}
            onChange={(v) => patchObject(['options', 'active'], v)}
          />
          <Choice
            label="Force type"
            hint={HINTS.forceType}
            value={f.options.type}
            options={['wind', 'turbulence', 'vortex', 'radial']}
            onChange={(v) => patchObject(['options'], newForce(v as any).options)}
          />
          {fields(
            f.options,
            ['options'],
            patchObject,
            (key) => (LIMITS[f.options.type] as Record<string, Range>)[key],
            (path) => HINTS[`${f.options.type}.${path.at(-1)}`],
          )}
        </Section>
        <Section title="Apply to">
          <div className="segmented">
            <button
              aria-pressed={f.targets === null}
              title="Act on the whole simulation."
              onClick={() => targeting(f, null)}
            >
              Everywhere
            </button>
            <button
              aria-pressed={f.targets !== null}
              title="Act only around the emitters chosen below."
              onClick={() =>
                targeting(
                  f,
                  document.emitters.filter((e) => e.mode !== 'burst').map((e) => e.id),
                )
              }
            >
              Selected emitters
            </button>
          </div>
          {f.targets !== null &&
            document.emitters
              .filter((e) => e.mode !== 'burst')
              .map((e) => (
                <TargetCheck
                  key={e.id}
                  name={e.name}
                  checked={f.targets!.includes(e.id)}
                  onChange={() =>
                    targeting(
                      f,
                      f.targets!.includes(e.id)
                        ? f.targets!.filter((id) => id !== e.id)
                        : [...f.targets!, e.id],
                    )
                  }
                />
              ))}
          <p className="help">
            {f.targets === null
              ? 'Applies everywhere in the simulation.'
              : f.targets.length
                ? 'Acts within the selected emitter regions. Overlap does not increase its strength.'
                : 'Choose an emitter to apply this force.'}
          </p>
        </Section>
      </>
    );
  }

  const objectIcon = source ? Sparkles : force ? Wind : Shield;
  const tabs: { id: Tab; label: string; icon: TabIcon }[] = [
    { id: 'scene', label: 'Scene', icon: Mountain },
    ...simulationTabs,
    ...(object ? [{ id: 'object' as Tab, label: kind, icon: objectIcon }] : []),
  ];
  const current = tabs.find((t) => t.id === active) ?? tabs[0];
  const Icon = current.icon;
  const renamed = active === 'object' ? object : undefined;
  return (
    <aside className="inspector">
      <nav className="inspector-tabs" role="tablist" aria-orientation="vertical">
        {tabs.map(({ id, label, icon: TabIcon }) => (
          <button
            key={id}
            role="tab"
            className="inspector-tab"
            aria-selected={id === active}
            aria-label={label}
            title={label}
            onClick={() => setTab(id)}
          >
            <TabIcon size={16} />
          </button>
        ))}
      </nav>
      <div className="inspector-panel" role="tabpanel" aria-label={current.label}>
        {renamed && (
          <header className="inspector-heading">
            <div className="selection-title">
              <span className="selection-icon">
                <Icon size={16} />
              </span>
              <div>
                <div className="eyebrow">{`${kind} / ${document.name}`}</div>
                <input
                  aria-label={`${kind} name`}
                  className="object-name"
                  key={renamed.id}
                  defaultValue={renamed.name}
                  onBlur={(e) => {
                    if (e.target.value.trim()) patchObject(['name'], e.target.value);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') e.currentTarget.blur();
                  }}
                />
              </div>
            </div>
          </header>
        )}
        <div className="inspector-scroll">
          {active === 'scene' && sceneTab()}
          {active === 'simulation' && simulationTab()}
          {active === 'motion' && groupTab('motion', 'Motion')}
          {active === 'flame' && flameTab()}
          {active === 'smoke' && groupTab('smoke', 'Smoke')}
          {active === 'fuel' && groupTab('fuel', 'Fuel')}
          {active === 'lighting' && groupTab('lighting', 'Lighting')}
          {active === 'rendering' && renderingTab()}
          {active === 'quality' && qualityTab()}
          {active === 'object' && objectTab()}
        </div>
      </div>
    </aside>
  );
}
