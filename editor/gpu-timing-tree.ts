import type { GpuTimingCategory, GpuTimingRow } from '../src/engine/GpuProfiler.ts';

export interface GpuTimingNode {
  id: string;
  label: string;
  category: GpuTimingCategory | 'all';
  kernel?: string;
  children?: GpuTimingNode[];
  milliseconds: number | null;
  calls: number;
}

/** Display order follows stages, never measured duration. Leaves are exclusive GPU costs. */
export function createGpuTimingTree(): GpuTimingNode {
  const leaf = (category: GpuTimingCategory, kernel: string, label = kernel): GpuTimingNode => ({
    id: `${category}:${kernel}`,
    label,
    category,
    kernel,
    milliseconds: null,
    calls: 0,
  });
  const group = (
    id: string,
    label: string,
    category: GpuTimingNode['category'],
    children: GpuTimingNode[],
  ): GpuTimingNode => ({ id, label, category, children, milliseconds: null, calls: 0 });
  const kernels = (category: GpuTimingCategory, names: string[]) =>
    names.map((name) => leaf(category, name));
  return group('all', 'Total GPU', 'all', [
    group('simulation', 'Simulation', 'simulation', [
      group(
        'bricks',
        'Active bricks',
        'simulation',
        kernels('simulation', [
          'markContent',
          'buildBricks',
          'freeBricks',
          'allocateBricks',
          'linkBricks',
          'buildSolids',
          'finishBricks',
          'zeroVelocityBricks',
          'zeroFieldBricks',
          'zeroSmokeBricks',
        ]),
      ),
      group(
        'transport',
        'Transport & sources',
        'simulation',
        kernels('simulation', [
          'advectVelocityWithBounds',
          'correctVelocity',
          'advectSmoke',
          'advectScalars',
          'evolveFields',
          'evolveSmoke',
        ]),
      ),
      group(
        'forces',
        'Forces & divergence',
        'simulation',
        kernels('simulation', ['computeCurl', 'applyForces', 'computeDivergence']),
      ),
      group(
        'pressure',
        'Pressure solve',
        'simulation',
        kernels('simulation', [
          'Multigrid initializePressure',
          'Multigrid buildBrickTopology',
          'Multigrid buildTileTopology',
          'Multigrid relaxBricks',
          'Multigrid relaxCoarseBricks',
          'Multigrid relaxTiles',
          'Multigrid relaxCoarse',
          'Multigrid restrictBricks',
          'Multigrid restrictTiles',
          'Multigrid prolongateBricks',
          'Multigrid prolongateCoarseBricks',
          'Multigrid prolongateTiles',
        ]),
      ),
      group('projection', 'Projection', 'simulation', kernels('simulation', ['project'])),
    ]),
    group('lighting', 'Lighting compute', 'lighting', [
      group(
        'volume-lighting',
        'Volume caches',
        'lighting',
        kernels('lighting', [
          'exportPages',
          'smoothSmoke',
          'smoothCoarseSmoke',
          'buildOccupancy',
          'buildLighting',
          'lightFresh',
        ]),
      ),
      group(
        'half-resolution',
        'Half-resolution march',
        'lighting',
        kernels('lighting', ['buildSlices']),
      ),
      group(
        'scene-lighting',
        'Scene illumination',
        'lighting',
        kernels('lighting', ['collectLights']),
      ),
    ]),
    group('render', 'Rendering', 'render', [leaf('render', 'Scene + fire', 'Scene + fire')]),
  ]);
}

/** Aggregate bottom-up exactly once. Absent kernels keep their place with zero cost. */
export function updateGpuTimingTree(root: GpuTimingNode, rows: Iterable<GpuTimingRow>): void {
  const leaves = new Map<string, GpuTimingNode>();
  const reset = (node: GpuTimingNode) => {
    node.milliseconds = 0;
    node.calls = 0;
    if (node.children) node.children.forEach(reset);
    else leaves.set(node.id, node);
  };
  reset(root);
  for (const row of rows) {
    const key = `${row.category}:${row.label}`;
    let node = leaves.get(key);
    if (!node) {
      // Application-authored extra passes append once in first-seen order.
      const category = root.children!.find((group) => group.id === row.category)!;
      let other = category.children!.find((group) => group.id === `${row.category}:other`);
      if (!other) {
        other = {
          id: `${row.category}:other`,
          label: 'Other passes',
          category: row.category,
          children: [],
          milliseconds: 0,
          calls: 0,
        };
        category.children!.push(other);
      }
      node = {
        id: key,
        label: row.label,
        kernel: row.label,
        category: row.category,
        milliseconds: 0,
        calls: 0,
      };
      other.children!.push(node);
      leaves.set(key, node);
    }
    node.milliseconds! += row.milliseconds;
    node.calls += row.calls;
  }
  const sum = (node: GpuTimingNode): void => {
    if (!node.children) return;
    for (const child of node.children) {
      sum(child);
      node.milliseconds! += child.milliseconds!;
      node.calls += child.calls;
    }
  };
  sum(root);
}
