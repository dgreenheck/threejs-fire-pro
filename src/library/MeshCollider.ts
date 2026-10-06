import { Matrix4, Mesh, Vector3 } from 'three';
import type { Vec3 } from '../engine/types.ts';

/** Cell-center voxelization for closed, consistently wound rigid triangle meshes. */
export class MeshCollider {
  private readonly geometry;
  private readonly positions;
  private readonly indices;
  private readonly positionVersion: number;
  private readonly indexVersion: number;
  private readonly vertices: Float64Array;
  private cacheKey = '';
  private cached?: Uint32Array<ArrayBuffer>;

  constructor(mesh: Mesh) {
    const geometry = mesh.geometry;
    const position = geometry.getAttribute('position');
    const index = geometry.index;
    if (
      mesh.type === 'SkinnedMesh' ||
      mesh.type === 'InstancedMesh' ||
      Object.keys(geometry.morphAttributes).length
    )
      throw new Error(
        'Mesh colliders require rigid, non-instanced geometry without morph targets.',
      );
    if (
      !position ||
      position.itemSize !== 3 ||
      geometry.drawRange.start !== 0 ||
      Number.isFinite(geometry.drawRange.count)
    )
      throw new Error(
        'Mesh colliders require complete triangle geometry with three-component positions.',
      );
    const count = index?.count ?? position.count;
    if (!count || count % 3)
      throw new Error('Mesh collider geometry must contain complete triangles.');
    this.geometry = geometry;
    this.positions = position;
    this.indices = index;
    this.positionVersion =
      'isInterleavedBufferAttribute' in position ? position.data.version : position.version;
    this.indexVersion = index?.version ?? 0;
    this.vertices = new Float64Array(count * 3);
    const edges = new Map<string, { count: number; winding: number }>();
    const a = new Vector3(),
      b = new Vector3(),
      c = new Vector3();
    let triangles = 0;
    for (let offset = 0; offset < count; offset += 3) {
      const keys: string[] = [];
      for (let corner = 0; corner < 3; corner++) {
        const vertex = index ? index.getX(offset + corner) : offset + corner;
        if (!Number.isInteger(vertex) || vertex < 0 || vertex >= position.count)
          throw new Error('Mesh collider contains an invalid triangle index.');
        const value = [position.getX(vertex), position.getY(vertex), position.getZ(vertex)];
        if (!value.every(Number.isFinite))
          throw new Error('Mesh collider contains nonfinite positions.');
        this.vertices.set(value, (offset + corner) * 3);
        keys.push(value.join(','));
      }
      a.fromArray(this.vertices, offset * 3);
      b.fromArray(this.vertices, (offset + 1) * 3);
      c.fromArray(this.vertices, (offset + 2) * 3);
      if (b.sub(a).cross(c.sub(a)).lengthSq() === 0) continue;
      triangles++;
      for (let edgeIndex = 0; edgeIndex < 3; edgeIndex++) {
        const from = keys[edgeIndex],
          to = keys[(edgeIndex + 1) % 3];
        const forward = from < to;
        const key = forward ? `${from}|${to}` : `${to}|${from}`;
        const edge = edges.get(key) ?? { count: 0, winding: 0 };
        edge.count++;
        edge.winding += forward ? 1 : -1;
        edges.set(key, edge);
      }
    }
    if (!triangles || [...edges.values()].some((edge) => edge.count !== 2 || edge.winding !== 0))
      throw new Error(
        'Mesh colliders require closed, consistently wound manifold surfaces. Weld matching seam positions and close open edges.',
      );
  }

  voxelize(
    mesh: Mesh,
    transform: Matrix4,
    domain: Vec3,
    resolution: Vec3,
  ): Uint32Array<ArrayBuffer> {
    const position = mesh.geometry.getAttribute('position');
    const version =
      position &&
      ('isInterleavedBufferAttribute' in position ? position.data.version : position.version);
    if (
      mesh.geometry !== this.geometry ||
      position !== this.positions ||
      version !== this.positionVersion ||
      mesh.geometry.index !== this.indices ||
      (mesh.geometry.index?.version ?? 0) !== this.indexVersion ||
      mesh.geometry.drawRange.start !== 0 ||
      Number.isFinite(mesh.geometry.drawRange.count)
    )
      throw new Error(
        'Mesh collider geometry changed. Remove and recreate the collider after editing geometry.',
      );
    const key = [...domain, ...resolution, ...transform.elements].join(',');
    if (key === this.cacheKey) return this.cached!;
    const [nx, ny, nz] = resolution;
    const cellSize = domain.map((size, axis) => size / resolution[axis]);
    const columns: { z: number; sign: number }[][] = Array.from({ length: nx * ny }, () => []);
    const vertices = this.vertices.slice(),
      point = new Vector3();
    for (let offset = 0; offset < vertices.length; offset += 3) {
      point.fromArray(vertices, offset).applyMatrix4(transform);
      vertices[offset] = (point.x + domain[0] / 2) / cellSize[0];
      vertices[offset + 1] = point.y / cellSize[1];
      vertices[offset + 2] = (point.z + domain[2] / 2) / cellSize[2];
    }
    const edge = (a: number[], b: number[], x: number, y: number) =>
      (a[0] - x) * (b[1] - y) - (a[1] - y) * (b[0] - x);
    const ownsEdge = (a: number[], b: number[]) => b[1] > a[1] || (b[1] === a[1] && b[0] < a[0]);
    for (let offset = 0; offset < vertices.length; offset += 9) {
      const a = Array.from(vertices.subarray(offset, offset + 3));
      let b = Array.from(vertices.subarray(offset + 3, offset + 6));
      let c = Array.from(vertices.subarray(offset + 6, offset + 9));
      let area = edge(a, b, c[0], c[1]);
      if (area === 0) continue;
      const sign = Math.sign(area);
      if (area < 0) {
        [b, c] = [c, b];
        area = -area;
      }
      const x0 = Math.max(0, Math.ceil(Math.min(a[0], b[0], c[0]) - 0.5));
      const x1 = Math.min(nx - 1, Math.floor(Math.max(a[0], b[0], c[0]) - 0.5));
      const y0 = Math.max(0, Math.ceil(Math.min(a[1], b[1], c[1]) - 0.5));
      const y1 = Math.min(ny - 1, Math.floor(Math.max(a[1], b[1], c[1]) - 0.5));
      for (let y = y0; y <= y1; y++)
        for (let x = x0; x <= x1; x++) {
          const values = [
            edge(b, c, x + 0.5, y + 0.5),
            edge(c, a, x + 0.5, y + 0.5),
            edge(a, b, x + 0.5, y + 0.5),
          ];
          if (
            values.some(
              (value, index) =>
                value < 0 || (value === 0 && !ownsEdge([b, c, a][index], [c, a, b][index])),
            )
          )
            continue;
          columns[y * nx + x].push({
            z: (values[0] * a[2] + values[1] * b[2] + values[2] * c[2]) / area,
            sign,
          });
        }
    }
    const mask = new Uint32Array(nx * ny * nz);
    columns.forEach((crossings, column) => {
      crossings.sort((a, b) => a.z - b.z);
      let winding = 0,
        cursor = 0;
      for (let z = 0; z < nz; z++) {
        while (cursor < crossings.length && crossings[cursor].z <= z + 0.5)
          winding += crossings[cursor++].sign;
        if (winding !== 0) mask[z * nx * ny + column] = 1;
      }
    });
    this.cacheKey = key;
    this.cached = mask;
    return mask;
  }
}
