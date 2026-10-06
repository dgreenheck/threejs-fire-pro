import {
  Mesh,
  Vector3,
  type BufferAttribute,
  type BufferGeometry,
  type InterleavedBufferAttribute,
} from 'three';

export interface MeshSurfaceSample {
  position: Vector3;
  weight: number;
}

/** Deterministic, triangle-area-weighted samples of rigid mesh geometry. */
export class MeshSurface {
  readonly samples: MeshSurfaceSample[];
  readonly surfaceArea: number;
  private readonly geometry: BufferGeometry;
  private readonly positions: BufferAttribute | InterleavedBufferAttribute;
  private readonly positionVersion: number;
  private readonly indices: BufferAttribute | null;
  private readonly indexVersion: number;

  constructor(mesh: Mesh, cellSize: number, seed: number) {
    const geometry = mesh.geometry;
    const position = geometry.getAttribute('position');
    if (
      mesh.type === 'SkinnedMesh' ||
      mesh.type === 'InstancedMesh' ||
      Object.keys(geometry.morphAttributes).length
    ) {
      throw new Error('Mesh emitters require a rigid, non-instanced mesh without morph targets.');
    }
    if (!position || position.itemSize !== 3)
      throw new Error('Mesh emitter geometry needs a three-component position attribute.');
    this.geometry = geometry;
    this.positions = position;
    this.positionVersion =
      'isInterleavedBufferAttribute' in position ? position.data.version : position.version;
    this.indices = geometry.getIndex();
    this.indexVersion = this.indices?.version ?? 0;
    const count = this.indices?.count ?? position.count;
    if (count % 3 !== 0) throw new Error('Mesh emitter geometry must contain complete triangles.');
    if (geometry.drawRange.start !== 0 || Number.isFinite(geometry.drawRange.count))
      throw new Error('Mesh emitters require complete geometry with no active draw range.');

    const cumulativeArea = new Float64Array(count / 3);
    const a = new Vector3(),
      b = new Vector3(),
      c = new Vector3();
    const normal = new Vector3(),
      edge = new Vector3();
    const readTriangle = (triangle: number) => {
      const offset = triangle * 3;
      a.fromBufferAttribute(position, this.indices ? this.indices.getX(offset) : offset);
      b.fromBufferAttribute(position, this.indices ? this.indices.getX(offset + 1) : offset + 1);
      c.fromBufferAttribute(position, this.indices ? this.indices.getX(offset + 2) : offset + 2);
      normal.subVectors(b, a).cross(edge.subVectors(c, a));
    };
    const finite = (value: Vector3) =>
      Number.isFinite(value.x) && Number.isFinite(value.y) && Number.isFinite(value.z);
    let area = 0;
    for (let triangle = 0; triangle < cumulativeArea.length; triangle++) {
      readTriangle(triangle);
      if (!finite(a) || !finite(b) || !finite(c))
        throw new Error('Mesh emitter geometry contains nonfinite positions.');
      const triangleArea = normal.length() * 0.5;
      if (!Number.isFinite(triangleArea))
        throw new Error('Mesh emitter triangle area exceeds the supported numeric range.');
      if (triangleArea > 1e-12) area += triangleArea;
      cumulativeArea[triangle] = area;
    }
    if (!(area > 0)) throw new Error('Mesh emitter geometry has no nondegenerate triangles.');
    this.surfaceArea = area;
    const sampleCount = Math.min(8192, Math.max(256, Math.ceil((area / cellSize ** 2) * 8)));
    let randomState = seed >>> 0;
    const random = () => {
      randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
      return randomState / 4294967296;
    };
    this.samples = [];
    let triangleIndex = 0;
    for (let index = 0; index < sampleCount; index++) {
      const target = ((index + 0.5) / sampleCount) * area;
      while (cumulativeArea[triangleIndex] < target) triangleIndex++;
      readTriangle(triangleIndex);
      normal.normalize();
      const root = Math.sqrt(random()),
        along = random();
      const point = a
        .clone()
        .multiplyScalar(1 - root)
        .addScaledVector(b, root * (1 - along))
        .addScaledVector(c, root * along)
        .addScaledVector(normal, cellSize * 0.75);
      this.samples.push({ position: point, weight: 1 / sampleCount });
    }
  }

  assertUnchanged(mesh: Mesh): void {
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
    ) {
      throw new Error(
        'Mesh emitter geometry changed. Recreate the emitter after editing geometry.',
      );
    }
  }
}
