import {
  CatmullRomCurve3,
  CylinderGeometry,
  Group,
  LatheGeometry,
  TubeGeometry,
  Vector2,
  Vector3,
} from 'three/webgpu';
import { box, cylinder, mesh } from './primitives';
import { sceneAssets } from './props';
import type { Scene } from 'three/webgpu';

/** Meter-scale swivel head; local +X is the bore and the muzzle is the origin. */
export function createNozzle(scene: Scene) {
  const assets = sceneAssets(scene);
  box(scene, [0.95, 0.16, 0.95], [-3.85, 0.08, 0], '#697079', 1);
  cylinder(scene, 0.12, 0.85, [-3.85, 0.54, 0], '#899097', 1);
  const nozzle = new Group();
  nozzle.position.set(-3.85, 1.05, 0);
  scene.add(nozzle);
  const profile = [
    [0.11, -0.5],
    [0.16, -0.5],
    [0.18, -0.4],
    [0.15, -0.32],
    [0.15, 0.25],
    [0.19, 0.3],
    [0.19, 0.38],
    [0.12, 0.38],
    [0.11, -0.5],
  ];
  const barrel = mesh(
    nozzle,
    new LatheGeometry(
      profile.map(([r, y]) => new Vector2(r, y)),
      64,
    ),
    assets.surface('metal'),
  );
  barrel.rotation.z = -Math.PI / 2;
  for (const x of [-0.3, 0.2]) {
    const collar = cylinder(nozzle, 0.185, 0.055, [x, 0, 0], '#8a9299', 1);
    collar.rotation.z = Math.PI / 2;
    for (let i = 0; i < 8; i++) {
      const a = (i * Math.PI) / 4;
      const bolt = mesh(
        nozzle,
        new CylinderGeometry(0.025, 0.025, 0.035, 6),
        assets.surface('metal', '#6a7075'),
        [x, Math.cos(a) * 0.184, Math.sin(a) * 0.184],
      );
      bolt.rotation.x = a;
    }
  }
  const bore = mesh(
    nozzle,
    new CylinderGeometry(0.105, 0.105, 0.04, 48),
    assets.surface('metal', '#141414'),
    [0.22, 0, 0],
  );
  bore.rotation.z = Math.PI / 2;
  const hoseCurve = new CatmullRomCurve3([
    new Vector3(-4.3, 0.9, 0),
    new Vector3(-4.7, 0.55, 0),
    new Vector3(-4.6, 0.12, 0.6),
    new Vector3(-3.7, 0.1, 1),
  ]);
  mesh(
    scene,
    new TubeGeometry(hoseCurve, 48, 0.055, 12, false),
    assets.surface('metal', '#24282b'),
  );
  const muzzle = new Group();
  muzzle.position.x = 0.39;
  nozzle.add(muzzle);
  return { nozzle, muzzle };
}
