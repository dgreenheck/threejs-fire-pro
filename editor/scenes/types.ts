import type { PerspectiveCamera, Scene, WebGPURenderer } from 'three/webgpu';
import type { DemoGround } from './primitives';
export interface DemoScene {
  objects: Record<string, any>;
  ground: DemoGround;
  dispose?(): void;
}
export interface SceneContext {
  scene: Scene;
  camera: PerspectiveCamera;
  renderer: WebGPURenderer;
}
