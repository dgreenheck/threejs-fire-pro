import type { PerspectiveCamera, Scene, WebGPURenderer } from 'three/webgpu';
export interface DemoScene {
  objects: Record<string, any>;
  dispose?(): void;
}
export interface SceneContext {
  scene: Scene;
  camera: PerspectiveCamera;
  renderer: WebGPURenderer;
}
