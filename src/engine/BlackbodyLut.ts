import { DataUtils } from 'three/webgpu';
import { blackbodyTable, BLACKBODY_TABLE_SIZE } from './blackbody.ts';

/** Spectral emission curve: 512 RGBA16F texels, 4 KiB per renderer. */
export function createBlackbodyTexture(device: GPUDevice): GPUTexture {
  const values = blackbodyTable(),
    data = new Uint16Array(values.length);
  for (let i = 0; i < data.length; i++) data[i] = DataUtils.toHalfFloat(values[i]);
  const texture = device.createTexture({
    label: 'Blackbody spectrum (4 KiB)',
    size: [BLACKBODY_TABLE_SIZE, 1],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture }, data, { bytesPerRow: BLACKBODY_TABLE_SIZE * 8 }, [
    BLACKBODY_TABLE_SIZE,
    1,
  ]);
  return texture;
}
