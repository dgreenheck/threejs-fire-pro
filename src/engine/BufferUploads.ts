/** Avoid queue writes for unchanged packed parameters and source records. */
export class BufferUploads {
  private previous = new WeakMap<GPUBuffer, Uint8Array<ArrayBuffer>>();
  constructor(private queue: GPUQueue) {}

  write(buffer: GPUBuffer, data: ArrayBuffer | ArrayBufferView<ArrayBuffer>): boolean {
    const bytes = ArrayBuffer.isView(data)
      ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
      : new Uint8Array(data);
    const previous = this.previous.get(buffer);
    if (previous?.length === bytes.length && bytes.every((value, i) => value === previous[i]))
      return false;
    this.queue.writeBuffer(buffer, 0, bytes);
    this.previous.set(buffer, bytes.slice());
    return true;
  }
}
