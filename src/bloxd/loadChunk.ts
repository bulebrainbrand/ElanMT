/** Keep a chunk resident until Bloxd has actually loaded it. */

interface PendingLoad {
  pos: [number, number, number];
  callback: () => void;
}

let queue: PendingLoad[] = [];
const ready: (() => void)[] = [];

export const waitLoadChunk = (pos: [number, number, number], callback: () => void): void => {
  if (api.isBlockInLoadedChunk(pos[0], pos[1], pos[2])) {
    callback();
    return;
  }
  queue.push({ pos, callback });
};

export const update = (): void => {
  ready.length = 0;
  let write = 0;
  for (let read = 0; read < queue.length; read++) {
    const entry = queue[read];
    const [x, y, z] = entry.pos;
    if (api.getBlock(x, y, z) !== "Unloaded") {
      ready.push(entry.callback);
    } else {
      queue[write++] = entry;
    }
  }
  queue.length = write;
  for (let i = 0; i < ready.length; i++) ready[i]();
};
