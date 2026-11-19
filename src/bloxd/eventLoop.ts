/**
 * Per-tick work budget.
 *
 * Bloxd's api.isNearInterrupt() is not reliable in every runtime, and Date.now()
 * can be frozen inside one tick, so the scheduler charges bytes decoded from
 * block data instead of measuring time. Every ByteCursor read reports its size
 * through spend(); update() stops starting new tasks once the tick's budget is
 * exhausted. A task already running always finishes, so the worst case is
 * budget + the largest single task.
 */

export const TASK_OVERHEAD = 1024;
export const DEFAULT_TICK_BYTE_BUDGET = 8 * 1024 * 1024;

let tickByteBudget = DEFAULT_TICK_BYTE_BUDGET;
let spent = 0;
const queue: (() => void)[] = [];

export const setTickBudget = (bytes: number): void => {
  if (!Number.isFinite(bytes) || bytes < 1) {
    throw new RangeError(`Tick budget must be a positive number: ${bytes}`);
  }
  tickByteBudget = Math.floor(bytes);
};

export const getTickBudget = (): number => tickByteBudget;

export const spend = (bytes: number): void => {
  spent += bytes;
};

export const queueTask = (task: () => void): void => {
  queue.push(task);
};

export const update = (): void => {
  let head = 0;
  try {
    while (head < queue.length && spent < tickByteBudget) {
      const task = queue[head];
      try {
        spent += TASK_OVERHEAD;
        task();
      } finally {
        head++;
      }
    }
  } finally {
    spent = 0;
    if (head > 0) {
      let write = 0;
      for (let read = head; read < queue.length; read++) {
        queue[write++] = queue[read];
      }
      queue.length = write;
    }
  }
};
