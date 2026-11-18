import { waitLoadChunk } from "./loadChunk";

export interface BlockData {
  persisted?: {
    shared?: {
      text?: string;
      uncensoredText?: string;
      textSize?: number;
    };
  };
}

function readText(pos: [number, number, number]): string | undefined {
  try {
    return api.getBlockData(pos[0], pos[1], pos[2])?.persisted?.shared?.text;
  } catch {
    return undefined;
  }
}

export function readData(
  pos: [number, number, number],
  callback: (text: string | undefined) => void,
): void {
  const text = readText(pos);
  if (text !== undefined) {
    callback(text);
    return;
  }
  if (api.isBlockInLoadedChunk(pos[0], pos[1], pos[2])) {
    callback(undefined);
    return;
  }
  waitLoadChunk(pos, () => callback(readText(pos)));
}

export function writeData(
  pos: [number, number, number],
  text: string,
  callback?: () => void,
): void {
  waitLoadChunk(pos, () => {
    const block = api.getBlock(pos[0], pos[1], pos[2]);
    if (block !== "Code Block") {
      api.setBlock(pos[0], pos[1], pos[2], "Code Block");
    }
    api.setBlockData(pos[0], pos[1], pos[2], {
      persisted: {
        shared: {
          text,
          uncensoredText: text,
          textSize: 0,
        },
      },
    });
    callback?.();
  });
}
