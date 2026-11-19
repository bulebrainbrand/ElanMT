export interface BloxdBlockData {
  persisted?: {
    shared?: {
      text?: string;
      uncensoredText?: string;
      textSize?: number;
    };
  };
}

export interface BloxdApi {
  isBlockInLoadedChunk(x: number, y: number, z: number): boolean;
  getBlockData(x: number, y: number, z: number): BloxdBlockData | undefined;
  getBlock(x: number, y: number, z: number): string | undefined;
  setBlock(x: number, y: number, z: number, block: string): void;
  setBlockData(x: number, y: number, z: number, data: BloxdBlockData): void;
  getPosition?(playerId: string): [number, number, number];
  getPlayerIds?(): string[];
  setClientOption?(playerId: string, key: string, value: unknown): void;
}

declare global {
  // Bloxd injects these globals into world code.
  var api: BloxdApi;
  var tick: (() => void) | undefined;
  var onPlayerClick: ((playerId: string) => void) | undefined;
  var think:
    | ((
        text: string,
        onComplete?: (tokens: number[], promptLength: number) => void,
        stream?: boolean,
        maxNewTokens?: number,
      ) => void)
    | undefined;
  var setTickBudget: ((bytes: number) => void) | undefined;
  var setLoadBudget: ((bytes: number) => void) | undefined;
}

export {};
