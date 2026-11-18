/** Block-data layout for the ElanMT world. */

export const CHUNK_SIZE = 16000;

/** Weight chunks live on the z=32 layer. */
export const WEIGHT_ORIGIN = [0, 1, 32] as const;

export const CODE_BLOCK_ID = 1510;
export const CODE_BLOCK_CHARS = CHUNK_SIZE;
export const BLOCKS_PER_SPATIAL_CHUNK = 8;
export const SLOT_SPACING = 4;
