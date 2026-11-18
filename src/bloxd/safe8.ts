/**
 * safe8: three bits per character, using only '0'..'7'. Eight characters carry
 * exactly three bytes, so encoded text contains no characters that can break
 * Bloxd's block-data serialization.
 */

const CODES: number[] = [48, 49, 50, 51, 52, 53, 54, 55];

export function encode(bytes: ArrayLike<number>): string {
  const outLen = Math.ceil((bytes.length * 8) / 3);
  const out = new Array<number>(outLen).fill(0);
  let idx = 0;
  let buffer = 0;
  let bits = 0;

  for (let i = 0; i < bytes.length; i++) {
    buffer = (buffer << 8) | (bytes[i] & 0xff);
    bits += 8;
    while (bits >= 3) {
      bits -= 3;
      out[idx++] = CODES[(buffer >> bits) & 0x07];
    }
    buffer &= (1 << bits) - 1;
  }
  if (bits > 0) {
    out[idx++] = CODES[(buffer << (3 - bits)) & 0x07];
  }
  return String.fromCharCode.apply(null, out);
}

export function decode(text: string): number[] {
  const bytes: number[] = [];
  let buffer = 0;
  let bits = 0;
  for (let i = 0; i < text.length; i++) {
    const value = text.charCodeAt(i) - 48;
    if (value < 0 || value > 7) {
      throw new Error(`Invalid safe8 character code: ${text.charCodeAt(i)}`);
    }
    buffer = (buffer << 3) | value;
    bits += 3;
    while (bits >= 8) {
      bits -= 8;
      bytes.push((buffer >> bits) & 0xff);
    }
    buffer &= (1 << bits) - 1;
  }
  return bytes;
}
