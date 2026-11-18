/**
 * UTF-8 helpers for runtimes without TextEncoder/TextDecoder (Bloxd's world
 * code VM does not provide them). The implementation follows the public
 * domain fast-text-encoding routines used by Brandon-Tiny-10M-Instruct.
 */

export function encodeUtf8(text: string): number[] {
  const out: number[] = [];
  for (let i = 0; i < text.length; i++) {
    let code = text.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      if (i + 1 < text.length) {
        const low = text.charCodeAt(i + 1);
        if ((low & 0xfc00) === 0xdc00) {
          i++;
          code = ((code & 0x3ff) << 10) + (low & 0x3ff) + 0x10000;
        }
      }
      if (code >= 0xd800 && code <= 0xdbff) continue; // lone surrogate
    }

    if (code < 0x80) {
      out.push(code);
    } else if (code < 0x800) {
      out.push(0xc0 | (code >> 6), 0x80 | (code & 0x3f));
    } else if (code < 0x10000) {
      out.push(0xe0 | (code >> 12), 0x80 | ((code >> 6) & 0x3f), 0x80 | (code & 0x3f));
    } else if (code < 0x200000) {
      out.push(
        0xf0 | (code >> 18),
        0x80 | ((code >> 12) & 0x3f),
        0x80 | ((code >> 6) & 0x3f),
        0x80 | (code & 0x3f),
      );
    }
  }
  return out;
}

export function decodeUtf8(bytes: ArrayLike<number>): string {
  const codeUnits: number[] = [];
  let i = 0;
  while (i < bytes.length) {
    const b1 = bytes[i++] & 0xff;
    if (b1 < 0x80) {
      codeUnits.push(b1);
    } else if ((b1 & 0xe0) === 0xc0) {
      const b2 = (bytes[i++] ?? 0) & 0x3f;
      codeUnits.push(((b1 & 0x1f) << 6) | b2);
    } else if ((b1 & 0xf0) === 0xe0) {
      const b2 = (bytes[i++] ?? 0) & 0x3f;
      const b3 = (bytes[i++] ?? 0) & 0x3f;
      codeUnits.push(((b1 & 0x0f) << 12) | (b2 << 6) | b3);
    } else if ((b1 & 0xf8) === 0xf0) {
      const b2 = (bytes[i++] ?? 0) & 0x3f;
      const b3 = (bytes[i++] ?? 0) & 0x3f;
      const b4 = (bytes[i++] ?? 0) & 0x3f;
      let code = ((b1 & 0x07) << 18) | (b2 << 12) | (b3 << 6) | b4;
      if (code > 0xffff) {
        code -= 0x10000;
        codeUnits.push(0xd800 | (code >> 10));
        code = 0xdc00 | (code & 0x3ff);
      }
      codeUnits.push(code & 0xffff);
    }
  }

  let out = "";
  const CHUNK = 4096;
  for (let start = 0; start < codeUnits.length; start += CHUNK) {
    const end = Math.min(start + CHUNK, codeUnits.length);
    const chunk: number[] = [];
    for (let j = start; j < end; j++) chunk.push(codeUnits[j]);
    out += String.fromCharCode.apply(null, chunk);
  }
  return out;
}
