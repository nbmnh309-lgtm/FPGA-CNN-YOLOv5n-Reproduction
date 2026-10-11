"""Integer oracles. HWC layout; nearest-even affine, half-away Q16 RTL."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Domain:
    scale: float
    zero: int
    lo: int = 0
    hi: int = 255

    def __post_init__(self):
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Scale must be finite and positive')
        if not self.lo <= self.zero <= self.hi:
            raise ValueError('Zero outside integer domain')

    def quantize(self, x):
        return np.clip(np.rint(np.asarray(x) / self.scale) + self.zero, self.lo, self.hi).astype(np.int64)

    def dequantize(self, q):
        return (np.asarray(q, dtype=np.int64) - self.zero) * self.scale


def calibrate(x, signed=False):
    x = np.asarray(x)
    if not np.all(np.isfinite(x)) or x.size == 0:
        raise ValueError('Calibration needs finite nonempty data')
    if signed:
        return Domain(max(float(np.abs(x).max()) / 127, 1e-12), 0, -128, 127)
    low, high = min(float(x.min()), 0), max(float(x.max()), 0)
    scale = max((high - low) / 255, 1e-12)
    return Domain(scale, int(np.clip(np.rint(-low / scale), 0, 255)))


def mqda_add(a, b, da, db, out, clamp_negative=False):
    aa, bb = da.dequantize(a), db.dequantize(b)
    if clamp_negative:
        aa, bb = np.maximum(aa, 0), np.maximum(bb, 0)
    return out.quantize(aa + bb)


def mqda_cat(a, b, da, db, out, axis=-1):
    return np.concatenate((out.quantize(da.dequantize(a)), out.quantize(db.dequantize(b))), axis=axis)


def naive_add(a, b, out):
    return np.clip(np.asarray(a, dtype=np.int64) + b - out.zero, out.lo, out.hi)


def requant_q16(value, multiplier, zero=0):
    """Round half away from zero, then add Z and saturate unsigned 8-bit."""
    p = np.asarray(value, dtype=np.int64) * int(multiplier)
    q = np.where(p >= 0, (p + 32768) >> 16, -((-p + 32768) >> 16))
    return np.clip(q + zero, 0, 255)


def dmp(a, b, c):
    """Signed 8-bit weights, signed/unsigned shared activation.

    A25 = a*2^16+b, B9 = c. Low signed lane borrows into high lane
    if b*c<0, so high extraction needs +1. Fits DSP48E1 25x18.
    """
    a, b, c = (np.asarray(v, dtype=np.int64) for v in (a, b, c))
    if np.any((a < -128) | (a > 127) | (b < -128) | (b > 127)) or np.any((c < -128) | (c > 255)):
        raise ValueError('DMP operand outside supported range')
    packed = (a << 16) + b
    product = packed * c
    low = ((product + 32768) & 65535) - 32768
    high = (product >> 16) + (low < 0)
    return high, low


def conv(x, weights, bias=None, stride=1, padding=0, zero=0):
    """HWC activations, OIHW signed weights; exact int64 accumulator oracle."""
    x, w = np.asarray(x, dtype=np.int64), np.asarray(weights, dtype=np.int64)
    co, ci, kh, kw = w.shape
    if ci != x.shape[-1] or stride < 1:
        raise ValueError('Convolution shape/stride mismatch')
    xp = np.pad(x - zero, ((padding, padding), (padding, padding), (0, 0)))
    oh, ow = (xp.shape[0] - kh) // stride + 1, (xp.shape[1] - kw) // stride + 1
    y = np.empty((oh, ow, co), dtype=np.int64)
    for h in range(oh):
        for col in range(ow):
            window = xp[h*stride:h*stride+kh, col*stride:col*stride+kw].transpose(2, 0, 1)
            y[h, col] = (window[None] * w).sum(axis=(1, 2, 3))
    if bias is not None:
        y += np.asarray(bias)
    return y


def focus(x):
    if x.shape[0] % 2 or x.shape[1] % 2:
        raise ValueError('Focus requires even H/W')
    return np.concatenate((x[::2, ::2], x[1::2, ::2], x[::2, 1::2], x[1::2, 1::2]), axis=-1)


def upsample(x):
    return x.repeat(2, axis=0).repeat(2, axis=1)


def maxpool5(x):
    p = np.pad(x.astype(np.int64), ((2, 2), (2, 2), (0, 0)), constant_values=np.iinfo(np.int64).min)
    return np.maximum.reduce([p[h:h+x.shape[0], w:w+x.shape[1]] for h in range(5) for w in range(5)])


def sppf(x):
    a = maxpool5(x); b = maxpool5(a); c = maxpool5(b)
    return np.concatenate((x, a, b, c), axis=-1)


def buffer_schedule(blocks, read_cycles, compute_cycles, write_cycles, overlap=True):
    """Discrete two-bank ownership model with independent read/compute/write engines.

    Each bank remains occupied until write completes. Latencies are assumed,
    never FPGA measurements. Return timeline for conflict checks.
    """
    if blocks < 0 or min(read_cycles, compute_cycles, write_cycles) < 1:
        raise ValueError('Invalid schedule')
    bank_free = [0, 0]; engines = [0, 0, 0]; events = []
    for i in range(blocks):
        bank = i % 2
        start = max(bank_free[bank], engines[0]) if overlap else max(engines)
        r = (start, start + read_cycles)
        c = (max(r[1], engines[1]), max(r[1], engines[1]) + compute_cycles)
        w = (max(c[1], engines[2]), max(c[1], engines[2]) + write_cycles)
        engines = [r[1], c[1], w[1]]; bank_free[bank] = w[1]
        events.append({'block': i, 'bank': bank, 'read': r, 'compute': c, 'write': w})
    return {'cycles': max(engines), 'events': events, 'result_kind': 'SIMULATED'}


class MemoryNPU:
    """Instruction-level cache-compute-cache model, no physical PCIe/DDR timing."""
    def __init__(self):
        self.memory = {}; self.transfers = []

    def run(self, instructions):
        for inst in instructions:
            sources = [self.memory[a].copy() for a in inst['src']]
            self.transfers.extend(('read', a, int(x.nbytes)) for a, x in zip(inst['src'], sources))
            op, args = inst['op'], inst.get('args', {})
            if op in ('conv33', 'conv11'):
                result = conv(sources[0], **args)
            elif op == 'add':
                result = mqda_add(*sources, **args)
            elif op == 'cat':
                result = mqda_cat(*sources, **args)
            else:
                result = {'focus': focus, 'upsample': upsample, 'sppf': sppf}[op](sources[0])
            self.memory[inst['dst']] = result
            self.transfers.append(('write', inst['dst'], int(result.nbytes)))
        return self.memory[instructions[-1]['dst']]
