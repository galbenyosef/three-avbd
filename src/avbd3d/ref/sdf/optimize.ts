/* Copyright (c) 2026 Chris Giles
 * Permission to use, copy, modify, distribute and sell this software and its
 * documentation for any purpose is hereby granted without fee, provided that
 * the above copyright notice appear in all copies. Chris Giles makes no
 * representations about the suitability of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */
import { add, sub, mul, sqrt, plus, minus, scale, divide, length, project, type Vec, type Aabb } from './math.ts';
import type { SdfBody } from './body.ts';
export const softmax = (a: number, b: number, epsilon: number) => mul(0.5, add(add(a, b), sqrt(add(mul(sub(a, b), sub(a, b)), epsilon))));
export interface Counters { pairs: number; contacts: number; queries: number; iterations: number }
export const counters = (): Counters => ({ pairs: 0, contacts: 0, queries: 0, iterations: 0 });
/** Eq. 3: softmax distance minus repulsion from the current manifold. */
export function objective(a: SdfBody, b: SdfBody, points: { x: Vec }[], epsilon: number, alpha: number, stats: Counters): (x: Vec) => number {
  return x => {
    stats.queries++;
    let value = softmax(a.sample(x).phi, b.sample(x).phi, epsilon);
    for (const p of points) value = sub(value, mul(alpha, length(minus(x, p.x))));
    return value;
  };
}
/** Sec. 3.3: upstream domain-clamped Nelder–Mead; stable insertion order for ties. */
export function nelderMead(g: (x: Vec) => number, seed: Vec, domain: Aabb, initialStep: Vec, tolerance: number, limit: number): { x: Vec; iterations: number } {
  const x = project(seed, domain), v: Vec[] = [x, [...x], [...x], [...x]];
  for (let i = 0; i < 3; i++) {
    v[i + 1][i] = add(v[i + 1][i], add(x[i], initialStep[i]) <= domain.max[i] ? initialStep[i] : -initialStep[i]);
    v[i + 1] = project(v[i + 1], domain);
  }
  const values = v.map(g); let it = 0;
  for (; it < limit; it++) {
    for (let i = 1; i < 4; i++) for (let j = i; j > 0 && values[j] < values[j - 1]; j--) {
      [values[j], values[j - 1]] = [values[j - 1], values[j]]; [v[j], v[j - 1]] = [v[j - 1], v[j]];
    }
    const extent = Math.max(length(minus(v[1], v[0])), length(minus(v[2], v[0])), length(minus(v[3], v[0])));
    if (extent < tolerance || sub(values[3], values[0]) < mul(0.05, tolerance)) break;
    const centroid = divide(plus(plus(v[0], v[1]), v[2]), 3);
    const reflected = project(plus(centroid, minus(centroid, v[3])), domain), fr = g(reflected);
    if (fr < values[0]) {
      const expanded = project(plus(centroid, scale(minus(centroid, v[3]), 2)), domain), fe = g(expanded);
      v[3] = fe < fr ? expanded : reflected; values[3] = Math.min(fe, fr);
    } else if (fr < values[2]) { v[3] = reflected; values[3] = fr; }
    else {
      const contracted = plus(centroid, scale(minus(fr < values[3] ? reflected : v[3], centroid), 0.5)), fc = g(contracted);
      if (fc < Math.min(fr, values[3])) { v[3] = contracted; values[3] = fc; }
      else for (let i = 1; i < 4; i++) { v[i] = plus(v[0], scale(minus(v[i], v[0]), 0.5)); values[i] = g(v[i]); }
    }
  }
  let best = 0; for (let i = 1; i < 4; i++) if (values[i] < values[best]) best = i;
  return { x: v[best], iterations: it };
}
