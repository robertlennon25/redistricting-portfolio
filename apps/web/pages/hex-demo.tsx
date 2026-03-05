import { useEffect, useMemo, useRef, useState } from "react";

/**
 * Fairness demo: Packing vs Cracking (contiguous districts)
 * - 91 hexes (radius=5 axial grid)
 * - 7 districts
 * - Left: vote distribution (Dem share gradient) + bold district outlines as they are drawn
 * - Right: final districts colored by winner (solid red/blue) + seat counts
 *
 * No external libs. Pure SVG. Contiguity guaranteed by region-growing on hex adjacency.
 */

type Mode = "pack" | "crack";

type HexCell = {
  id: string;
  q: number;
  r: number;
  total: number; // equal-pop
  dem: number;
  rep: number;
  demShare: number; // 0..1
};

type Frame = {
  assign: Record<string, number>; // cellId -> districtId (0..6), partial during animation
};

type TooltipState = {
  visible: boolean;
  x: number;
  y: number;
  content: React.ReactNode;
};

const RADIUS = 5; // radius-5 hex board => 91 cells
const NUM_DISTRICTS = 7;
const TOTAL_PER_HEX = 100;

const HEX_SIZE = 16;
const PANEL_PAD = 22;

const sqrt3 = Math.sqrt(3);

function clamp(x: number, a: number, b: number): number {
  return Math.max(a, Math.min(b, x));
}
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
function rgb(r: number, g: number, b: number): string {
  return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
}

/** Axial hex distance */
function hexDist(q: number, r: number): number {
  const x = q;
  const z = r;
  const y = -x - z;
  return Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
}

/** Axial neighbors */
function neighbors(q: number, r: number): Array<[number, number]> {
  return [
    [q + 1, r],
    [q - 1, r],
    [q, r + 1],
    [q, r - 1],
    [q + 1, r - 1],
    [q - 1, r + 1],
  ];
}

/** Generate radius board: all (q,r) with dist<=R */
function generateHexBoard(radius: number): Array<{ q: number; r: number }> {
  const coords: Array<{ q: number; r: number }> = [];
  for (let q = -radius; q <= radius; q++) {
    for (let r = -radius; r <= radius; r++) {
      if (hexDist(q, r) <= radius) coords.push({ q, r });
    }
  }
  coords.sort((a, b) => (a.r - b.r) || (a.q - b.q));
  return coords;
}

/** Axial -> pixel (pointy-top) */
function axialToPixel(q: number, r: number, size: number): { x: number; y: number } {
  return {
    x: size * sqrt3 * (q + r / 2),
    y: size * 1.5 * r,
  };
}

/** Hex polygon points for SVG */
function hexPoints(cx: number, cy: number, size: number): string {
  const pts: Array<[number, number]> = [];
  for (let k = 0; k < 6; k++) {
    const ang = (Math.PI / 180) * (30 + 60 * k);
    pts.push([cx + size * Math.cos(ang), cy + size * Math.sin(ang)]);
  }
  return pts.map(([x, y]) => `${x},${y}`).join(" ");
}

/** Hex corners as coordinate array (for boundary edges) */
function hexCorners(cx: number, cy: number, size: number): Array<[number, number]> {
  const pts: Array<[number, number]> = [];
  for (let k = 0; k < 6; k++) {
    const ang = (Math.PI / 180) * (30 + 60 * k);
    pts.push([cx + size * Math.cos(ang), cy + size * Math.sin(ang)]);
  }
  return pts;
}

/**
 * Dem share -> red/blue gradient.
 * 0 => red, 0.5 => light neutral, 1 => blue.
 */
function shareToRedBlue(share: number): string {
  const s = clamp(share, 0, 1);
  const red = { r: 220, g: 60, b: 60 };
  const mid = { r: 245, g: 245, b: 245 };
  const blue = { r: 60, g: 110, b: 220 };

  if (s <= 0.5) {
    const t = s / 0.5;
    return rgb(lerp(red.r, mid.r, t), lerp(red.g, mid.g, t), lerp(red.b, mid.b, t));
  } else {
    const t = (s - 0.5) / 0.5;
    return rgb(lerp(mid.r, blue.r, t), lerp(mid.g, blue.g, t), lerp(mid.b, blue.b, t));
  }
}

function solidPartyColor(p: "dem" | "rep"): string {
  return p === "dem" ? "rgb(60,110,220)" : "rgb(220,60,60)";
}
function winnerColor(demShare: number): "dem" | "rep" {
  return demShare > 0.5 ? "dem" : "rep";
}

/**
 * Build a partisan landscape:
 * - Central "city" blob is Dem-heavy
 * - Outer becomes Rep-leaning
 *
 * Tuned so packing typically yields ~2 Dem seats, cracking typically yields ~1 Dem seat.
 */
function buildVoteGrid(coords: Array<{ q: number; r: number }>): HexCell[] {
  // city center offset
  const cityQ = -1;
  const cityR = 1;

  // tighten the blob to make packing/cracking visually strong
  const sigma = 2.0;

  return coords.map(({ q, r }) => {
    const d = hexDist(q - cityQ, r - cityR);
    const bump = Math.exp(-(d * d) / (2 * sigma * sigma)); // 0..1

    // Base is slightly Rep-leaning; city bump makes strong Dem core
    let share = 0.28 + 0.60 * bump;

    // gentle gradient to avoid perfect symmetry
    share += 0.035 * (q / RADIUS) - 0.03 * (r / RADIUS);

    share = clamp(share, 0.10, 0.90);

    const dem = Math.round(TOTAL_PER_HEX * share);
    const rep = TOTAL_PER_HEX - dem;

    return {
      id: `${q},${r}`,
      q,
      r,
      total: TOTAL_PER_HEX,
      dem,
      rep,
      demShare: dem / TOTAL_PER_HEX,
    };
  });
}

/** Compute district totals from a (partial) assignment */
function computeDistrictTotals(cells: HexCell[], assign: Record<string, number>) {
  const totals = Array.from({ length: NUM_DISTRICTS }, () => ({
    dem: 0,
    rep: 0,
    total: 0,
    assignedCells: 0,
  }));

  const cellById = new Map(cells.map((c) => [c.id, c] as const));
  for (const [cellId, d] of Object.entries(assign)) {
    const cell = cellById.get(cellId);
    if (!cell) continue;
    totals[d].dem += cell.dem;
    totals[d].rep += cell.rep;
    totals[d].total += cell.total;
    totals[d].assignedCells += 1;
  }

  const winners = totals.map((t) => {
    const share = t.total > 0 ? t.dem / t.total : 0.5;
    return winnerColor(share);
  });

  const seats = winners.reduce(
    (acc, w) => {
      if (w === "dem") acc.dem += 1;
      else acc.rep += 1;
      return acc;
    },
    { dem: 0, rep: 0 }
  );

  return { totals, winners, seats };
}

/**
 * Contiguous district generator via region-growing.
 * - Seeds chosen deterministically
 * - Growth only via frontier neighbors => contiguity guaranteed
 * - Mode-specific scoring drives "pack" vs "crack" behavior
 */
function buildContiguousPlan(cells: HexCell[], mode: Mode) {
  const targetSize = Math.floor(cells.length / NUM_DISTRICTS); // 13 for 91/7
  const cellById = new Map(cells.map((c) => [c.id, c] as const));
  const allIds = cells.map((c) => c.id);

  const exists = new Set(allIds);

  // adjacency map: id -> neighbor ids (only those on board)
  const adj = new Map<string, string[]>();
  for (const c of cells) {
    const ns: string[] = [];
    for (const [nq, nr] of neighbors(c.q, c.r)) {
      const nid = `${nq},${nr}`;
      if (exists.has(nid)) ns.push(nid);
    }
    adj.set(c.id, ns);
  }

  // Pick seeds
  // City-ish: closest to (-1,1)
  const cityQ = -1;
  const cityR = 1;

  function distToCity(c: HexCell) {
    return hexDist(c.q - cityQ, c.r - cityR);
  }

  // Greedy farthest-point sampling for well-spread seeds
  function pickFarthestSeeds(initial: HexCell[], k: number): HexCell[] {
    const picked: HexCell[] = [...initial];
    while (picked.length < k) {
      let best: HexCell | null = null;
      let bestScore = -Infinity;
      for (const c of cells) {
        if (picked.includes(c)) continue;
        let minD = Infinity;
        for (const p of picked) {
          const d = hexDist(c.q - p.q, c.r - p.r);
          minD = Math.min(minD, d);
        }
        if (minD > bestScore) {
          bestScore = minD;
          best = c;
        }
      }
      if (!best) break;
      picked.push(best);
    }
    return picked.slice(0, k);
  }

  const citySeed = [...cells].sort((a, b) => distToCity(a) - distToCity(b))[0];

  let seeds: HexCell[] = [];
    if (mode === "pack") {
    // One city seed + spread the rest
    seeds = pickFarthestSeeds([citySeed], NUM_DISTRICTS);
  } else {
    // CRACK: start MULTIPLE districts inside/near the city core so the blob gets split immediately.
    // Pick 4 distinct near-city seeds, slightly spread by farthest-point selection restricted to nearCity.
    const nearCity = [...cells].sort((a, b) => distToCity(a) - distToCity(b)).slice(0, 20);

    const crackSeeds: HexCell[] = [];
    crackSeeds.push(citySeed);

    // pick next 3 seeds from nearCity maximizing distance to existing crackSeeds
    while (crackSeeds.length < 4) {
      let best: HexCell | null = null;
      let bestScore = -Infinity;
      for (const c of nearCity) {
        if (crackSeeds.some((s) => s.id === c.id)) continue;
        let minD = Infinity;
        for (const s of crackSeeds) {
          minD = Math.min(minD, hexDist(c.q - s.q, c.r - s.r));
        }
        if (minD > bestScore) {
          bestScore = minD;
          best = c;
        }
      }
      if (!best) break;
      crackSeeds.push(best);
    }

    // Now spread remaining seeds globally
    seeds = pickFarthestSeeds(crackSeeds, NUM_DISTRICTS);
  }

  // Data structures
  const assign: Record<string, number> = {};
  const districtCells: string[][] = Array.from({ length: NUM_DISTRICTS }, () => []);
  const frontier: Array<Set<string>> = Array.from({ length: NUM_DISTRICTS }, () => new Set<string>());

  // Init with seeds
  for (let d = 0; d < NUM_DISTRICTS; d++) {
    const s = seeds[d];
    assign[s.id] = d;
    districtCells[d].push(s.id);
  }
  // Init frontier sets
  for (let d = 0; d < NUM_DISTRICTS; d++) {
    const sid = districtCells[d][0];
    for (const nid of adj.get(sid) ?? []) {
      if (assign[nid] === undefined) frontier[d].add(nid);
    }
  }

  // Track district running totals (for crack scoring)
  const dTotals = Array.from({ length: NUM_DISTRICTS }, () => ({ dem: 0, total: 0 }));
  for (let d = 0; d < NUM_DISTRICTS; d++) {
    const c = cellById.get(districtCells[d][0])!;
    dTotals[d].dem += c.dem;
    dTotals[d].total += c.total;
  }

  // Mode params (tuned for obvious seat contrast)
    // CRACK tuning: keep districts just under 50% Dem, but allow districts to actually take city hexes.
  const crackTarget = 0.49;

  // Less "avoid super-blue" so the core actually gets distributed among districts.
  const crackHighPenalty = 0.6;

  // Mild penalty for extreme red-only grabs (keeps things from becoming too artificial).
  const crackLowPenalty = 0.25;

  // Hard cap: discourage any one district from hogging too many super-blue hexes.
  const superBlueThreshold = 0.70;
  const superBlueCapPerDistrict = 2;
  function pickFromFrontier(d: number): string | null {
    const f = frontier[d];
    if (!f || f.size === 0) return null;

    let bestId: string | null = null;
    let bestScore = Infinity;

    const curDem = dTotals[d].dem;
    const curTot = dTotals[d].total;
    const curShare = curTot > 0 ? curDem / curTot : 0.5;

    for (const cid of f) {
      const cell = cellById.get(cid)!;

      if (mode === "pack") {
        // District 0 (city district): prefer high Dem share
        // Others: prefer low Dem share
        const score = (d === 0)
          ? (1 - cell.demShare) // maximize demShare
          : (cell.demShare);    // minimize demShare
        if (score < bestScore) {
          bestScore = score;
          bestId = cid;
        }
      } else {
        // Crack: prefer moves that keep district share near crackTarget,
        // while penalizing swallowing the most Dem-heavy hexes (spreads them out).
                const nextDem = curDem + cell.dem;
        const nextTot = curTot + cell.total;
        const nextShare = nextTot > 0 ? nextDem / nextTot : curShare;

        // Primary objective: keep share near just-under-50
        let score = Math.abs(nextShare - crackTarget);

        // Count how many "super blue" hexes are already in this district
        let superBlueCount = 0;
        for (const alreadyId of districtCells[d]) {
          const ac = cellById.get(alreadyId)!;
          if (ac.demShare >= superBlueThreshold) superBlueCount++;
        }

        // If adding this would exceed the cap, heavily penalize (forces splitting the core)
        if (cell.demShare >= superBlueThreshold && superBlueCount >= superBlueCapPerDistrict) {
          score += 3.0; // big push away
        }

        // Gentle penalties for extremes (but not so strong that we never take city hexes)
        if (cell.demShare >= superBlueThreshold) score += crackHighPenalty * (cell.demShare - superBlueThreshold);
        if (cell.demShare <= 0.25) score += crackLowPenalty * (0.25 - cell.demShare);

        // Slight preference not to grow too far outward early
        const outward = hexDist(cell.q, cell.r) / RADIUS;
        score += 0.05 * outward;
        if (score < bestScore) {
          bestScore = score;
          bestId = cid;
        }
      }
    }

    return bestId;
  }

  // If frontier empties (rare), we need a fallback to keep contiguity:
  // Find any unassigned cell adjacent to the district (should exist if graph connected).
  function expandFrontier(d: number) {
    for (const id of districtCells[d]) {
      for (const nid of adj.get(id) ?? []) {
        if (assign[nid] === undefined) frontier[d].add(nid);
      }
    }
  }

  // Growth order:
  // - pack: fill district 0 first to target, then round-robin others
  // - crack: round-robin all districts
  const order: number[] = [];
  if (mode === "pack") {
    // push many 0's early
    for (let i = 0; i < (targetSize - 1); i++) order.push(0);
    // then round robin for remaining assignments
    for (let t = 0; t < (cells.length - NUM_DISTRICTS) - (targetSize - 1); t++) {
      order.push(1 + (t % (NUM_DISTRICTS - 1)));
    }
  } else {
    for (let t = 0; t < (cells.length - NUM_DISTRICTS); t++) order.push(t % NUM_DISTRICTS);
  }

  // Build frames progressively (every k assignments)
  const frames: Frame[] = [];
  frames.push({ assign: {} }); // frame 0: nothing

  // Start with seeds shown quickly
  const currentAssign: Record<string, number> = {};
  for (let d = 0; d < NUM_DISTRICTS; d++) {
    const sid = districtCells[d][0];
    currentAssign[sid] = d;
  }
  frames.push({ assign: { ...currentAssign } });

  const snapshotEvery = 5; // makes “drawing” feel progressive
  let steps = 0;

  // helper to add one cell to district
  function addCellToDistrict(d: number, cid: string) {
    assign[cid] = d;
    currentAssign[cid] = d;
    districtCells[d].push(cid);

    const cell = cellById.get(cid)!;
    dTotals[d].dem += cell.dem;
    dTotals[d].total += cell.total;

    // remove from all frontiers
    for (let j = 0; j < NUM_DISTRICTS; j++) frontier[j].delete(cid);

    // add its neighbors to this district frontier
    for (const nid of adj.get(cid) ?? []) {
      if (assign[nid] === undefined) frontier[d].add(nid);
    }
  }

  // Growth loop
  for (const d of order) {
    if (districtCells[d].length >= targetSize) continue;

    if (frontier[d].size === 0) expandFrontier(d);

    let pick = pickFromFrontier(d);

    // If still null, try to steal a frontier cell from other district’s frontier that is adjacent
    if (!pick) {
      expandFrontier(d);
      pick = pickFromFrontier(d);
    }
    if (!pick) {
      // As a last resort (should be extremely rare on this board),
      // assign any unassigned cell adjacent to any cell in district.
      let found: string | null = null;
      for (const id of districtCells[d]) {
        for (const nid of adj.get(id) ?? []) {
          if (assign[nid] === undefined) {
            found = nid;
            break;
          }
        }
        if (found) break;
      }
      if (!found) continue;
      pick = found;
    }

    addCellToDistrict(d, pick);
    steps++;

    if (steps % snapshotEvery === 0) {
      frames.push({ assign: { ...currentAssign } });
    }
  }

  // Ensure all assigned (should be exact)
  for (const c of cells) {
    if (assign[c.id] === undefined) {
      // attach unassigned cell to a neighboring district (contiguous)
      const ns = adj.get(c.id) ?? [];
      const nd = ns.map((nid) => assign[nid]).find((x) => x !== undefined);
      assign[c.id] = nd ?? (NUM_DISTRICTS - 1);
      currentAssign[c.id] = assign[c.id]!;
    }
  }

  frames.push({ assign: { ...currentAssign } });

  return { frames, finalAssign: assign };
}

/**
 * Build bold boundary edges as SVG paths.
 * We draw edges only where:
 * - both cells exist, and
 * - both are assigned, and
 * - districts differ
 *
 * This creates crisp outlines (not fat hex outlines everywhere).
 */
/**
 * Build bold boundary edges as SVG paths.
 *
 * IMPORTANT changes vs old version:
 * 1) We treat "unassigned neighbor" as boundary too, so districts get a full outline
 *    at every animation step (even while still being drawn).
 * 2) We DO NOT use a hard-coded direction->edge mapping (which caused wonky edges).
 *    Instead, we pick the edge whose midpoint points most toward the neighbor center.
 * 3) Dedupe per (cellId, edgeIndex) so we don't drop segments incorrectly.
 */
function buildBoundaryPaths(
  cells: HexCell[],
  assign: Record<string, number>,
  size: number,
  exists: (id: string) => boolean
): string[] {
  const paths: string[] = [];

  // Precompute pixel centers + corners
  const center = new Map<string, { x: number; y: number }>();
  const corners = new Map<string, Array<[number, number]>>();
  for (const c of cells) {
    const p = axialToPixel(c.q, c.r, size);
    center.set(c.id, p);
    corners.set(c.id, hexCorners(p.x, p.y, size));
  }

  // Unique key per cell edge so we don't draw duplicates.
  // (We intentionally do NOT dedupe by neighbor-pair only; that was dropping edges.)
  const drawn = new Set<string>();

  for (const c of cells) {
    const d = assign[c.id];
    if (d === undefined) continue; // only draw outlines for districts that exist so far

    const cc = center.get(c.id)!;
    const cs = corners.get(c.id)!;

    // For each of the 6 neighbor directions
    for (const [nq, nr] of neighbors(c.q, c.r)) {
      const nid = `${nq},${nr}`;

      // If off-board, this is definitely an outer boundary edge
      const neighborOnBoard = exists(nid);

      // Determine if this neighbor direction should produce a boundary edge
      let isBoundary = false;

      if (!neighborOnBoard) {
        isBoundary = true;
      } else {
        const nd = assign[nid];
        if (nd === undefined) {
          // neighbor exists but is not drawn yet -> boundary of the current partial district
          isBoundary = true;
        } else if (nd !== d) {
          // neighbor is different district
          isBoundary = true;
        } else {
          // same district, not a boundary
          isBoundary = false;
        }
      }

      if (!isBoundary) continue;

      // Determine the neighbor center direction in pixel space.
      // If neighbor is off-board, approximate direction using axial neighbor vector -> pixel.
      let nx: number, ny: number;
      if (neighborOnBoard) {
        const np = center.get(nid)!;
        nx = np.x;
        ny = np.y;
      } else {
        // approximate by shifting in the neighbor axial direction
        const approx = axialToPixel(nq, nr, size);
        nx = approx.x;
        ny = approx.y;
      }

      const vx = nx - cc.x;
      const vy = ny - cc.y;

      // Pick the edge whose midpoint points most in direction (vx, vy)
      // Edge i connects corner i -> corner (i+1)
      let bestEdge = 0;
      let bestDot = -Infinity;

      for (let i = 0; i < 6; i++) {
        const a = cs[i];
        const b = cs[(i + 1) % 6];
        const mx = (a[0] + b[0]) / 2;
        const my = (a[1] + b[1]) / 2;
        const ex = mx - cc.x;
        const ey = my - cc.y;
        const dot = ex * vx + ey * vy; // maximize alignment
        if (dot > bestDot) {
          bestDot = dot;
          bestEdge = i;
        }
      }

      const key = `${c.id}|e${bestEdge}`;
      if (drawn.has(key)) continue;
      drawn.add(key);

      const a = cs[bestEdge];
      const b = cs[(bestEdge + 1) % 6];
      paths.push(`M ${a[0]} ${a[1]} L ${b[0]} ${b[1]}`);
    }
  }

  return paths;
}

export default function FairnessPage() {
  const [mode, setMode] = useState<Mode>("pack");
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [tooltip, setTooltip] = useState<TooltipState>({ visible: false, x: 0, y: 0, content: null });

  const coords = useMemo(() => generateHexBoard(RADIUS), []);
  const cells = useMemo(() => buildVoteGrid(coords), [coords]);
  const statewide = useMemo(() => {
    const dem = cells.reduce((s, c) => s + c.dem, 0);
    const rep = cells.reduce((s, c) => s + c.rep, 0);
    const total = dem + rep;
    const demShare = total > 0 ? dem / total : 0.5;
    return { dem, rep, total, demShare };
    }, [cells]);

  const idSet = useMemo(() => new Set(cells.map((c) => c.id)), [cells]);
  const existsId = (id: string) => idSet.has(id);

  const { frames, finalAssign } = useMemo(() => buildContiguousPlan(cells, mode), [cells, mode]);

  const maxStep = frames.length - 1;
  const frame = frames[clamp(step, 0, maxStep)];

  const { totals, winners, seats } = useMemo(
    () => computeDistrictTotals(cells, frame.assign),
    [cells, frame.assign]
  );

  // build boundary paths for current frame (left/right use same)
  const boundaryPaths = useMemo(
    () => buildBoundaryPaths(cells, frame.assign, HEX_SIZE, existsId),
    [cells, frame.assign, idSet]
  );

  // play/pause
  const intervalRef = useRef<number | null>(null);
  useEffect(() => {
    if (!playing) {
      if (intervalRef.current) window.clearInterval(intervalRef.current);
      intervalRef.current = null;
      return;
    }
    if (intervalRef.current) window.clearInterval(intervalRef.current);

    intervalRef.current = window.setInterval(() => {
      setStep((s) => (s >= maxStep ? maxStep : s + 1));
    }, 520);

    return () => {
      if (intervalRef.current) window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    };
  }, [playing, maxStep]);

  useEffect(() => {
    if (playing && step >= maxStep) setPlaying(false);
  }, [playing, step, maxStep]);

  useEffect(() => {
    setPlaying(false);
    setStep(0);
    setTooltip({ visible: false, x: 0, y: 0, content: null });
  }, [mode]);

  // SVG viewBox bounds
  const bounds = useMemo(() => {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const c of cells) {
      const p = axialToPixel(c.q, c.r, HEX_SIZE);
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }
    minX -= HEX_SIZE + PANEL_PAD;
    minY -= HEX_SIZE + PANEL_PAD;
    maxX += HEX_SIZE + PANEL_PAD;
    maxY += HEX_SIZE + PANEL_PAD;
    return { minX, minY, maxX, maxY, w: maxX - minX, h: maxY - minY };
  }, [cells]);

  function showTooltip(e: React.MouseEvent, content: React.ReactNode) {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setTooltip({
      visible: true,
      x: e.clientX - rect.left + 12,
      y: e.clientY - rect.top + 12,
      content,
    });
  }
  function hideTooltip() {
    setTooltip((t) => ({ ...t, visible: false }));
  }

  const headline =
    mode === "pack"
      ? "Pack votes: concentrate a voter cluster into fewer districts"
      : "Crack votes: split a voter cluster across many districts";

  const explainer =
    mode === "pack"
      ? "Packing grows a contiguous 'city district' around the dense cluster, creating a landslide seat and leaving fewer winnable districts elsewhere."
      : "Cracking grows multiple contiguous districts through the dense cluster, spreading those voters out so they fall just under a majority in more places.";

  return (
    <div style={{ padding: 20, maxWidth: 1240, margin: "0 auto" }}>
      <h1 style={{ fontSize: 28, marginBottom: 8 }}>Fairness Demo: Packing vs Cracking</h1>
      <p style={{ marginTop: 0, color: "#555", lineHeight: 1.4 }}>
        A teaching demo on a <b>91-hex</b> board with <b>7 contiguous districts</b>. Each hex has equal population.
        Left shows <b>vote distribution</b>; right shows <b>representation outcome</b>.
      </p>

      {/* Controls */}
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", margin: "14px 0" }}>
        <div style={{ display: "inline-flex", border: "1px solid #ddd", borderRadius: 10, overflow: "hidden" }}>
          <button
            onClick={() => setMode("pack")}
            style={{
              padding: "10px 14px",
              border: "none",
              background: mode === "pack" ? "#111" : "#fff",
              color: mode === "pack" ? "#fff" : "#111",
              cursor: "pointer",
            }}
          >
            Pack
          </button>
          <button
            onClick={() => setMode("crack")}
            style={{
              padding: "10px 14px",
              border: "none",
              background: mode === "crack" ? "#111" : "#fff",
              color: mode === "crack" ? "#fff" : "#111",
              cursor: "pointer",
            }}
          >
            Crack
          </button>
        </div>

        <button
          onClick={() => setPlaying((p) => !p)}
          style={{
            padding: "10px 14px",
            borderRadius: 10,
            border: "1px solid #ddd",
            background: "#fff",
            cursor: "pointer",
          }}
        >
          {playing ? "Pause" : "Play"}
        </button>

        <button
          onClick={() => {
            setPlaying(false);
            setStep(0);
          }}
          style={{
            padding: "10px 14px",
            borderRadius: 10,
            border: "1px solid #ddd",
            background: "#fff",
            cursor: "pointer",
          }}
        >
          Reset
        </button>

        <div style={{ marginLeft: 6, color: "#333" }}>
          <b>Step:</b> {step}/{maxStep}
        </div>

        <div style={{ marginLeft: "auto", display: "flex", gap: 18, alignItems: "center" }}>

        {/* Statewide vote */}
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <div style={{ fontWeight: 800 }}>Statewide vote share:</div>

            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 12, height: 12, borderRadius: 3, background: solidPartyColor("dem") }} />
            {(statewide.demShare * 100).toFixed(1)}%
            </span>

            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 12, height: 12, borderRadius: 3, background: solidPartyColor("rep") }} />
            {((1 - statewide.demShare) * 100).toFixed(1)}%
            </span>
        </div>

        {/* Seats */}
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <div style={{ fontWeight: 800 }}>Seats:</div>

            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 12, height: 12, borderRadius: 3, background: solidPartyColor("dem") }} />
            {seats.dem}
            </span>

            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 12, height: 12, borderRadius: 3, background: solidPartyColor("rep") }} />
            {seats.rep}
            </span>
        </div>

        </div>
      </div>

      <div style={{ padding: 14, border: "1px solid #eee", borderRadius: 14, background: "#fafafa" }}>
        <div style={{ fontSize: 16, fontWeight: 800, marginBottom: 6 }}>{headline}</div>
        <div style={{ color: "#555" }}>{explainer}</div>
      </div>

      {/* Two panels */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 16 }}>
        {/* Left */}
        <div style={{ border: "1px solid #eee", borderRadius: 16, padding: 12, position: "relative" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <div>
              <div style={{ fontWeight: 900, fontSize: 16 }}>Vote distribution</div>
              <div style={{ color: "#666", fontSize: 13 }}>Hex shading shows Dem vote share</div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ color: "#666", fontSize: 12 }}>More Rep</span>
              <div
                style={{
                  width: 120,
                  height: 10,
                  borderRadius: 999,
                  background: "linear-gradient(90deg, rgb(220,60,60), rgb(245,245,245), rgb(60,110,220))",
                  border: "1px solid #ddd",
                }}
              />
              <span style={{ color: "#666", fontSize: 12 }}>More Dem</span>
            </div>
          </div>

          <svg
            width="100%"
            viewBox={`${bounds.minX} ${bounds.minY} ${bounds.w} ${bounds.h}`}
            style={{ marginTop: 10, background: "#fff", borderRadius: 12 }}
          >
            {/* fills */}
            {cells.map((c) => {
              const p = axialToPixel(c.q, c.r, HEX_SIZE);
              const pts = hexPoints(p.x, p.y, HEX_SIZE);
              const fill = shareToRedBlue(c.demShare);

              return (
                <polygon
                  key={`Lfill-${c.id}`}
                  points={pts}
                  fill={fill}
                  stroke={"rgba(0,0,0,0.16)"}
                  strokeWidth={0.6}
                  onMouseMove={(e) =>
                    showTooltip(
                      e,
                      <div style={{ fontSize: 12 }}>
                        <div style={{ fontWeight: 800, marginBottom: 4 }}>Hex {c.id}</div>
                        <div>Total voters: {c.total}</div>
                        <div>Dem: {c.dem}</div>
                        <div>Rep: {c.rep}</div>
                        <div>Dem share: {(c.demShare * 100).toFixed(1)}%</div>
                        <div>
                          District:{" "}
                          {frame.assign[c.id] === undefined ? (
                            <span style={{ color: "#bbb" }}>unassigned</span>
                          ) : (
                            frame.assign[c.id] + 1
                          )}
                        </div>
                      </div>
                    )
                  }
                  onMouseLeave={hideTooltip}
                />
              );
            })}

            {/* bold district boundaries (crisp edges) */}
            <g>
              {boundaryPaths.map((d, i) => (
                <path
                  key={`Lbd-${i}`}
                  d={d}
                  fill="none"
                  stroke="rgba(0,0,0,0.75)"
                  strokeWidth={3.2}
                  strokeLinecap="round"
                />
              ))}
            </g>
          </svg>

          {/* Tooltip */}
          {tooltip.visible && (
            <div
              style={{
                position: "absolute",
                left: tooltip.x,
                top: tooltip.y,
                background: "rgba(20,20,20,0.92)",
                color: "#fff",
                padding: "10px 10px",
                borderRadius: 10,
                pointerEvents: "none",
                maxWidth: 260,
              }}
            >
              {tooltip.content}
            </div>
          )}
        </div>

        {/* Right */}
        <div style={{ border: "1px solid #eee", borderRadius: 16, padding: 12, position: "relative" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <div>
              <div style={{ fontWeight: 900, fontSize: 16 }}>Final districts & seats</div>
              <div style={{ color: "#666", fontSize: 13 }}>Hex fill shows district winner (solid)</div>
            </div>
            <div style={{ color: "#666", fontSize: 13 }}>
              Dem {seats.dem} — Rep {seats.rep}
            </div>
          </div>

          <svg
            width="100%"
            viewBox={`${bounds.minX} ${bounds.minY} ${bounds.w} ${bounds.h}`}
            style={{ marginTop: 10, background: "#fff", borderRadius: 12 }}
          >
            {/* fills */}
            {cells.map((c) => {
              const p = axialToPixel(c.q, c.r, HEX_SIZE);
              const pts = hexPoints(p.x, p.y, HEX_SIZE);

              const d = frame.assign[c.id];
              const isAssigned = d !== undefined;

              let fill = "rgb(235,235,235)";
              if (isAssigned) fill = solidPartyColor(winners[d]);

              return (
                <polygon
                  key={`Rfill-${c.id}`}
                  points={pts}
                  fill={fill}
                  stroke={"rgba(0,0,0,0.16)"}
                  strokeWidth={0.6}
                  onMouseMove={(e) => {
                    if (!isAssigned) {
                      showTooltip(
                        e,
                        <div style={{ fontSize: 12 }}>
                          <div style={{ fontWeight: 800, marginBottom: 4 }}>Hex {c.id}</div>
                          <div style={{ color: "#ddd" }}>Unassigned (animation step)</div>
                        </div>
                      );
                      return;
                    }
                    const t = totals[d];
                    const share = t.total > 0 ? t.dem / t.total : 0.5;
                    const win = winners[d];
                    const marginPts = Math.abs(share - 0.5) * 200; // 0..100

                    showTooltip(
                      e,
                      <div style={{ fontSize: 12 }}>
                        <div style={{ fontWeight: 800, marginBottom: 4 }}>District {d + 1}</div>
                        <div>Dem: {t.dem}</div>
                        <div>Rep: {t.rep}</div>
                        <div>Dem share: {(share * 100).toFixed(1)}%</div>
                        <div>
                          Winner: {win === "dem" ? "Dem" : "Rep"} by {marginPts.toFixed(1)} pts
                        </div>
                      </div>
                    );
                  }}
                  onMouseLeave={hideTooltip}
                />
              );
            })}

            {/* bold boundaries */}
            <g>
              {boundaryPaths.map((d, i) => (
                <path
                  key={`Rbd-${i}`}
                  d={d}
                  fill="none"
                  stroke="rgba(0,0,0,0.82)"
                  strokeWidth={3.6}
                  strokeLinecap="round"
                />
              ))}
            </g>
          </svg>

          {/* District summary cards */}
          <div style={{ marginTop: 12 }}>
            <div style={{ fontWeight: 900, marginBottom: 8 }}>District totals (current frame)</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 8 }}>
              {totals.map((t, d) => {
                const assigned = t.assignedCells > 0;
                const share = t.total > 0 ? t.dem / t.total : 0.5;
                const win = winners[d];

                return (
                  <div
                    key={`card-${d}`}
                    style={{
                      border: "1px solid #eee",
                      borderRadius: 12,
                      padding: 10,
                      background: "#fafafa",
                      opacity: assigned ? 1 : 0.55,
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div style={{ fontWeight: 900 }}>D{d + 1}</div>
                      <div
                        style={{
                          width: 10,
                          height: 10,
                          borderRadius: 3,
                          background: assigned ? solidPartyColor(win) : "rgb(210,210,210)",
                        }}
                      />
                    </div>
                    <div style={{ fontSize: 12, color: "#555", marginTop: 6 }}>
                      {assigned ? (
                        <>
                          <div>Dem: {t.dem}</div>
                          <div>Rep: {t.rep}</div>
                          <div>Dem%: {(share * 100).toFixed(1)}%</div>
                        </>
                      ) : (
                        <div>Not drawn yet</div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Tooltip (right panel) */}
          {tooltip.visible && (
            <div
              style={{
                position: "absolute",
                left: tooltip.x,
                top: tooltip.y,
                background: "rgba(20,20,20,0.92)",
                color: "#fff",
                padding: "10px 10px",
                borderRadius: 10,
                pointerEvents: "none",
                maxWidth: 260,
              }}
            >
              {tooltip.content}
            </div>
          )}
        </div>
      </div>

      <div style={{ marginTop: 18, color: "#666", lineHeight: 1.5 }}>
        <p style={{ marginBottom: 6 }}>
          <b>Contiguity:</b> Every district is grown by repeatedly adding neighboring hexes, so districts stay contiguous
          in both Pack and Crack.
        </p>
      </div>
    </div>
  );
}