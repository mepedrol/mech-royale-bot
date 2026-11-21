"""Static map model: a 2 m grid of ground/solid heights, walkability, line of sight, and A*."""
import array
import heapq
import json
import math
import os
import time

C = 2.0  # cell size, meters
SQ2 = math.sqrt(2.0)


def _point_in_ring(px, py, ring):
    inside = False
    n = len(ring) // 2
    j = n - 1
    for i in range(n):
        xi, yi = ring[2 * i], ring[2 * i + 1]
        xj, yj = ring[2 * j], ring[2 * j + 1]
        if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


class NavMap:
    def __init__(self, map_dir, mech_cfg=None):
        t0 = time.time()
        mech_cfg = mech_cfg or {}
        self.step_up = mech_cfg.get("step_up", 1.2)
        self.mech_h = mech_cfg.get("height", 9.0)
        self.radius = mech_cfg.get("radius", 2.5)
        with open(os.path.join(map_dir, "v2", "world.json")) as f:
            w = json.load(f)
        tr = w["terrain"]
        self.tnx, self.tny = tr["nx"], tr["ny"]
        self.tminx, self.tminy, self.tstep = tr["minX"], tr["minY"], tr["step"]
        scale = tr.get("scale", 0.01)
        a = array.array("h")
        with open(os.path.join(map_dir, "v2", tr.get("file", "terrain.i16")), "rb") as f:
            a.frombytes(f.read())
        if array.array("h", [1]).tobytes() != b"\x01\x00":
            a.byteswap()
        self.terr = [v * scale for v in a]
        b = w["bounds"]
        self.minx, self.miny = b["minX"], b["minY"]
        self.maxx, self.maxy = b["maxX"], b["maxY"]
        self.gx = int(math.ceil((self.maxx - self.minx) / C))
        self.gy = int(math.ceil((self.maxy - self.miny) / C))
        gx, gy = self.gx, self.gy
        n = gx * gy
        # ground height at cell centers
        ground = [0.0] * n
        for j in range(gy):
            cy = self.miny + (j + 0.5) * C
            for i in range(gx):
                ground[j * gx + i] = self.terrain(self.minx + (i + 0.5) * C, cy)
        self.ground = ground
        # solid tops (prisms resting on the ground); overhead decks are ignored for movement
        solid = [-1e9] * n
        isprism = bytearray(n)
        over_base = {}  # cell -> (base, top) of overhead solids (bridges you can walk under)
        for p in w["prisms"]:
            ring = p["ring"]
            holes = p.get("holes") or []
            xs = ring[0::2]
            ys = ring[1::2]
            j0 = max(0, int((min(ys) - self.miny) / C))
            j1 = min(gy - 1, int((max(ys) - self.miny) / C))
            i0 = max(0, int((min(xs) - self.minx) / C))
            i1 = min(gx - 1, int((max(xs) - self.minx) / C))
            base, top = p["base"], p["top"]
            for j in range(j0, j1 + 1):
                cy = self.miny + (j + 0.5) * C
                # scanline crossings
                xsx = []
                rings = [ring] + list(holes)
                for r in rings:
                    m = len(r) // 2
                    for k in range(m):
                        x1, y1 = r[2 * k], r[2 * k + 1]
                        x2, y2 = r[2 * ((k + 1) % m)], r[2 * ((k + 1) % m) + 1]
                        if (y1 > cy) != (y2 > cy):
                            xsx.append(x1 + (cy - y1) * (x2 - x1) / (y2 - y1))
                if not xsx:
                    continue
                xsx.sort()
                for k in range(0, len(xsx) - 1, 2):
                    ia = max(i0, int(math.ceil((xsx[k] - self.minx) / C - 0.5)))
                    ib = min(i1, int(math.floor((xsx[k + 1] - self.minx) / C - 0.5)))
                    for i in range(ia, ib + 1):
                        c = j * gx + i
                        g = ground[c]
                        if base > g + self.mech_h + 0.3:
                            ob = over_base.get(c)
                            if ob is None or base < ob[0]:
                                over_base[c] = (base, top)
                            continue
                        if top > solid[c]:
                            solid[c] = top
                        isprism[c] = 1
        self.over = over_base
        h = [0.0] * n
        for c in range(n):
            h[c] = solid[c] if solid[c] > ground[c] else ground[c]
        self.h = h
        self.isprism = isprism
        # blocked for a mech center: a wall (prism rising > step_up above us) within radius
        blocked = bytearray(n)
        near = bytearray(n)  # within ~6 m of a wall: path cost
        rr = int(math.ceil((self.radius + 0.6) / C))
        rn = 3
        offs_b = [(di, dj) for di in range(-rr, rr + 1) for dj in range(-rr, rr + 1) if (di * di + dj * dj) * C * C <= (self.radius + 0.9) ** 2]
        offs_n = [(di, dj) for di in range(-rn, rn + 1) for dj in range(-rn, rn + 1) if di * di + dj * dj <= rn * rn]
        for c in range(n):
            if not isprism[c]:
                continue
            i, j = c % gx, c // gx
            hc = h[c]
            for di, dj in offs_b:
                ii, jj = i + di, j + dj
                if 0 <= ii < gx and 0 <= jj < gy:
                    cc = jj * gx + ii
                    if hc - h[cc] > self.step_up + 0.2:
                        blocked[cc] = 1
            for di, dj in offs_n:
                ii, jj = i + di, j + dj
                if 0 <= ii < gx and 0 <= jj < gy:
                    cc = jj * gx + ii
                    if hc - h[cc] > self.step_up + 0.2:
                        near[cc] = 1
        # map edge margin
        for i in range(gx):
            for j in (0, 1, gy - 2, gy - 1):
                blocked[j * gx + i] = 1
        for j in range(gy):
            for i in (0, 1, gx - 2, gx - 1):
                blocked[j * gx + i] = 1
        self.blocked = blocked
        self.near = near
        # trees: canopy cover count per cell (for hiding)
        tr = w.get("trees") or {}
        cover = bytearray(n)
        fields = tr.get("fields") or []
        data = tr.get("data") or []
        nf = len(fields)
        self.trees = []
        if nf:
            fi = {k: i for i, k in enumerate(fields)}
            for q in range(0, len(data) - nf + 1, nf):
                cx = data[q + fi.get("crownX", fi["x"])]
                cy = data[q + fi.get("crownY", fi["y"])]
                cr = data[q + fi["crownR"]]
                ci, cj = self.cell_ij(cx, cy)
                k = int(cr / C) + 1
                for dj in range(-k, k + 1):
                    for di in range(-k, k + 1):
                        ii, jj = ci + di, cj + dj
                        if 0 <= ii < gx and 0 <= jj < gy and (di * di + dj * dj) * C * C <= cr * cr:
                            cc = jj * gx + ii
                            if cover[cc] < 250:
                                cover[cc] += 1
        self.cover = cover
        self._flood()
        self.load_s = time.time() - t0

    def _flood(self):
        """Cells reachable on foot from the big central ground area."""
        gx, gy = self.gx, self.gy
        n = gx * gy
        best = None
        for r in range(0, 200, 4):
            for x, y in ((41.3 + r, -72.8), (41.3 - r, -72.8), (41.3, -72.8 + r), (41.3, -72.8 - r)):
                c = self.cell(x, y)
                if not self.blocked[c] and not self.isprism[c]:
                    best = c
                    break
            if best is not None:
                break
        reach = bytearray(n)
        reach[best] = 1
        stack = [best]
        h, blocked, isprism, su = self.h, self.blocked, self.isprism, self.step_up
        while stack:
            c = stack.pop()
            ci, cj = c % gx, c // gx
            hc = h[c]
            pc = isprism[c]
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = ci + di, cj + dj
                if 0 <= ii < gx and 0 <= jj < gy:
                    nc = jj * gx + ii
                    if reach[nc] or blocked[nc]:
                        continue
                    dh = h[nc] - hc
                    if (pc or isprism[nc]) and dh > su:
                        continue
                    if dh > 2.2:
                        continue
                    reach[nc] = 1
                    stack.append(nc)
        self.reach = reach
        self.reach_cells = [c for c in range(n) if reach[c]]

    def nearest_reach(self, x, y, maxr=40):
        c0 = self.cell(x, y)
        if self.reach[c0]:
            return c0
        ci, cj = c0 % self.gx, c0 // self.gx
        for r in range(1, maxr + 1):
            best, bd = None, 1e9
            for dj in range(-r, r + 1):
                for di in ((-r, r) if abs(dj) != r else range(-r, r + 1)):
                    ii, jj = ci + di, cj + dj
                    if 0 <= ii < self.gx and 0 <= jj < self.gy:
                        cc = jj * self.gx + ii
                        if self.reach[cc]:
                            dd = di * di + dj * dj
                            if dd < bd:
                                bd, best = dd, cc
            if best is not None:
                return best
        return None

    # ------------------------------------------------------------ sampling
    def terrain(self, x, y):
        fx = (x - self.tminx) / self.tstep
        fy = (y - self.tminy) / self.tstep
        i = int(math.floor(fx))
        j = int(math.floor(fy))
        if i < 0:
            i, fx = 0, 0.0
        if j < 0:
            j, fy = 0, 0.0
        if i >= self.tnx - 1:
            i, fx = self.tnx - 2, float(self.tnx - 1)
        if j >= self.tny - 1:
            j, fy = self.tny - 2, float(self.tny - 1)
        u, v = fx - i, fy - j
        u = min(1.0, max(0.0, u))
        v = min(1.0, max(0.0, v))
        t = self.terr
        nx = self.tnx
        a = t[j * nx + i]
        b = t[j * nx + i + 1]
        c = t[(j + 1) * nx + i]
        d = t[(j + 1) * nx + i + 1]
        return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v

    def cell_ij(self, x, y):
        i = int((x - self.minx) / C)
        j = int((y - self.miny) / C)
        return min(self.gx - 1, max(0, i)), min(self.gy - 1, max(0, j))

    def cell(self, x, y):
        i, j = self.cell_ij(x, y)
        return j * self.gx + i

    def center(self, c):
        return (self.minx + (c % self.gx + 0.5) * C, self.miny + (c // self.gx + 0.5) * C)

    def height_at(self, x, y):
        return self.h[self.cell(x, y)]

    def ground_at(self, x, y):
        return self.ground[self.cell(x, y)]

    def walkable(self, c):
        return not self.blocked[c]

    def in_bounds(self, x, y):
        return self.minx + 4 < x < self.maxx - 4 and self.miny + 4 < y < self.maxy - 4

    # ------------------------------------------------------------ line of sight
    def los(self, a, b, step=1.5):
        """True if the segment a->b (3D) clears buildings/terrain (grid approximation)."""
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        d = math.sqrt(dx * dx + dy * dy)
        if d < 1e-6:
            return True
        n = int(d / step) + 1
        h = self.h
        over = self.over
        gx = self.gx
        minx, miny = self.minx, self.miny
        for k in range(1, n):
            t = k / n
            x = a[0] + dx * t
            y = a[1] + dy * t
            z = a[2] + dz * t
            i = int((x - minx) / C)
            j = int((y - miny) / C)
            if i < 0 or j < 0 or i >= gx or j >= self.gy:
                return False
            c = j * gx + i
            if z < h[c]:
                return False
            if over:
                o = over.get(c)
                if o is not None and o[0] <= z <= o[1]:
                    return False
        return True

    # ------------------------------------------------------------ walking
    def can_step(self, c, nc):
        """Directed edge c -> nc."""
        if self.blocked[nc]:
            return False
        dh = self.h[nc] - self.h[c]
        if self.isprism[nc] or self.isprism[c]:
            return dh <= self.step_up
        return dh <= 2.2  # terrain slopes (50 deg over 2 m)

    def straight_ok(self, x0, y0, x1, y1):
        """Can we walk straight from (x0,y0) to (x1,y1)?"""
        dx, dy = x1 - x0, y1 - y0
        d = math.hypot(dx, dy)
        n = int(d / (C * 0.5)) + 1
        prev = self.cell(x0, y0)
        for k in range(1, n + 1):
            t = k / n
            c = self.cell(x0 + dx * t, y0 + dy * t)
            if c != prev:
                if not self.can_step(prev, c):
                    return False
                prev = c
        return True

    def nearest_walkable(self, x, y, maxr=12):
        c0 = self.cell(x, y)
        if not self.blocked[c0]:
            return c0
        ci, cj = c0 % self.gx, c0 // self.gx
        best, bd = None, 1e9
        for r in range(1, maxr + 1):
            for dj in range(-r, r + 1):
                for di in (-r, r) if abs(dj) != r else range(-r, r + 1):
                    ii, jj = ci + di, cj + dj
                    if 0 <= ii < self.gx and 0 <= jj < self.gy:
                        cc = jj * self.gx + ii
                        if not self.blocked[cc]:
                            dd = di * di + dj * dj
                            if dd < bd:
                                bd, best = dd, cc
            if best is not None:
                return best
        return None


class AStar:
    """Resumable A* on the NavMap grid. Call step(budget_s) until done."""
    NB = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)]

    def __init__(self, nm, start, goal, cost_fn=None, weight=1.3, max_exp=60000):
        self.nm = nm
        self.start, self.goal = start, goal
        self.weight = weight
        self.cost_fn = cost_fn
        self.max_exp = max_exp
        self.g = {start: 0.0}
        self.parent = {start: -1}
        self.open = [(0.0, start)]
        self.closed = set()
        self.done = False
        self.path = None
        self.exp = 0
        self.best = start
        self.best_h = 1e18
        gi, gj = goal % nm.gx, goal // nm.gx
        self.gi, self.gj = gi, gj

    def _h(self, c):
        gx = self.nm.gx
        di = abs(c % gx - self.gi)
        dj = abs(c // gx - self.gj)
        return (max(di, dj) + (SQ2 - 1) * min(di, dj)) * C

    def step(self, budget_s):
        if self.done:
            return True
        nm = self.nm
        gx, gy = nm.gx, nm.gy
        h = nm.h
        blocked = nm.blocked
        isprism = nm.isprism
        near = nm.near
        cover = nm.cover
        step_up = nm.step_up
        cost_fn = self.cost_fn
        g = self.g
        parent = self.parent
        openl = self.open
        closed = self.closed
        goal = self.goal
        w = self.weight
        tend = time.perf_counter() + budget_s
        cnt = 0
        while openl:
            cnt += 1
            if (cnt & 255) == 0 and time.perf_counter() > tend:
                return False
            _, c = heapq.heappop(openl)
            if c in closed:
                continue
            closed.add(c)
            self.exp += 1
            if c == goal:
                self._finish(goal)
                return True
            if self.exp > self.max_exp:
                self._finish(self.best)
                return True
            ci, cj = c % gx, c // gx
            gc = g[c]
            hc = h[c]
            pc = isprism[c]
            for di, dj, dc in self.NB:
                ii, jj = ci + di, cj + dj
                if ii < 0 or jj < 0 or ii >= gx or jj >= gy:
                    continue
                nc = jj * gx + ii
                if blocked[nc] or nc in closed:
                    continue
                dh = h[nc] - hc
                if pc or isprism[nc]:
                    if dh > step_up:
                        continue
                elif dh > 2.2:
                    continue
                if di and dj:  # no corner cutting
                    if blocked[cj * gx + ii] or blocked[jj * gx + ci]:
                        continue
                cost = dc * C
                if near[nc] or cover[nc]:
                    cost *= 0.8
                if dh > 0.5:
                    cost += dh
                if cost_fn is not None:
                    cost += cost_fn(nc)
                ng = gc + cost
                og = g.get(nc)
                if og is None or ng < og:
                    g[nc] = ng
                    parent[nc] = c
                    hh = self._h(nc)
                    if hh < self.best_h:
                        self.best_h, self.best = hh, nc
                    heapq.heappush(openl, (ng + w * hh, nc))
        self._finish(self.best)
        return True

    def _finish(self, end):
        p = []
        c = end
        while c != -1:
            p.append(c)
            c = self.parent.get(c, -1)
        p.reverse()
        self.path = p
        self.done = True
        self.reached = end == self.goal


class Field:
    """Incremental Dijkstra distance-to-goal field over the walk grid (reverse edges)."""
    NB = AStar.NB

    def __init__(self, nm, goal):
        self.nm = nm
        self.goal = goal
        n = nm.gx * nm.gy
        self.dist = [3.0e38] * n
        self.dist[goal] = 0.0
        self.open = [(0.0, goal)]
        self.done = False

    def step(self, budget_s):
        if self.done:
            return True
        nm = self.nm
        gx, gy = nm.gx, nm.gy
        h, blocked, isprism, near, cover, su = nm.h, nm.blocked, nm.isprism, nm.near, nm.cover, nm.step_up
        dist = self.dist
        openl = self.open
        pop, push = heapq.heappop, heapq.heappush
        tend = time.perf_counter() + budget_s
        cnt = 0
        NB = self.NB
        while openl:
            cnt += 1
            if (cnt & 511) == 0 and time.perf_counter() > tend:
                return False
            du, u = pop(openl)
            if du > dist[u]:
                continue
            ui, uj = u % gx, u // gx
            hu = h[u]
            pu = isprism[u]
            for di, dj, dc in NB:
                ii, jj = ui + di, uj + dj
                if ii < 0 or jj < 0 or ii >= gx or jj >= gy:
                    continue
                v = jj * gx + ii
                if blocked[v]:
                    continue
                # edge v -> u must be walkable
                dh = hu - h[v]
                if pu or isprism[v]:
                    if dh > su:
                        continue
                elif dh > 2.2:
                    continue
                if di and dj and (blocked[uj * gx + ii - di] or blocked[(uj - dj) * gx + ui]):
                    pass
                cost = dc * C
                if near[u] or cover[u]:
                    cost *= 0.85
                if dh > 0.5:
                    cost += dh
                nd = du + cost
                if nd < dist[v]:
                    dist[v] = nd
                    push(openl, (nd, v))
        self.done = True
        return True

    def d(self, c):
        v = self.dist[c]
        return None if v > 1e37 else v
