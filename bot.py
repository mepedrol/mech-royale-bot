"""Survivor bot for mechfight/1: stay in the zone, avoid fights, regen, loot safely, shoot well when forced."""
import math
import random
import sys
import time

import os

import mechfight as mf

AGGRO = os.environ.get("MF_AGGRO") == "1"  # test opponent: hunts instead of hiding
SKILL = int(os.environ.get("MF_SKILL", "3"))  # test opponent aim quality 0..3
AIM_NOISE = [0.25, 0.10, 0.035, 0.01][max(0, min(3, SKILL))] if AGGRO else 0.0
from navmap import AStar, Field, NavMap

TAU = 2 * math.pi

UPGRADE_ORDER = [
    "armor", "plating", "nanites", "laser_charge", "laser_damage", "gun_damage", "gun_rate",
    "lightning_efficiency", "gyros", "servos", "sensors", "gun_mag", "gun_velocity", "reactor",
    "capacitor", "jets", "lightning_range", "lightning_chain", "laser_pierce",
    "mortar_reload", "mortar_splash", "mortar_velocity",
]
WEAPON_VALUE = {"laser": 3.0, "lightning": 2.5, "gun": 2.0, "mortar": -1.0}


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def dist2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def log(*a):
    print(*a, file=sys.stderr, flush=True)


LOGF = os.environ.get("MF_LOG")
NODUEL = os.environ.get("MF_NODUEL") == "1"
HOME_CENTER = os.environ.get("MF_HOMECENTER") == "1" or AGGRO


class Bot:
    name = "hunter" if AGGRO else "survivor"

    def __init__(self, hello):
        self.cfg = cfg = hello["config"]
        self.slot = hello.get("slot", 0)
        self.rng = random.Random(hello.get("seed", 1))
        self.mech = cfg["mech"]
        self.W = cfg["weapons"]
        self.loot_r = cfg["loot"]["pickup_radius"]
        self.cube_r = cfg["cubes"]["pickup_radius"]
        drops = cfg.get("drops") or {}
        self.drop_capture = drops.get("capture_s", 4)
        self.nm = None
        try:
            self.nm = NavMap(hello["map"]["dir"], self.mech)
            log(f"map loaded in {self.nm.load_s:.2f}s")
        except Exception as e:  # play without a map rather than crash
            log("map load failed:", repr(e))
        # navigation
        self.search = None
        self.path = None
        self.path_i = 0
        self.path_goal = None
        self.path_t = -99
        self.goal_kind = ""
        # state
        self.home = None
        self.home_key = None
        self.home_t = -99
        self.last_hurt = -99.0
        self.last_dealt = -99.0
        self.attackers = {}  # id -> (t, bearing)
        self.enemies = {}  # id -> last seen dict + t
        self.npcs = {}  # id -> (pos, t)
        self.sound_threat = []  # (t, bearing, distclass, kind)
        self.strafe_dir = 1
        self.strafe_t = 0
        self.stuck_t = 0.0
        self.unstick_until = -1
        self.unstick_dir = None
        self.last_pos = None
        self.cover = None
        self.cover_t = -99
        self.scan_yaw = 0.0
        self.bad_pickups = {}
        self.bad_crates = {}
        self.t = 0
        self.mode = ""
        self.cubes_lost = {}
        self.fight_mode = None
        self.duel_state = ""
        self.flee_hard = False

    # ================================================================== helpers
    def zone_at(self, z, t):
        s0, s1 = z["shrink_start"], z["shrink_end"]
        if t <= s0:
            return z["center"], z["radius"]
        if t >= s1 or s1 <= s0:
            return z["next_center"], z["next_radius"]
        f = (t - s0) / (s1 - s0)
        c = z["center"]
        n = z["next_center"]
        return [c[0] + (n[0] - c[0]) * f, c[1] + (n[1] - c[1]) * f], z["radius"] + (z["next_radius"] - z["radius"]) * f

    def zone_ok(self, p, t, margin=0.0):
        c, r = self.zone_at(self.zone, t)
        return dist2(p, c) <= r - margin

    def h_at(self, x, y):
        return self.nm.height_at(x, y) if self.nm else 0.0

    def los(self, a, b):
        return self.nm.los(a, b) if self.nm else True

    def eye(self, p, cubes=0):
        return [p[0], p[1], p[2] + 8.0 * (1 + 0.05 * cubes)]

    def walk_speed(self):
        return self.me["stats"].get("walk_speed", 11.7)

    # ================================================================== main
    def act(self, obs):
        t0 = time.perf_counter()
        self.obs = obs
        self.t = t = obs["t"]
        self.me = me = obs["self"]
        self.zone = obs["zone"]
        self.pos = pos = me["pos"]
        act = {"tick": obs["tick"]}
        self._events(obs)
        self._track(obs)

        if obs.get("offer"):
            act["pick"] = self._pick_upgrade(obs["offer"])

        # ---------------- threats
        mechs = [m for m in obs["mechs"] if m["kind"] == "mech" and m["id"] != me["id"]]
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        for m in mechs:
            m["d"] = dist2(m["pos"], pos)
        for m in npcs:
            m["d"] = dist2(m["pos"], pos)
        mechs.sort(key=lambda m: m["d"])
        self.vis_mechs = mechs
        recent_att = {i: v for i, v in self.attackers.items() if t - v[0] < 4.0}
        hpf = me["hp"] / max(1.0, me["max_hp"])

        # ---------------- weapons: take/drop
        self._loot_take(obs, act, any(m["d"] < 80 or m["id"] in recent_att for m in mechs))

        # ---------------- choose a shooting target
        target, tkind = self._choose_target(mechs, npcs, obs, recent_att, hpf)

        # ---------------- movement decision
        move, boost, jump, label = self._decide_move(obs, mechs, npcs, recent_att, hpf, target)

        # ---------------- aim and fire
        aim, fire, reload = self._aim_fire(obs, target, tkind, mechs, recent_att, move)
        if AIM_NOISE:
            aim = [aim[0] + self.rng.gauss(0, AIM_NOISE), aim[1] + self.rng.gauss(0, AIM_NOISE * 0.5)]
        act["aim"] = aim
        act["fire"] = fire
        if reload:
            act["reload"] = True
        # laser charging slows us; if we must run, don't charge
        act["move"] = move
        act["boost"] = boost
        act["jump"] = jump
        self.last_pos = pos
        dt = (time.perf_counter() - t0) * 1000
        mf.draw.label(f"{label} hp{me['hp']:.0f} {dt:.0f}ms")
        if LOGF and self.slot == 0 and obs["tick"] % 10 == 0:
            with open(LOGF, "a") as f:
                f.write(f"t={t:.0f} {label} hp={me['hp']:.0f} e={me['energy']:.0f} pos=({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f}) v={math.hypot(*me['vel'][:2]):.1f} inz={me['in_zone']} "
                        f"mechs={[(m['id'], round(m['d']), round(m.get('hp', 0))) for m in mechs]} att={list(recent_att)} tgt={target and target.get('id')} fire={act['fire']} w={self.held_kinds()} cubes={me['cubes']} home={[round(v) for v in self.home]} {dt:.1f}ms path={len(self.path) if self.path else None}/{self.path_i} pend={[round(v) for v in self.nm.center(self.path[-1])] if self.path else None} mv={[round(v, 2) for v in act['move']]} unst={self.t < self.unstick_until}\n")
        return act

    # ================================================================== bookkeeping
    def _events(self, obs):
        t = self.t
        for e in obs.get("events") or []:
            ty = e.get("type")
            if ty == "damaged":
                self.last_hurt = t
                by = e.get("by")
                if by is not None and by < 100:
                    self.attackers[by] = (t, e.get("bearing"))
                elif by is not None:
                    self.attackers[by] = (t, e.get("bearing"))
            elif ty == "hit":
                self.last_dealt = t
            elif ty == "take_failed":
                self.bad_pickups[e.get("id")] = t
        for s in obs.get("sounds") or []:
            if s["kind"] in ("gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet", "reload"):
                self.sound_threat.append((t - s.get("age", 0), s["bearing"], s["dist"], s["kind"]))
        self.sound_threat = [s for s in self.sound_threat if t - s[0] < 6.0][-40:]

    def _track(self, obs):
        t = self.t
        for m in obs["mechs"]:
            if m["kind"] == "mech":
                self.enemies[m["id"]] = dict(m, seen=t)
            else:
                self.npcs[m["id"]] = (m["pos"], t)
        for k in [k for k, v in self.enemies.items() if t - v["seen"] > 25]:
            del self.enemies[k]

    def _pick_upgrade(self, offer):
        ch = offer.get("choices") or []
        best, bi = 1e9, 0
        held = {w["kind"] for w in self.me["weapons"] if w}
        for i, c in enumerate(ch):
            cid = c.get("id")
            r = UPGRADE_ORDER.index(cid) if cid in UPGRADE_ORDER else 50
            if cid.startswith("laser") and "laser" not in held:
                r += 30
            if r < best:
                best, bi = r, i
        return bi

    # ================================================================== loot
    def held_kinds(self):
        return [w["kind"] if w else None for w in self.me["weapons"]]

    def _pickup_gain(self, kind):
        """(gain, drop_slot) for taking this kind."""
        held = self.held_kinds()
        if kind in held:
            return 0, None
        v = WEAPON_VALUE.get(kind, 0)
        if v <= 0:
            return 0, None
        for i, k in enumerate(held):
            if k is None:
                return v, None
        # both full: replace the worst non-gun slot
        worst, wi = 1e9, None
        for i, k in enumerate(held):
            if k == "gun":
                continue
            if WEAPON_VALUE.get(k, 0) < worst:
                worst, wi = WEAPON_VALUE.get(k, 0), i
        if wi is not None and v > worst + 0.1:
            return v - worst, wi
        return 0, None

    def _loot_take(self, obs, act, threat):
        pos = self.pos
        best = None
        for p in obs.get("pickups") or []:
            if self.t - self.bad_pickups.get(p["id"], -99) < 3:
                continue
            if dist2(p["pos"], pos) > self.loot_r - 0.3 or abs(p["pos"][2] - pos[2]) > 3.8:
                continue
            gain, drop = self._pickup_gain(p["kind"])
            if gain <= 0:
                continue
            if drop is not None:
                w = self.me["weapons"][drop]
                if threat or (w and w.get("charge", 0) > 0):
                    continue
            if best is None or gain > best[0]:
                best = (gain, p["id"], drop)
        if best:
            act["take"] = best[1]
            if best[2] is not None:
                act["drop"] = best[2]

    # ================================================================== targeting
    def _choose_target(self, mechs, npcs, obs, recent_att, hpf):
        pos = self.pos
        me = self.me
        best, bs = None, -1e9
        for m in mechs:
            d = m["d"]
            if d > 240:
                continue
            s = -d
            if m["id"] in recent_att:
                s += 120
            hp = m.get("hp", 1400)
            if hp < 400:
                s += 60
            if not self._los_to(m):
                s -= 150
            if s > bs:
                bs, best = s, m
        if best is not None:
            engage = AGGRO or (best["id"] in recent_att or best["d"] < 55
                               or (self._late(obs) and best["d"] < 160)
                               or (best.get("hp", 1400) < min(350, me["hp"] * 0.6) and best["d"] < 220))
            if engage:
                return best, "mech"
        # NPC drones attacking us, or farmable
        for m in sorted(npcs, key=lambda m: m["d"]):
            if m["d"] < 90 and (m["id"] in recent_att or (hpf > 0.75 and not mechs and m["d"] < 70)):
                if self._los_to(m):
                    return m, "npc"
        # crates for cubes / weapons
        if not mechs and hpf > 0.85:
            bc, bd = None, 1e9
            for c in obs.get("crates") or []:
                d = dist2(c["pos"], pos)
                if d > 70 or self.t - self.bad_crates.get(c["id"], -99) < 8:
                    continue
                gz = self.h_at(c["pos"][0], c["pos"][1])
                if c["pos"][2] > self.nm.ground_at(c["pos"][0], c["pos"][1]) + 4 if self.nm else False:
                    continue  # rooftop crate: loot would be out of reach
                tgt = [c["pos"][0], c["pos"][1], c["pos"][2] + 1.2]
                if not self.los(self._muzzle(), tgt):
                    continue
                if d < bd:
                    bd, bc = d, c
            if bc is not None:
                return dict(bc, kind="crate", vel=[0, 0, 0]), "crate"
        return None, None

    def _late(self, obs):
        return obs["alive"] <= 3 or self.zone["next_radius"] <= 30

    def _can_see_me(self, m):
        """Could this mech see us right now (we're in its view wedge or proximity range, with line of sight)?"""
        a = m.get("aim") or [0, 0]
        yaw = math.atan2(self.pos[1] - m["pos"][1], self.pos[0] - m["pos"][0])
        if m["d"] > 45 and abs(wrap(a[0] - yaw)) > math.radians(self.cfg["vision"].get("fov_deg", 120)) / 2 + 0.15:
            return False
        e = self.eye(m["pos"], m.get("cubes", 0) or 0)
        p = self.pos
        return self.los(e, [p[0], p[1], p[2] + 8.5]) or self.los(e, [p[0], p[1], p[2] + 4.5])

    def _looking_at_me(self, m):
        a = m.get("aim") or [0, 0]
        yaw = math.atan2(self.pos[1] - m["pos"][1], self.pos[0] - m["pos"][0])
        return abs(wrap(a[0] - yaw)) < 0.25

    def _muzzle(self):
        p = self.pos
        return [p[0], p[1], p[2] + self.me["stats"].get("muzzle_height", 7)]

    def _tcenter(self, m):
        if m.get("kind") == "npc":
            return [m["pos"][0], m["pos"][1], m["pos"][2] + 1.75]
        if m.get("kind") == "crate":
            return [m["pos"][0], m["pos"][1], m["pos"][2] + 1.2]
        cubes = m.get("cubes", 0) or 0
        return [m["pos"][0], m["pos"][1], m["pos"][2] + 4.5 * (1 + 0.05 * cubes)]

    def _trad(self, m):
        if m.get("kind") == "npc":
            return 1.8
        if m.get("kind") == "crate":
            return 1.4
        return 2.5 * (1 + 0.05 * (m.get("cubes", 0) or 0))

    def _los_to(self, m):
        mz = self._muzzle()
        c = self._tcenter(m)
        if self.los(mz, c):
            return True
        top = [c[0], c[1], m["pos"][2] + (8.5 if m.get("kind") == "mech" else 3.0)]
        return self.los(mz, top)

    # ================================================================== aiming
    def _aim_fire(self, obs, target, tkind, mechs, recent_att, move):
        me = self.me
        pos = self.pos
        weapons = me["weapons"]
        fire = [False, False]
        reload = False
        aim = None
        cur = me["aim"]
        gun_i = laser_i = light_i = mortar_i = None
        for i, w in enumerate(weapons):
            if not w:
                continue
            k = w["kind"]
            if k == "gun":
                gun_i = i
            elif k == "laser":
                laser_i = i
            elif k == "lightning":
                light_i = i
            elif k == "mortar":
                mortar_i = i
        muz = self._muzzle()
        if target is not None:
            c = self._tcenter(target)
            vel = target.get("vel") or [0, 0, 0]
            d3 = math.dist(muz, c)
            # gun lead
            gstats = weapons[gun_i]["stats"] if gun_i is not None else self.W["gun"]
            spd = gstats.get("speed", 252)
            tt = d3 / spd
            for _ in range(3):
                p = [c[0] + vel[0] * tt, c[1] + vel[1] * tt, c[2] + (vel[2] * tt if abs(vel[2]) > 1 else 0)]
                tt = math.dist(muz, p) / spd
            p[2] = max(p[2], self.h_at(p[0], p[1]) + 1.0) if tkind == "mech" else p[2]
            want = self._dir(muz, p)
            # laser: hitscan, tiny lead
            lp = [c[0] + vel[0] * 0.1, c[1] + vel[1] * 0.1, c[2]]
            lwant = self._dir(muz, lp)
            use_laser = laser_i is not None and d3 < weapons[laser_i]["stats"].get("range", 280) * 0.97
            charging = laser_i is not None and weapons[laser_i].get("charge", 0) > 0
            aim = lwant if (charging or (use_laser and weapons[laser_i]["cooldown"] < 0.3 and (gun_i is None or d3 > 60))) else want
            rad = self._trad(target)
            tol = math.atan2(rad * 0.85, max(1.0, d3)) + 0.004
            los_ok = self._los_to(target)
            err = self._angerr(cur, want)
            if gun_i is not None and los_ok:
                g = weapons[gun_i]
                if d3 < gstats.get("range", 230) * 0.98 and err < tol and g.get("reload", 0) == 0 and g.get("ammo", 1) > 0:
                    fire[gun_i] = True
            if laser_i is not None:
                L = weapons[laser_i]
                ls = L["stats"]
                full = ls.get("charge", 0.75)
                ch = L.get("charge", 0)
                lerr = self._angerr(cur, lwant)
                ltol = math.atan2(rad * 0.7, max(1.0, d3)) + 0.002
                if ch > 0:
                    # holding: release when full and on target (or when it would auto-fire anyway)
                    release = (ch >= full - 0.02 and lerr < ltol and los_ok) or ch > full + ls.get("hold_s", 2) - 0.15
                    if not release and not los_ok and ch >= full:
                        release = ch > full + 1.0
                    fire[laser_i] = not release
                elif use_laser and L["cooldown"] <= 0.1 and tkind in ("mech", "npc") and not self.flee_hard:
                    precharge = tkind == "mech" and self.fight_mode in ("fight", "cover") and d3 < 250
                    if (los_ok and lerr < 0.25) or precharge:
                        fire[laser_i] = True
            if light_i is not None:
                Ls = weapons[light_i]["stats"]
                yaw_err = abs(wrap(cur[0] - math.atan2(c[1] - muz[1], c[0] - muz[0])))
                if d3 < Ls.get("range", 45) * 0.95 and yaw_err < math.radians(Ls.get("cone_deg", 30)) * 0.4 and los_ok and me["energy"] > Ls.get("energy", 3) + 4:
                    fire[light_i] = True
            if mortar_i is not None and tkind == "mech":
                M = weapons[mortar_i]
                if M["cooldown"] <= 0 and 30 < d3 < 210:
                    sol = self._mortar_pitch(muz, c, vel, M["stats"])
                    if sol is not None and not fire[gun_i or 0]:
                        myaw, mp = sol
                        if self._angerr(cur, [myaw, mp]) < 0.03:
                            fire[mortar_i] = True
            if tkind == "crate" and gun_i is not None:
                if obs.get("events"):
                    pass
            # gun reload discipline
            if gun_i is not None:
                g = weapons[gun_i]
                if g.get("ammo", 1) == 0 and g.get("reload", 0) == 0:
                    reload = True
        else:
            # no target: release any laser charge safely only if it will waste; keep holding otherwise
            if laser_i is not None and weapons[laser_i].get("charge", 0) > 0:
                fire[laser_i] = weapons[laser_i]["charge"] < weapons[laser_i]["stats"].get("charge", 0.75) + 1.5
            if gun_i is not None:
                g = weapons[gun_i]
                if g.get("ammo", 0) < g["stats"].get("mag", 12) * 0.6 and g.get("reload", 0) == 0 and self.t - self.last_dealt > 1.5:
                    reload = True
            aim = self._idle_aim(recent_att, move)
        return aim, fire, reload

    def _dir(self, a, b):
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(dz, math.hypot(dx, dy))
        mn, mx = self.me["stats"].get("min_pitch", -0.7), self.me["stats"].get("max_pitch", 1.3)
        return [yaw, max(mn, min(mx, pitch))]

    def _angerr(self, a, b):
        ca = [math.cos(a[1]) * math.cos(a[0]), math.cos(a[1]) * math.sin(a[0]), math.sin(a[1])]
        cb = [math.cos(b[1]) * math.cos(b[0]), math.cos(b[1]) * math.sin(b[0]), math.sin(b[1])]
        d = max(-1.0, min(1.0, ca[0] * cb[0] + ca[1] * cb[1] + ca[2] * cb[2]))
        return math.acos(d)

    def _mortar_pitch(self, muz, c, vel, st):
        v = st.get("speed", 72)
        g = st.get("gravity", 22)
        p = list(c)
        for _ in range(3):
            d = math.hypot(p[0] - muz[0], p[1] - muz[1])
            h = (c[2] - 4.0) - muz[2]
            disc = v ** 4 - g * (g * d * d + 2 * h * v * v)
            if disc < 0 or d < 1:
                return None
            th = math.atan2(v * v - math.sqrt(disc), g * d)
            tf = d / (v * math.cos(th))
            p = [c[0] + vel[0] * tf, c[1] + vel[1] * tf, c[2]]
        return math.atan2(p[1] - muz[1], p[0] - muz[0]), th

    def _idle_aim(self, recent_att, move):
        t = self.t
        # face whoever is hurting us
        best = None
        for i, (ta, b) in recent_att.items():
            if b is None:
                continue
            if best is None or ta > best[0]:
                best = (ta, b)
        if best is not None and t - best[0] < 3:
            return [best[1], 0.0]
        # remembered enemies (not visible right now)
        rec = [e for e in self.enemies.values() if t - e["seen"] < 8 and dist2(e["pos"], self.pos) < 200]
        if rec:
            e = min(rec, key=lambda e: dist2(e["pos"], self.pos))
            return [math.atan2(e["pos"][1] - self.pos[1], e["pos"][0] - self.pos[0]), 0.0]
        # recent threatening sounds
        snd = [s for s in self.sound_threat if t - s[0] < 2.5 and s[2] in ("near", "mid")]
        if snd:
            s = snd[-1]
            return [s[1], 0.0]
        # moving: look where we go (with a slow sweep), holding: sweep
        if move and (move[0] or move[1]):
            yaw = math.atan2(move[1], move[0])
            return [yaw + 0.7 * math.sin(t * 0.9), 0.0]
        self.scan_yaw = wrap(self.scan_yaw + 0.12)
        return [self.scan_yaw, 0.0]

    # ================================================================== movement
    def _decide_move(self, obs, mechs, npcs, recent_att, hpf, target):
        t = self.t
        pos = self.pos
        me = self.me
        z = self.zone
        self.flee_hard = False
        wsp = self.walk_speed()
        boost = False
        jump = False
        energy = me["energy"]
        # ---- zone urgency
        goal_home = self._home()
        in_zone = me.get("in_zone", True)
        tleft = self._time_until_zone_hits(pos)
        f = getattr(self, "field", None)
        if f is not None and not f.done:
            f.step(0.008)
        dhome = self._home_dist()
        urgent = (not in_zone) or tleft < dhome / wsp + 14
        self.was_urgent = urgent
        if AGGRO:
            rec = [e for e in self.enemies.values() if t - e["seen"] < 10 and self.zone_ok(e["pos"], t + 10, 10)]
            if rec:
                e = min(rec, key=lambda e: dist2(e["pos"], pos))
                goal_home = [e["pos"][0], e["pos"][1]]
                dhome = 0
        att_mech = {i for i in recent_att if i < 100}
        late = self._late(obs)
        weak = [m for m in mechs if m["d"] < 220 and m.get("hp", 1400) < min(350, me["hp"] * 0.6)]
        engaged = [m for m in mechs if m["id"] in att_mech or m["d"] < 55 or (late and m["d"] < 160) or m in weak]
        if AGGRO:
            engaged = [m for m in mechs if m["d"] < 240]
        watchers = [m for m in mechs if m not in engaged and m["d"] < 120 and self._can_see_me(m)]
        mech_threats = engaged or watchers
        fight_mode = None
        my = me["hp"]
        if engaged:
            primary = engaged[0]
            php = primary.get("hp", 1400)
            tot = sum(m.get("hp", 1400) for m in engaged if m["d"] < 150)
            flee = ((my < 0.3 * me["max_hp"] and php > my)
                    or (len([m for m in engaged if m["d"] < 150]) >= 2 and my < tot * 0.9)
                    or (my < php * 0.7 and php > 300))
            if late and len(engaged) == 1 and my > php * 0.8:
                flee = False
            if primary in weak and len([m for m in engaged if m["d"] < 150]) <= 1:
                flee = False
            if AGGRO:
                flee = my < 0.2 * me["max_hp"]
            if flee:
                fight_mode = "flee"
            elif primary["d"] > 120 and not late and not AGGRO and primary not in weak:
                fight_mode = "cover"
            else:
                fight_mode = "fight"
        elif att_mech:
            fight_mode = "evade"  # hurt by someone we can't see
        elif watchers:
            fight_mode = "avoid"
        self.fight_mode = fight_mode
        self.duel_state = ""

        dodge = self._dodge(obs)
        label = ""
        if not in_zone:
            boost = energy > 15
        if fight_mode == "flee":
            self.flee_hard = True
            goal = self._cover_point(mech_threats, urgent)
            label = "flee"
            mv = self._go(goal, "cover")
            if energy > 12:
                boost = True
        elif fight_mode in ("cover", "avoid", "evade") and urgent:
            label = "zone!"
            mv = self._go(goal_home, "home")
            if energy > 30:
                boost = True
        elif fight_mode in ("cover", "avoid"):
            label = fight_mode
            self.flee_hard = fight_mode == "cover"
            goal = self._cover_point(mech_threats, urgent)
            mv = self._go(goal, "cover")
        elif fight_mode == "fight" and mech_threats:
            label = "fight"
            mv = self._duel_move(mech_threats[0], urgent, goal_home)
            label = self.duel_state
        elif fight_mode == "evade":
            label = "evade"
            # move to cover from the bearing we were hit from
            mv = self._go(goal_home, "home") if urgent else self._evade_move(recent_att)
        else:
            mv = None
            if urgent:
                label = "zone"
                mv = self._go(goal_home, "home")
                if tleft < dhome / wsp + 2 and energy > 40:
                    boost = True
            if mv is None:
                g = self._drop_goal(obs, hpf)
                if g is not None:
                    label = "drop"
                    mv = self._go(g, "drop")
                    if dist2(pos, g) > 30 and energy > 50:
                        boost = True
            if mv is None and hpf < 0.8 and not AGGRO:
                label = "rest"
                mv = [0.0, 0.0]
            if mv is None:
                g = self._loot_goal(obs, mechs, hpf, target)
                if g is not None:
                    label = "loot"
                    mv = self._go(g, "loot")
            if mv is None:
                if dist2(pos, goal_home) > 6:
                    label = "home"
                    mv = self._go(goal_home, "home")
                    # far from danger and not hurried: walk quietly when close
                    if dist2(pos, goal_home) < 40 and mv:
                        mv = [mv[0] * 0.4, mv[1] * 0.4]
                else:
                    label = "hold"
                    mv = [0.0, 0.0]
        if dodge is not None:
            mv = [mv[0] * 0.4 + dodge[0], mv[1] * 0.4 + dodge[1]]
            n = math.hypot(*mv)
            if n > 1:
                mv = [mv[0] / n, mv[1] / n]
            if energy > 30:
                boost = True
        mv, j2, b2 = self._unstick(mv)
        jump = jump or j2
        boost = boost or b2
        # laser charge slows us; flee overrides it (handled in aim)
        return mv, boost, jump, label

    def _time_until_zone_hits(self, p):
        z = self.zone
        t = self.t
        if not self.zone_ok(p, t, 3):
            return 0.0
        end = max(z["shrink_end"], t)
        # sample forward
        tt = t
        step = 2.0
        while tt < end:
            tt += step
            if not self.zone_ok(p, min(tt, end), 3):
                return tt - t
        # after the next circle is reached, the following phase: assume ~30 s hold + shrink toward unknown
        if dist2(p, z["next_center"]) > z["next_radius"] * 0.6:
            return max(0.0, end - t) + 30.0
        return max(0.0, end - t) + 60.0

    def _home(self):
        """A quiet spot inside the next circle, near its center but not on it."""
        z = self.zone
        key = (round(z["next_center"][0], 1), round(z["next_center"][1], 1), round(z["next_radius"], 1))
        t = self.t
        need = self.home is None or key != self.home_key
        if not need and self.home is not None and t - self.home_t > 20 and not getattr(self, "was_urgent", False):
            # move away if enemies are known near it
            for e in self.enemies.values():
                if t - e["seen"] < 5 and dist2(e["pos"], self.home) < 45:
                    need = True
                    break
        if need:
            self.home_key = key
            self.home_t = t
            self.home = self._pick_home(z["next_center"], z["next_radius"])
            self.field = None
            if self.nm is not None:
                hc = self.nm.nearest_reach(self.home[0], self.home[1])
                if hc is not None:
                    self.field = Field(self.nm, hc)
        return self.home

    def _home_dist(self):
        """Walking distance to home (field when ready, else a straight-line guess)."""
        straight = dist2(self.pos, self.home) * 1.5
        f = getattr(self, "field", None)
        if f is None or not f.done:
            return straight
        c = self._my_cell()
        d = f.d(c) if c is not None else None
        return d if d is not None else straight

    def _my_cell(self):
        nm = self.nm
        c = nm.cell(self.pos[0], self.pos[1])
        if nm.blocked[c]:
            c = nm.nearest_walkable(self.pos[0], self.pos[1], 3)
        return c

    def _field_move(self):
        nm = self.nm
        f = self.field
        c = self._my_cell()
        if c is None or f.d(c) is None:
            return None
        gx = nm.gx
        chain = [c]
        cur = c
        dist = f.dist
        for _ in range(16):
            ci, cj = cur % gx, cur // gx
            best, bd = None, dist[cur]
            for di, dj, _dc in AStar.NB:
                ii, jj = ci + di, cj + dj
                if 0 <= ii < gx and 0 <= jj < nm.gy:
                    v = jj * gx + ii
                    if dist[v] < bd and nm.can_step(cur, v):
                        bd, best = dist[v], v
            if best is None:
                break
            chain.append(best)
            cur = best
        pos = self.pos
        tgt = None
        for v in reversed(chain[1:]):
            x, y = nm.center(v)
            if nm.straight_ok(pos[0], pos[1], x, y):
                tgt = (x, y)
                break
        if tgt is None:
            if len(chain) < 2:
                return None
            tgt = nm.center(chain[1])
        dx, dy = tgt[0] - pos[0], tgt[1] - pos[1]
        n = math.hypot(dx, dy)
        if n < 0.3:
            return None
        return [dx / n, dy / n]

    def _pick_home(self, nc, nr):
        nm = self.nm
        pos = self.pos
        if nm is None:
            return list(nc)
        lim = max(8.0, min(nr * 0.5, nr - 12))
        best, bs = None, 1e18
        cand = []
        for _ in range(260):
            a = self.rng.uniform(0, TAU)
            if HOME_CENTER or nr < 40:
                r = lim * math.sqrt(self.rng.random())
            else:
                r = nr * self.rng.uniform(0.35, 0.78)
            x, y = nc[0] + r * math.cos(a), nc[1] + r * math.sin(a)
            if not nm.in_bounds(x, y):
                continue
            c = nm.cell(x, y)
            if not nm.reach[c]:
                continue
            cand.append(c)
        scored = []
        for c in cand:
            x, y = nm.center(c)
            s = math.hypot(x - pos[0], y - pos[1]) + (0.5 if HOME_CENTER else 0.15) * math.hypot(x - nc[0], y - nc[1])
            s -= 6 * min(3, nm.cover[c])
            s -= 10 * nm.near[c]
            for e in self.enemies.values():
                if self.t - e["seen"] < 15:
                    de = math.hypot(x - e["pos"][0], y - e["pos"][1])
                    if de < 80:
                        s += (80 - de) * 3
            scored.append((s, c))
        scored.sort()
        for s, c in scored[:40]:
            x, y = nm.center(c)
            z = nm.h[c] + 8.0
            op = 0
            for k in range(12):
                a = k * TAU / 12
                if nm.los([x, y, z], [x + 70 * math.cos(a), y + 70 * math.sin(a), z], 3.0):
                    op += 1
            s += 7 * op
            if s < bs:
                bs, best = s, (x, y)
        if best is None:
            c = nm.nearest_reach(nc[0], nc[1])
            best = nm.center(c) if c is not None else tuple(nc)
        return [best[0], best[1]]

    def _duel_move(self, m, urgent, home):
        """Hit and hide: show ourselves only while a shot is ready, duck behind cover while reloading."""
        me = self.me
        laser = gun = None
        for w in me["weapons"]:
            if w and w["kind"] == "laser":
                laser = w
            elif w and w["kind"] == "gun":
                gun = w
        d = m["d"]
        ready = False
        if laser is not None:
            full = laser["stats"].get("charge", 0.75)
            ready = laser.get("charge", 0) >= full * 0.8
        gun_ok = gun is not None and gun.get("reload", 0) == 0 and gun.get("ammo", 0) > 2
        if laser is None:
            ready = gun_ok
        else:
            ready = ready or (gun_ok and laser["cooldown"] < 0.3)
        php = m.get("hp", 1400)
        # brawl when close, when they're nearly dead, or when there's no time to play cover games
        if AGGRO or NODUEL or d < 35 or urgent or php < 350 or ("lightning" in self.held_kinds() and d < 45):
            self.duel_state = "brawl"
            return self._fight_move(m, urgent, home)
        if ready:
            self.duel_state = "peek"
            return self._fight_move(m, urgent, home)
        self.duel_state = "duck"
        self.cover_t = min(self.cover_t, self.t - 0.6)  # refresh cover often while dueling
        goal = self._cover_point([m], urgent)
        return self._go(goal, "cover") or [0.0, 0.0]

    def _fight_move(self, m, urgent, home):
        t = self.t
        pos = self.pos
        if t > self.strafe_t:
            self.strafe_dir = -self.strafe_dir if self.rng.random() < 0.75 else self.strafe_dir
            self.strafe_t = t + self.rng.uniform(0.5, 1.3)
        dx, dy = m["pos"][0] - pos[0], m["pos"][1] - pos[1]
        d = math.hypot(dx, dy) or 1.0
        ux, uy = dx / d, dy / d
        px, py = -uy * self.strafe_dir, ux * self.strafe_dir
        held = self.held_kinds()
        enemy_w = m.get("weapons") or []
        if "lightning" in held and "lightning" not in enemy_w:
            lo, hi = 18, 38
        else:
            lo, hi = 55, 130
            if "lightning" in enemy_w:
                lo = 60
        radial = 0.0
        if d < lo:
            radial = -0.8
        elif d > hi:
            radial = 0.5
        mx, my = px + ux * radial, py + uy * radial
        # stay in the zone: bias toward home when needed
        if urgent or not self.zone_ok([pos[0] + mx * 8, pos[1] + my * 8], self.t + 3, 8):
            hx, hy = home[0] - pos[0], home[1] - pos[1]
            hn = math.hypot(hx, hy) or 1.0
            mx, my = mx * 0.4 + hx / hn, my * 0.4 + hy / hn
        n = math.hypot(mx, my) or 1.0
        mx, my = mx / n, my / n
        # don't strafe into walls
        if self.nm and not self.nm.straight_ok(pos[0], pos[1], pos[0] + mx * 6, pos[1] + my * 6):
            self.strafe_dir = -self.strafe_dir
            mx, my = -uy * self.strafe_dir + ux * radial, ux * self.strafe_dir + uy * radial
            n = math.hypot(mx, my) or 1.0
            mx, my = mx / n, my / n
            if not self.nm.straight_ok(pos[0], pos[1], pos[0] + mx * 6, pos[1] + my * 6):
                g = self._go(home, "home")
                return g or [0.0, 0.0]
        return [mx, my]

    def _evade_move(self, recent_att):
        b = None
        for i, (ta, bb) in recent_att.items():
            if bb is not None:
                b = bb
        if b is None:
            return self._go(self._home(), "home") or [0, 0]
        fake = [{"pos": [self.pos[0] + 80 * math.cos(b), self.pos[1] + 80 * math.sin(b), self.pos[2]], "d": 80}]
        g = self._cover_point(fake, False)
        return self._go(g, "cover") or [0.0, 0.0]

    def _cover_point(self, threats, urgent):
        t = self.t
        if self.cover is not None and t - self.cover_t < 1.2:
            return self.cover
        pos = self.pos
        nm = self.nm
        if nm is None:
            m = threats[0]
            dx, dy = pos[0] - m["pos"][0], pos[1] - m["pos"][1]
            n = math.hypot(dx, dy) or 1
            return [pos[0] + dx / n * 30, pos[1] + dy / n * 30]
        eyes = [self.eye(m["pos"], m.get("cubes", 0) or 0) for m in threats[:3]]
        best, bs = None, 1e18
        for r in (10, 20, 32, 46):
            for k in range(16):
                a = k * TAU / 16
                x, y = pos[0] + r * math.cos(a), pos[1] + r * math.sin(a)
                if not nm.in_bounds(x, y):
                    continue
                c = nm.cell(x, y)
                if not nm.reach[c]:
                    continue
                hz = nm.h[c]
                vis = 0
                for e in eyes:
                    if nm.los(e, [x, y, hz + 8.5]) or nm.los(e, [x, y, hz + 4.5]):
                        vis += 1
                s = r * 1.0 + vis * 120
                for m in threats[:3]:
                    s -= 0.4 * (math.hypot(x - m["pos"][0], y - m["pos"][1]) - m["d"])
                if not self.zone_ok([x, y], t + 8, 6):
                    s += 1000
                if not nm.straight_ok(pos[0], pos[1], x, y):
                    s += 25
                if s < bs:
                    bs, best = s, [x, y]
        if best is None:
            best = list(self._home())
        self.cover = best
        self.cover_t = t
        return best

    def _dodge(self, obs):
        """Sidestep incoming bullets, shells and telegraphed lasers."""
        pos = self.pos
        me = self.me
        cz = pos[2] + 4.5
        best = None
        for p in obs.get("projectiles") or []:
            pp, v = p["pos"], p["vel"]
            if p["kind"] == "mortar":
                land = self._shell_land(pp, v)
                if land is not None and dist2(land[0], pos) < 10 and land[1] < 3.0:
                    dx, dy = pos[0] - land[0][0], pos[1] - land[0][1]
                    n = math.hypot(dx, dy) or 1.0
                    return [dx / n, dy / n]
                continue
            vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
            if vv < 1:
                continue
            rx, ry, rz = pos[0] - pp[0], pos[1] - pp[1], cz - pp[2]
            tc = (rx * v[0] + ry * v[1] + rz * v[2]) / vv
            if tc < 0 or tc > 0.7:
                continue
            cx, cy = pp[0] + v[0] * tc - pos[0], pp[1] + v[1] * tc - pos[1]
            miss = math.hypot(cx, cy)
            if miss < 3.5:
                # move perpendicular to the bullet, away from its line
                sp = math.hypot(v[0], v[1]) or 1
                px, py = -v[1] / sp, v[0] / sp
                if px * (-cx) + py * (-cy) < 0:
                    px, py = -px, -py
                best = [px, py]
        if best is not None:
            return best
        for m in self.vis_mechs:
            if m.get("charging") and m["d"] < 280 and self._looking_at_me(m) and abs(wrap(m["aim"][0] - math.atan2(pos[1] - m["pos"][1], pos[0] - m["pos"][0]))) < 0.08:
                dx, dy = pos[0] - m["pos"][0], pos[1] - m["pos"][1]
                n = math.hypot(dx, dy) or 1
                s = self.strafe_dir
                return [-dy / n * s, dx / n * s]
        return None

    def _shell_land(self, p, v):
        g = self.W["mortar"].get("gravity", 22)
        x, y, z = p
        vx, vy, vz = v
        tt = 0.0
        while tt < 6:
            tt += 0.1
            x += vx * 0.1
            y += vy * 0.1
            vz -= g * 0.1
            z += vz * 0.1
            if z < self.h_at(x, y):
                return (x, y), tt
        return None

    def _drop_goal(self, obs, hpf):
        t = self.t
        pos = self.pos
        best = None
        for d in obs.get("drops") or []:
            dp = d["pos"]
            dd = dist2(dp, pos)
            eta = dd * 1.3 / self.walk_speed()
            if dd > 200:
                continue
            if d.get("holder") is not None and d["holder"] != self.me["id"] and dd > 20:
                continue
            if not self.zone_ok(dp, t + eta + 5, 5):
                continue
            # enemies near it?
            busy = False
            for e in self.enemies.values():
                if t - e["seen"] < 6 and e["id"] != self.me["id"] and dist2(e["pos"], dp) < 50:
                    busy = True
            if busy:
                continue
            if d["lands_in"] > eta + 12:
                continue  # too early: wait elsewhere
            if hpf < 0.25 and dd > 80:
                continue
            if best is None or dd < best[0]:
                best = (dd, dp)
        if best is None:
            return None
        dp = best[1]
        return [dp[0], dp[1]]

    def _loot_goal(self, obs, mechs, hpf, target):
        if mechs and any(m["d"] < 150 for m in mechs):
            return None
        pos = self.pos
        t = self.t
        nm = self.nm
        best, bs = None, 1e9
        for c in obs.get("cubes") or []:
            d = dist2(c["pos"], pos)
            if d > 90 or abs(c["pos"][2] - pos[2]) > 6 and (nm and c["pos"][2] > nm.ground_at(c["pos"][0], c["pos"][1]) + 3):
                continue
            if not self.zone_ok(c["pos"], t + d / 10 + 5, 4):
                continue
            s = d - 20
            if s < bs:
                bs, best = s, c["pos"]
        for p in obs.get("pickups") or []:
            if t - self.bad_pickups.get(p["id"], -99) < 3:
                continue
            gain, drop = self._pickup_gain(p["kind"])
            if gain <= 0:
                continue
            d = dist2(p["pos"], pos)
            if d > 110:
                continue
            if nm and p["pos"][2] > nm.ground_at(p["pos"][0], p["pos"][1]) + 3:
                continue  # on a roof
            if not self.zone_ok(p["pos"], t + d / 10 + 5, 4):
                continue
            s = d - 25 * gain
            if s < bs:
                bs, best = s, p["pos"]
        # crate being shot: approach a little if far
        if best is None and target is not None and target.get("kind") == "crate":
            d = dist2(target["pos"], pos)
            if d > 40:
                best = target["pos"]
        if best is None:
            return None
        return [best[0], best[1]]

    # ================================================================== path following
    def _go(self, goal, kind):
        """Return a unit move vector toward goal via A*."""
        nm = self.nm
        pos = self.pos
        t = self.t
        if goal is None:
            return None
        if nm is None:
            dx, dy = goal[0] - pos[0], goal[1] - pos[1]
            n = math.hypot(dx, dy)
            return [dx / n, dy / n] if n > 0.5 else [0.0, 0.0]
        gd = dist2(goal, pos)
        if gd < 1.5:
            return [0.0, 0.0]
        # straight shot when clear and close
        if gd < 40 and nm.straight_ok(pos[0], pos[1], goal[0], goal[1]):
            dx, dy = goal[0] - pos[0], goal[1] - pos[1]
            sc = 1.0 if gd > 4 else gd / 4
            return [dx / gd * sc, dy / gd * sc]
        if kind == "home" and getattr(self, "field", None) is not None and self.field.done and goal is self.home:
            mv = self._field_move()
            if mv is not None:
                return mv
        gc = nm.nearest_reach(goal[0], goal[1])
        if gc is None:
            gc = nm.cell(goal[0], goal[1])
        need = (self.path_goal is None or dist2(nm.center(gc), nm.center(self.path_goal)) > 5
                or (self.path is None and self.search is None) or t - self.path_t > 4.0)
        if need and self.search is None:
            sc = nm.cell(pos[0], pos[1])
            if nm.blocked[sc]:
                s2 = nm.nearest_walkable(pos[0], pos[1], 4)
                sc = s2 if s2 is not None else sc
            self.search = AStar(nm, sc, gc, cost_fn=self._danger_cost_fn())
            self.search_goal = gc
            self.path_t = t
        if self.search is not None:
            if self.search.step(0.012):
                if self.search.path and len(self.search.path) > 1:
                    self.path = self.search.path
                    self.path_i = 0
                    self.path_goal = self.search_goal
                self.search = None
        if self.path:
            return self._follow()
        dx, dy = goal[0] - pos[0], goal[1] - pos[1]
        return [dx / gd, dy / gd]

    def _danger_cost_fn(self):
        nm = self.nm
        t = self.t
        pts = []
        for e in self.enemies.values():
            if t - e["seen"] < 12:
                pts.append((e["pos"][0], e["pos"][1], 70.0, 6.0))
        for i, (p, ts) in self.npcs.items():
            if t - ts < 60:
                pts.append((p[0], p[1], 40.0, 3.0))
        if not pts:
            return None
        cache = {}

        def f(c):
            v = cache.get(c)
            if v is not None:
                return v
            x, y = nm.center(c)
            s = 0.0
            for px, py, r, w in pts:
                dx, dy = x - px, y - py
                if dx * dx + dy * dy < r * r:
                    s += w
            cache[c] = s
            return s
        return f

    def _follow(self):
        nm = self.nm
        pos = self.pos
        path = self.path
        # advance along the path
        best_i, best_d = self.path_i, 1e9
        for i in range(self.path_i, min(len(path), self.path_i + 30)):
            x, y = nm.center(path[i])
            d = math.hypot(x - pos[0], y - pos[1])
            if d < best_d:
                best_d, best_i = d, i
        self.path_i = best_i
        if best_d > 12:
            self.path_goal = None  # off path: replan next time
        # look ahead for the farthest cell we can walk straight to
        tgt = None
        hi = min(len(path) - 1, best_i + 14)
        for i in range(hi, best_i, -1):
            x, y = nm.center(path[i])
            if nm.straight_ok(pos[0], pos[1], x, y):
                tgt = (x, y)
                break
        if tgt is None:
            tgt = nm.center(path[min(len(path) - 1, best_i + 1)])
        dx, dy = tgt[0] - pos[0], tgt[1] - pos[1]
        n = math.hypot(dx, dy)
        if n < 0.3:
            if best_i >= len(path) - 1:
                return [0.0, 0.0]
            return [0.0, 0.0]
        return [dx / n, dy / n]

    def _unstick(self, mv):
        t = self.t
        me = self.me
        if t < self.unstick_until:
            d = self.unstick_dir or mv
            return d, me.get("grounded", False), me["energy"] > 10 and self.fight_mode is None
        want = math.hypot(mv[0], mv[1])
        sp = math.hypot(me["vel"][0], me["vel"][1])
        pm = getattr(self, "prev_mv", None) or [0.0, 0.0]
        self.prev_mv = mv
        steady = mv[0] * pm[0] + mv[1] * pm[1] > 0.8 * want * math.hypot(pm[0], pm[1]) and want > 0.5
        if steady and sp < 1.5 and me.get("grounded", True):
            self.stuck_t += 0.1
        else:
            self.stuck_t = max(0.0, self.stuck_t - 0.05)
        if self.stuck_t > 0.8:
            self.stuck_t = 0.0
            self.unstick_until = t + 0.9
            a = math.atan2(mv[1], mv[0]) + self.rng.choice([-1.2, 1.2, 0.0, 0.0])
            self.unstick_dir = [math.cos(a), math.sin(a)]
            self.path_goal = None
            return self.unstick_dir, True, self.fight_mode is None
        return mv, False, False


if __name__ == "__main__":
    mf.run(Bot)
