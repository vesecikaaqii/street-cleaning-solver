#!/usr/bin/env python3
import heapq

def parse_input(filename):
    with open(filename, 'r') as fh:
        lines = [l for l in fh.read().split('\n') if l.strip()]
    idx = 0
    hdr = lines[idx].split(); idx += 1
    N, M, T, C, depot = int(hdr[0]), int(hdr[1]), int(hdr[2]), int(hdr[3]), int(hdr[4])
    alpha = float(hdr[5])
    # Skip optional junction coordinate lines (2-field lines: lat lon)
    if len(lines[idx].split()) == 2:
        idx += N
    streets = []
    adj = [[] for _ in range(N)]
    for i in range(M):
        p = lines[idx].split(); idx += 1
        a, b, d, t = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        length, cat, req = int(p[4]), p[5], int(p[6])
        streets.append({'a': a, 'b': b, 'd': d, 't': t, 'l': length, 'cat': cat, 'req': req})
        adj[a].append((b, i, t))
        if d == 2:
            adj[b].append((a, i, t))
    vtypes = lines[idx].split()
    assert len(vtypes) == C, f"expected {C} vehicles, got {len(vtypes)}"
    return N, M, T, C, depot, alpha, streets, adj, vtypes


def dijkstra(src, adj, N):
    INF = float('inf')
    dist   = [INF] * N
    parent = [-1]  * N
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, _si, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v]   = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, parent


def path_to(src, dst, parent):
    """Reconstruct [src, ..., dst] using parent array from dijkstra(src)."""
    if src == dst:
        return [src]
    p = []
    cur = dst
    while cur != src:
        if cur == -1:
            return None
        p.append(cur)
        cur = parent[cur]
    p.append(src)
    p.reverse()
    return p


VCAP = {'S': 10, 'M': 20, 'L': 30}

def street_score(is_mandatory, st, cap, alpha, t_to):
    """
    Returns a float score (higher = better choice) or None to skip.

    Mandatory streets get base = 1e12 + urgency, where
    urgency = req * 1e8  (30e8 > 20e8 > 10e8).
    This guarantees req=30 is always preferred over req=20 over req=10
    for any vehicle that can clean them, keeping L vehicles focused on
    the hardest streets that no other type can handle.
    """
    waste  = (cap - st['req']) * st['l'] / 1000.0
    length = float(st['l'])

    if is_mandatory:
        urgency = st['req'] * 1e8        # 3e9 / 2e9 / 1e9
        base    = 1e12 + urgency
        # Among same urgency, prefer the nearest street (minimises travel overhead)
        return base - t_to

    # Optional
    if alpha >= 0.99:
        return length - t_to * 0.01
    if alpha <= 0.01:
        if waste > 0.0:
            return None             # skip optional streets with any waste
        return -t_to * 0.001
    return alpha * length - (1.0 - alpha) * waste * 1000.0 - t_to * 0.01



def solve_instance(filename, outfile):
    N, M, T, C, depot, alpha, streets, adj, vtypes = parse_input(filename)
    print(f"\n{'='*60}")
    print(f"{filename}  ->  {outfile}")
    print(f"N={N}  M={M}  T={T}  C={C}  depot={depot}  alpha={alpha}")
    print(f"Vehicles: {vtypes}")

    print(f"Dijkstra x{N} ...", end=' ', flush=True)
    AD = [None] * N   # AD[i] = dist array from node i
    AP = [None] * N   # AP[i] = parent array from node i
    for i in range(N):
        AD[i], AP[i] = dijkstra(i, adj, N)
    print("done.")

    mandatory  = [i for i, s in enumerate(streets) if s['cat'] == 'M']
    optional_s = [i for i, s in enumerate(streets) if s['cat'] == 'O']
    print(f"Mandatory={len(mandatory)}  Optional={len(optional_s)}")

    cleaned   = [False] * M
    caps      = [VCAP[v] for v in vtypes]
    v_cur     = [depot] * C
    v_time    = [0]     * C
    v_route   = [[depot] for _ in range(C)]
    v_cl      = [[]      for _ in range(C)]



    for vi in range(C):
        cap = caps[vi]

        while True:
            bsi = bapp = bdep = None
            bsc = float('-inf')
            btt = bts = None

            for is_mand in [True, False]:
                cands = mandatory if is_mand else optional_s

                for si in cands:
                    if cleaned[si]:
                        continue
                    st = streets[si]
                    if cap < st['req']:
                        continue

                    dirs = [(st['a'], st['b'])]
                    if st['d'] == 2:
                        dirs.append((st['b'], st['a']))

                    for app, dep in dirs:
                        t_to   = AD[v_cur[vi]][app]
                        t_st   = st['t']
                        t_back = AD[dep][depot]

                        if t_to == float('inf') or t_back == float('inf'):
                            continue
                        if v_time[vi] + t_to + t_st + t_back > T:
                            continue

                        sc = street_score(is_mand, st, cap, alpha, t_to)
                        if sc is None:
                            continue

                        if sc > bsc:
                            bsc = sc; bsi = si; bapp = app; bdep = dep
                            btt = t_to; bts = t_st

                if bsi is not None:
                    break       # found best at this priority level

            if bsi is None:
                break

            if v_cur[vi] != bapp:
                p = path_to(v_cur[vi], bapp, AP[v_cur[vi]])
                if p and len(p) > 1:
                    v_route[vi].extend(p[1:])

            v_route[vi].append(bdep)
            v_cl[vi].append(bsi)
            cleaned[bsi] = True
            v_time[vi]  += btt + bts
            v_cur[vi]    = bdep

        print(f"  V{vi:02d}({vtypes[vi]}): cleaned={len(v_cl[vi]):3d}  "
              f"time={v_time[vi]:6d}/{T}")

    # ------------------------------------------------------------------
    # Rescue pass — catch any mandatory streets the greedy missed
    # ------------------------------------------------------------------
    uncleaned_m = [si for si in mandatory if not cleaned[si]]
    if uncleaned_m:
        print(f"  Rescue pass: {len(uncleaned_m)} mandatory streets left")
        rescued = 0
        for si in uncleaned_m:
            st  = streets[si]
            bv  = None; bapp = None; bdep = None; btt = None
            bsc = float('-inf')

            for vi in range(C):
                if caps[vi] < st['req']:
                    continue
                dirs = [(st['a'], st['b'])]
                if st['d'] == 2:
                    dirs.append((st['b'], st['a']))
                for app, dep in dirs:
                    t_to   = AD[v_cur[vi]][app]
                    t_st   = st['t']
                    t_back = AD[dep][depot]
                    if t_to == float('inf') or t_back == float('inf'):
                        continue
                    rem = T - v_time[vi]
                    if t_to + t_st + t_back > rem:
                        continue
                    sc = rem - (t_to + t_st + t_back)   # prefer most slack
                    if sc > bsc:
                        bsc = sc; bv = vi; bapp = app; bdep = dep; btt = t_to

            if bv is not None:
                if v_cur[bv] != bapp:
                    p = path_to(v_cur[bv], bapp, AP[v_cur[bv]])
                    if p and len(p) > 1:
                        v_route[bv].extend(p[1:])
                v_route[bv].append(bdep)
                v_cl[bv].append(si)
                cleaned[si]  = True
                v_time[bv]  += btt + st['t']
                v_cur[bv]    = bdep
                rescued += 1
            else:
                print(f"    IMPOSSIBLE to rescue street {si} "
                      f"({st['a']}-{st['b']} req={st['req']})")

        print(f"  Rescued {rescued}/{len(uncleaned_m)}")

    # ------------------------------------------------------------------
    # Route every vehicle back to depot
    # ------------------------------------------------------------------
    for vi in range(C):
        if v_cur[vi] != depot:
            p = path_to(v_cur[vi], depot, AP[v_cur[vi]])
            if p and len(p) > 1:
                v_route[vi].extend(p[1:])

    # ------------------------------------------------------------------
    # Final diagnostics
    # ------------------------------------------------------------------
    still_uncleaned = [si for si in mandatory if not cleaned[si]]
    if still_uncleaned:
        print(f"  *** INVALID: {len(still_uncleaned)} mandatory streets uncleaned!")
    else:
        print(f"  All {len(mandatory)} mandatory streets cleaned.")
    print(f"  Optional cleaned: "
          f"{sum(1 for si in optional_s if cleaned[si])}/{len(optional_s)}")

    # ------------------------------------------------------------------
    # Write output  (n = steps = len(route) - 1)
    # ------------------------------------------------------------------
    with open(outfile, 'w', newline='\n') as fh:
        fh.write(f"{C}\n")
        for vi in range(C):
            junctions = v_route[vi]
            cl        = v_cl[vi]
            fh.write(f"{len(junctions) - 1}\n")
            fh.write(' '.join(map(str, junctions)) + '\n')
            fh.write((' '.join(map(str, cl)) + '\n') if cl else '\n')

    print(f"  Written: {outfile}")



if __name__ == '__main__':
    solve_instance('data/test_c.txt', 'output_c.txt')
    solve_instance('data/test_e.txt', 'output_e.txt')
    solve_instance('data/test_o.txt', 'output_o.txt')
    print("\nAll done.")
