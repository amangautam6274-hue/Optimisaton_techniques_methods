"""
Transportation Problem Solver
=============================
Implements:
  1. Vogel's Approximation Method (VAM)   -> Initial Basic Feasible Solution (BFS)
  2. MODI Method (Modified Distribution)  -> Optimality test + iterative improvement

Follows the flowchart logic:
  VAM  : balance check -> penalties -> highest penalty -> lowest cost cell in that
         row/col -> allocate min(supply,demand) -> update -> cross out exhausted
         row/col(s) -> repeat until fully allocated.
  MODI : count basic cells (m+n-1) -> fix degeneracy with epsilon if needed ->
         set u1=0 -> solve ui+vj=cij for basic cells -> compute Delta_ij = cij-(ui+vj)
         for non-basic cells -> if any Delta<0 pick most negative -> build closed
         loop -> theta = min allocation on '-' cells -> shift allocations -> repeat.

"""

from copy import deepcopy

EPS = 1e-9           # tolerance for float comparisons (must be « EPSILON_ALLOC)
EPSILON_ALLOC = 1e-6  # symbolic "almost zero" allocation used to break degeneracy


# ----------------------------------------------------------------------------- #
#  Small helper: Union-Find (used to detect / fix degeneracy for MODI)
# ----------------------------------------------------------------------------- #
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry
            return True
        return False


# ----------------------------------------------------------------------------- #
#  Main solver class
# ----------------------------------------------------------------------------- #
class TransportationProblem:
    def __init__(self, cost, supply, demand, row_names=None, col_names=None, verbose=True):
        """
        cost   : m x n list of lists -> cost[i][j] = cost of Oi -> Dj
        supply : list of length m
        demand : list of length n
        """
        self.orig_m = len(supply)
        self.orig_n = len(demand)
        self.cost = [row[:] for row in cost]
        self.supply = supply[:]
        self.demand = demand[:]
        self.verbose = verbose

        self.row_names = row_names[:] if row_names else [f"O{i+1}" for i in range(self.orig_m)]
        self.col_names = col_names[:] if col_names else [f"D{j+1}" for j in range(self.orig_n)]

        self.dummy_row_added = False
        self.dummy_col_added = False

        self._balance()

        self.m = len(self.supply)
        self.n = len(self.demand)

        self.allocation = None   # filled by vam()
        self.basic = None        # set of (i,j) that are basic cells (incl. epsilon ones)

    # ------------------------------------------------------------------ #
    # STEP 0 : Balance the problem  (VAM flowchart: "Are Supply = Demand?")
    # ------------------------------------------------------------------ #
    def _balance(self):
        total_supply = sum(self.supply)
        total_demand = sum(self.demand)

        self._log(f"Total supply = {total_supply}, Total demand = {total_demand}")

        if abs(total_supply - total_demand) < EPS:
            self._log("Supply = Demand -> problem is already balanced.\n")
            return

        if total_supply > total_demand:
            diff = total_supply - total_demand
            for row in self.cost:
                row.append(0)
            self.demand.append(diff)
            self.col_names.append("Dummy")
            self.dummy_col_added = True
            self._log(f"Supply > Demand -> added a DUMMY destination column "
                       f"with demand={diff} and cost=0.\n")
        else:
            diff = total_demand - total_supply
            self.cost.append([0] * len(self.demand))
            self.supply.append(diff)
            self.row_names.append("Dummy")
            self.dummy_row_added = True
            self._log(f"Demand > Supply -> added a DUMMY origin row "
                       f"with supply={diff} and cost=0.\n")

    def _log(self, msg=""):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------ #
    # STEP 1 : Vogel's Approximation Method -> initial BFS
    # ------------------------------------------------------------------ #
    def vam(self):
        m, n = self.m, self.n
        supply = self.supply[:]
        demand = self.demand[:]
        allocation = [[0] * n for _ in range(m)]

        active_rows = set(range(m))
        active_cols = set(range(n))

        self._log("=" * 70)
        self._log("STEP 1: VOGEL'S APPROXIMATION METHOD (VAM) - Initial BFS")
        self._log("=" * 70)

        iteration = 1
        while active_rows and active_cols:

            # If only one row or one column remains, allocate directly to all
            # remaining cells (no penalty computation needed / possible).
            if len(active_rows) == 1 or len(active_cols) == 1:
                for i in list(active_rows):
                    for j in list(active_cols):
                        qty = min(supply[i], demand[j])
                        if qty > 0:
                            allocation[i][j] += qty
                            supply[i] -= qty
                            demand[j] -= qty
                            self._log(f"[Final cell] Allocate {qty} to "
                                      f"({self.row_names[i]},{self.col_names[j]})")
                active_rows.clear()
                active_cols.clear()
                break

            # ---- Calculate penalties for every active ROW and COLUMN ----
            row_penalty = {}
            for i in active_rows:
                vals = sorted(self.cost[i][j] for j in active_cols)
                row_penalty[i] = vals[1] - vals[0] if len(vals) >= 2 else vals[0]

            col_penalty = {}
            for j in active_cols:
                vals = sorted(self.cost[i][j] for i in active_rows)
                col_penalty[j] = vals[1] - vals[0] if len(vals) >= 2 else vals[0]

            self._log(f"\n-- Iteration {iteration} --")
            self._log("Row penalties:    " +
                      ", ".join(f"{self.row_names[i]}={row_penalty[i]}" for i in sorted(active_rows)))
            self._log("Column penalties: " +
                      ", ".join(f"{self.col_names[j]}={col_penalty[j]}" for j in sorted(active_cols)))

            max_row_pen = max(row_penalty.values())
            max_col_pen = max(col_penalty.values())

            # ---- Find highest penalty (tie -> cheapest cell rule) ----
            if max_row_pen >= max_col_pen:
                tied_rows = [i for i in active_rows if row_penalty[i] == max_row_pen]
                # among tied rows, and also comparing overall, pick cell w/ lowest cost
                best = None
                for i in tied_rows:
                    for j in active_cols:
                        c = self.cost[i][j]
                        if best is None or c < best[0]:
                            best = (c, i, j)
                sel_i, sel_j = best[1], best[2]
            else:
                tied_cols = [j for j in active_cols if col_penalty[j] == max_col_pen]
                best = None
                for j in tied_cols:
                    for i in active_rows:
                        c = self.cost[i][j]
                        if best is None or c < best[0]:
                            best = (c, i, j)
                sel_i, sel_j = best[1], best[2]

            self._log(f"Highest penalty -> chosen cell ({self.row_names[sel_i]},"
                      f"{self.col_names[sel_j]}) [lowest cost in that row/col = "
                      f"{self.cost[sel_i][sel_j]}]")

            # ---- Allocate max possible quantity ----
            qty = min(supply[sel_i], demand[sel_j])
            allocation[sel_i][sel_j] += qty
            supply[sel_i] -= qty
            demand[sel_j] -= qty
            self._log(f"Allocate min(supply={supply[sel_i]+qty}, demand={demand[sel_j]+qty}) "
                      f"= {qty} units to ({self.row_names[sel_i]},{self.col_names[sel_j]})")

            # ---- Cross out exhausted row/column ----
            row_done = supply[sel_i] < EPS
            col_done = demand[sel_j] < EPS
            if row_done and col_done:
                # Per flowchart: if both become zero, cross out one of them.
                # We cross out both here (creates a degenerate BFS which MODI
                # will repair with an epsilon allocation) — this matches the
                # standard handling shown in the flowchart notes.
                active_rows.discard(sel_i)
                active_cols.discard(sel_j)
                self._log(f"Both {self.row_names[sel_i]} and {self.col_names[sel_j]} "
                          f"exhausted -> both crossed out (possible degeneracy).")
            elif row_done:
                active_rows.discard(sel_i)
                self._log(f"{self.row_names[sel_i]} supply exhausted -> row crossed out.")
            elif col_done:
                active_cols.discard(sel_j)
                self._log(f"{self.col_names[sel_j]} demand exhausted -> column crossed out.")

            iteration += 1

        self.allocation = allocation
        self.basic = {(i, j) for i in range(m) for j in range(n) if allocation[i][j] > EPS}

        self._log("\nInitial BFS obtained via VAM:")
        self.print_table(allocation)
        self._log(f"Initial VAM cost = {self.total_cost(allocation)}\n")
        return allocation

    # ------------------------------------------------------------------ #
    # STEP 2 : MODI method -> optimality test + improvement
    # ------------------------------------------------------------------ #
    def modi(self, max_iter=100):
        assert self.allocation is not None, "Run vam() first to get an initial BFS."

        allocation = deepcopy(self.allocation)
        m, n = self.m, self.n

        self._log("=" * 70)
        self._log("STEP 2: MODI METHOD - Optimality Test & Improvement")
        self._log("=" * 70)

        for it in range(1, max_iter + 1):
            self._log(f"\n-- MODI Iteration {it} --")

            basic = {(i, j) for i in range(m) for j in range(n) if allocation[i][j] > EPS}

            # ---- Degeneracy check: basic cells must equal m + n - 1 ----
            basic = self._fix_degeneracy(allocation, basic)

            # ---- Compute u_i, v_j  (set u_1 = 0) ----
            u = [None] * m
            v = [None] * n
            u[0] = 0
            solved = True
            # BFS/relaxation over the basic-cell graph until all u,v known
            changed = True
            passes = 0
            while changed and passes < (m + n) * 2:
                changed = False
                for (i, j) in basic:
                    if u[i] is not None and v[j] is None:
                        v[j] = self.cost[i][j] - u[i]
                        changed = True
                    elif v[j] is not None and u[i] is None:
                        u[i] = self.cost[i][j] - v[j]
                        changed = True
                passes += 1
            if any(x is None for x in u) or any(x is None for x in v):
                solved = False

            self._log("u (row multipliers): " +
                      ", ".join(f"u{i+1}={u[i]}" for i in range(m)))
            self._log("v (col multipliers): " +
                      ", ".join(f"v{j+1}={v[j]}" for j in range(n)))

            if not solved:
                # Shouldn't normally happen after degeneracy fix; bail out safely.
                self._log("Warning: could not solve all u,v (disconnected basis). "
                          "Stopping.")
                break

            # ---- Opportunity costs for non-basic cells ----
            delta = {}
            for i in range(m):
                for j in range(n):
                    if (i, j) not in basic:
                        delta[(i, j)] = self.cost[i][j] - (u[i] + v[j])

            self._log("Opportunity costs (Delta_ij) for non-basic cells:")
            for (i, j), d in sorted(delta.items()):
                self._log(f"  Delta[{self.row_names[i]},{self.col_names[j]}] = {d:.4g}")

            most_negative = min(delta.values()) if delta else 0

            if most_negative >= -EPS:
                self._log("\nAll Delta_ij >= 0  ->  CURRENT SOLUTION IS OPTIMAL.")
                self.allocation = allocation
                self.basic = basic
                return allocation

            # ---- Choose entering cell (most negative Delta) ----
            entering = min(delta, key=delta.get)
            self._log(f"Most negative Delta at ({self.row_names[entering[0]]},"
                      f"{self.col_names[entering[1]]}) = {delta[entering]:.4g} "
                      f"-> ENTERING cell.")

            # ---- Build closed loop ----
            loop = self._find_closed_loop(entering, basic)
            if loop is None:
                self._log("Error: could not find a closed loop. Stopping.")
                break

            # loop alternates +,-,+,-,... starting with '+' at entering cell
            plus_cells = loop[0::2]
            minus_cells = loop[1::2]

            self._log("Closed loop: " +
                      " -> ".join(f"({self.row_names[i]},{self.col_names[j]})"
                                  f"{'(+)' if k % 2 == 0 else '(-)'}"
                                  for k, (i, j) in enumerate(loop)))

            theta = min(allocation[i][j] for (i, j) in minus_cells)
            self._log(f"theta = min allocation among '-' cells = {theta:.4g}")

            for (i, j) in plus_cells:
                allocation[i][j] += theta
            for (i, j) in minus_cells:
                allocation[i][j] -= theta

            # Clean near-zero allocations
            for (i, j) in minus_cells:
                if abs(allocation[i][j]) < EPS:
                    allocation[i][j] = 0

            self._log("Updated allocation (new BFS):")
            self.print_table(allocation)
            self._log(f"New total cost = {self.total_cost(allocation):.4g}")

        self.allocation = allocation
        self.basic = {(i, j) for i in range(m) for j in range(n) if allocation[i][j] > EPS}
        return allocation

    # ------------------------------------------------------------------ #
    # Degeneracy fix: add epsilon allocation(s) so basic cells = m+n-1
    # and the basic-cell graph is connected (spanning tree over rows+cols)
    # ------------------------------------------------------------------ #
    def _fix_degeneracy(self, allocation, basic):
        m, n = self.m, self.n
        needed = m + n - 1

        uf = UnionFind(m + n)
        for (i, j) in basic:
            uf.union(i, m + j)

        basic = set(basic)

        while len(basic) < needed:
            # find a zero cell (i, j) that connects two different components,
            # preferring the lowest cost such cell
            candidate = None
            for i in range(m):
                for j in range(n):
                    if (i, j) in basic:
                        continue
                    if uf.find(i) != uf.find(m + j):
                        c = self.cost[i][j]
                        if candidate is None or c < candidate[0]:
                            candidate = (c, i, j)
            if candidate is None:
                break  # nothing left to connect (shouldn't happen)
            _, i, j = candidate
            allocation[i][j] = EPSILON_ALLOC
            basic.add((i, j))
            uf.union(i, m + j)
            self._log(f"Degenerate case: basic cells < m+n-1. Added epsilon "
                      f"allocation at ({self.row_names[i]},{self.col_names[j]}) "
                      f"to restore m+n-1 = {needed} basic cells.")

        return basic

    # ------------------------------------------------------------------ #
    # Find a closed loop (stepping-stone loop) starting/ending at `start`,
    # moving only through cells in `basic`, alternating row-move/col-move.
    # Returns list [start, c1, c2, ..., ] with alternating +/- signs
    # (start is '+', c1 is '-', c2 is '+', ...).
    # ------------------------------------------------------------------ #
    def _find_closed_loop(self, start, basic):
        cells = list(basic)

        def next_candidates(path):
            last = path[-1]
            if len(path) == 1:
                # first move: can go along the row or the column of start
                return [c for c in cells if (c[0] == last[0] or c[1] == last[1]) and c != last]
            prev = path[-2]
            same_row_as_last = (prev[0] == last[0])
            if same_row_as_last:
                # last move was horizontal -> next move must be vertical
                return [c for c in cells if c[1] == last[1] and c != last]
            else:
                # last move was vertical -> next move must be horizontal
                return [c for c in cells if c[0] == last[0] and c != last]

        def dfs(path):
            if len(path) >= 4:
                last = path[-1]
                # can we close back to start? (must be a valid alternating move)
                closes_row = (last[0] == start[0])
                closes_col = (last[1] == start[1])
                if closes_row or closes_col:
                    # also validate alternating pattern with previous cell
                    prev = path[-2]
                    prev_was_row_move = (prev[0] == last[0])
                    needed_col_move = prev_was_row_move  # next must be vertical
                    if (needed_col_move and closes_col) or ((not needed_col_move) and closes_row):
                        return path

            for nxt in next_candidates(path):
                if nxt in path:
                    continue
                result = dfs(path + [nxt])
                if result:
                    return result
            return None

        result = dfs([start])
        return result

    # ------------------------------------------------------------------ #
    # Utility: total cost, printing
    # ------------------------------------------------------------------ #
    def total_cost(self, allocation=None):
        allocation = allocation if allocation is not None else self.allocation
        return sum(self.cost[i][j] * allocation[i][j]
                   for i in range(self.m) for j in range(self.n))

    def print_table(self, allocation=None, clean_epsilon=False):
        allocation = allocation if allocation is not None else self.allocation
        header = "        " + "".join(f"{c:>10}" for c in self.col_names)
        self._log(header)
        for i in range(self.m):
            vals = []
            for j in range(self.n):
                v = allocation[i][j]
                if clean_epsilon and abs(v) < 1e-4:
                    v = 0
                vals.append(v)
            row_str = f"{self.row_names[i]:>8}" + "".join(f"{v:>10.4g}" for v in vals)
            self._log(row_str)

    def solve(self):
        self.vam()
        self.modi()
        self._log("=" * 70)
        self._log("FINAL OPTIMAL SOLUTION")
        self._log("=" * 70)
        self.print_table(clean_epsilon=True)
        self._log(f"\nMinimum Total Transportation Cost = {self.total_cost():.6g}")
        return self.allocation, self.total_cost()


# ----------------------------------------------------------------------------- #
#  Interactive input helper (for running the script directly on a terminal)
# ----------------------------------------------------------------------------- #
def read_problem_from_user():
    print("=" * 70)
    print("TRANSPORTATION PROBLEM - Data Entry")
    print("=" * 70)
    m = int(input("Enter number of ORIGINS (rows): "))
    n = int(input("Enter number of DESTINATIONS (columns): "))

    print(f"\nEnter the {m} x {n} cost matrix (row by row, space-separated):")
    cost = []
    for i in range(m):
        row = list(map(float, input(f"  Costs for O{i+1}: ").split()))
        assert len(row) == n, f"Expected {n} values, got {len(row)}"
        cost.append(row)

    supply = list(map(float, input(f"\nEnter {m} supply values (space-separated): ").split()))
    demand = list(map(float, input(f"Enter {n} demand values (space-separated): ").split()))
    assert len(supply) == m and len(demand) == n

    return cost, supply, demand


if __name__ == "__main__":
    print("Choose input mode:")
    print("  1. Enter data manually")
    print("  2. Run the sample problem from the assignment")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "2":
        cost = [
            [1, 2, -2, 3],
            [2, 4, 0, 1],
            [1, 2, -2, 5],
        ]
        supply = [70, 38, 32]
        demand = [40, 28, 30, 42]
    else:
        cost, supply, demand = read_problem_from_user()

    problem = TransportationProblem(cost, supply, demand)
    problem.solve()