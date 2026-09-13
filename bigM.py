"""
================================================================================
 GENERALIZED BIG-M SIMPLEX SOLVER
================================================================================
Solves a Linear Program of the form given only:
    - the ORIGINAL objective function coefficients (max or min), and
    - the ORIGINAL constraint coefficients (<=, =, >=)

The program automatically:
    1. Converts a MINIMIZE problem into an equivalent MAXIMIZE problem.
    2. Adds slack variables (<=), surplus variables (>=) and artificial
       variables (= , >=) to build the augmented/standard form.
    3. Assigns a cost of -M to every artificial variable (M -> +infinity)
       so that they are the entering objective's "Big-M" penalty.
    4. Runs the simplex method symbolically -- every number that can
       contain the symbol M is stored as a pair (real_part, M_part) and
       compared as   real_part + M_part * M   with M -> +infinity.
       This is EXACTLY how the coursework slides compute the "Dev. Row"
       (e.g. "3 - 6M", "M - 1", "3M - 1" ...): no floating point M is
       ever substituted, so there is no numerical error.
    5. Prints every simplex tableau (Basis, Cj row, body, Dev./(Cj-Zj)
       row and RHS) exactly like the course slides.
    6. Classifies the final tableau:
           - UNIQUE optimal solution   -> every non-basic var has (Cj-Zj) < 0
           - MULTIPLE optimal solutions -> some non-basic var has (Cj-Zj) == 0
           - UNBOUNDED                 -> an entering column has no positive
                                           entry to run the ratio test on
           - INFEASIBLE                -> optimal reached but an artificial
                                           variable is still in the basis at
                                           a strictly positive value

--------------------------------------------------------------------------------
INPUT FORMAT (matches the shorthand described in the prompt)
--------------------------------------------------------------------------------
Objective:
    A list of tuples.  Every tuple except the last is (var_index, coeff).
    The LAST tuple is a 1-tuple holding the objective type code:
        11  -> maximize
        10  -> minimize

    Example (Minimize z = -3x1 + x2 + x3):
        [(1, -3), (2, 1), (3, 1), (10,)]

Constraints:
    A list of constraints. Each constraint is itself a list of tuples.
    Every tuple except the last TWO is (var_index, coeff).
    The SECOND-LAST tuple is a 1-tuple holding the RHS constant.
    The LAST tuple is a 1-tuple holding the relation type code:
         1  ->  <=
         0  ->  =
        -1  ->  >=

    Example  (x1 - 2x2 + x3 <= 11):
        [(1, 1), (2, -2), (3, 1), (11,), (1,)]

    Example  (-4x1 + x2 + 2x3 >= 3):
        [(1, -4), (2, 1), (3, 2), (3,), (-1,)]

    Example  (2x1 - x3 = -1):
        [(1, 2), (3, -1), (-1,), (0,)]
================================================================================
"""

from fractions import Fraction


# ==============================================================================
# 1. Symbolic  (real  +  M * m)   number, used for every COST-derived quantity
# ==============================================================================
class MVal:
    """Represents  r + m*M  where M -> +infinity symbolically (no float M)."""

    __slots__ = ("r", "m")

    def __init__(self, r=0, m=0):
        self.r = Fraction(r)
        self.m = Fraction(m)

    # ---- arithmetic -----------------------------------------------------
    def __add__(self, other):
        return MVal(self.r + other.r, self.m + other.m)

    def __sub__(self, other):
        return MVal(self.r - other.r, self.m - other.m)

    def __neg__(self):
        return MVal(-self.r, -self.m)

    def __mul__(self, k):
        k = Fraction(k)
        return MVal(self.r * k, self.m * k)

    __rmul__ = __mul__

    # ---- comparisons (M dominates, since M -> +infinity) -----------------
    def __eq__(self, other):
        return self.m == other.m and self.r == other.r

    def __lt__(self, other):
        if self.m != other.m:
            return self.m < other.m
        return self.r < other.r

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        if self.m != other.m:
            return self.m > other.m
        return self.r > other.r

    def __ge__(self, other):
        return self > other or self == other

    def is_zero(self):
        return self.r == 0 and self.m == 0

    # ---- pretty printing --------------------------------------------------
    @staticmethod
    def _fmt(frac):
        if frac.denominator == 1:
            return str(frac.numerator)
        return f"{frac.numerator}/{frac.denominator}"

    def __repr__(self):
        if self.m == 0:
            return self._fmt(self.r)
        m_part = ("M" if self.m == 1 else "-M" if self.m == -1
                  else f"{self._fmt(self.m)}M")
        if self.r == 0:
            return m_part
        sign = "+" if self.r > 0 else "-"
        return f"{m_part}{sign}{self._fmt(abs(self.r))}"


ZERO = MVal(0, 0)


# ==============================================================================
# 2. Parsing the shorthand tuple notation into an internal LP description
# ==============================================================================
class LP:
    def __init__(self, n_vars, obj_coeffs, sense, constraints):
        self.n_vars = n_vars          # number of ORIGINAL decision variables
        self.obj_coeffs = obj_coeffs  # dict {var_index: Fraction}
        self.sense = sense            # 'max' or 'min'
        self.constraints = constraints  # list of (coeffs_dict, rhs, op)


def parse_objective(tuples):
    *coeff_tuples, type_tuple = tuples
    type_code = type_tuple[0]
    sense = {11: "max", 10: "min"}[type_code]
    coeffs = {idx: Fraction(c) for idx, c in coeff_tuples}
    return coeffs, sense


def parse_constraint(tuples):
    *coeff_tuples, rhs_tuple, type_tuple = tuples
    rhs = Fraction(rhs_tuple[0])
    op = {1: "<=", 0: "=", -1: ">="}[type_tuple[0]]
    coeffs = {idx: Fraction(c) for idx, c in coeff_tuples}
    return coeffs, rhs, op


def build_lp(objective_tuples, constraints_tuples):
    obj_coeffs, sense = parse_objective(objective_tuples)
    constraints = [parse_constraint(c) for c in constraints_tuples]
    n_vars = max(
        [i for i in obj_coeffs] +
        [i for c, _, _ in constraints for i in c]
    )
    return LP(n_vars, obj_coeffs, sense, constraints)


# ==============================================================================
# 3. Big-M Simplex engine
# ==============================================================================
class BigMSimplex:
    def __init__(self, lp: LP, verbose=True):
        self.lp = lp
        self.verbose = verbose
        self.original_sense = lp.sense
        self._build_standard_form()

    # ------------------------------------------------------------------
    def _build_standard_form(self):
        lp = self.lp
        n = lp.n_vars

        # Step 1: force MAXIMIZE. If original is 'min', negate objective;
        # remember this so we can flip the reported optimum back at the end.
        sign = 1 if lp.sense == "max" else -1
        self.min_to_max = (lp.sense == "min")
        obj = {i: sign * lp.obj_coeffs.get(i, Fraction(0)) for i in range(1, n + 1)}

        rows = []          # list of dict {var_index: Fraction}
        rhs = []
        extra_names = []   # names of extra vars added, in creation order
        extra_cost = {}    # var_name -> MVal cost
        artificial_names = []

        col_order = [f"x{i}" for i in range(1, n + 1)]

        for k, (coeffs, r, op) in enumerate(lp.constraints, start=1):
            row = {f"x{i}": coeffs.get(i, Fraction(0)) for i in range(1, n + 1)}

            # Ensure RHS >= 0 (multiply whole row by -1 and flip relation if needed)
            if r < 0:
                row = {v: -c for v, c in row.items()}
                r = -r
                op = {"<=": ">=", ">=": "<=", "=": "="}[op]

            if op == "<=":
                s_name = f"s{k}"
                row[s_name] = Fraction(1)
                extra_cost[s_name] = MVal(0, 0)
                col_order.append(s_name)
                extra_names.append(s_name)

            elif op == ">=":
                s_name = f"s{k}"
                a_name = f"a{k}"
                row[s_name] = Fraction(-1)
                row[a_name] = Fraction(1)
                extra_cost[s_name] = MVal(0, 0)
                extra_cost[a_name] = MVal(0, -1)     # -M
                col_order.append(s_name)
                col_order.append(a_name)
                extra_names.append(s_name)
                extra_names.append(a_name)
                artificial_names.append(a_name)

            else:  # "="
                a_name = f"a{k}"
                row[a_name] = Fraction(1)
                extra_cost[a_name] = MVal(0, -1)     # -M
                col_order.append(a_name)
                extra_names.append(a_name)
                artificial_names.append(a_name)

            rows.append(row)
            rhs.append(r)

        # de-duplicate col_order while keeping first-seen order
        seen = set()
        ordered_cols = []
        for c in col_order:
            if c not in seen:
                ordered_cols.append(c)
                seen.add(c)

        self.columns = ordered_cols
        self.cost = {f"x{i}": MVal(obj[i], 0) for i in range(1, n + 1)}
        self.cost.update(extra_cost)
        self.artificial_names = artificial_names

        # Build the tableau body as a plain matrix aligned with self.columns
        m = len(rows)
        self.tableau = [[rows[i].get(col, Fraction(0)) for col in self.columns]
                         for i in range(m)]
        self.rhs = rhs[:]

        # Initial basis: the slack (<=) or artificial (=, >=) var of each row
        self.basis = []
        for k, (coeffs, r, op) in enumerate(lp.constraints, start=1):
            if op == "<=":
                self.basis.append(f"s{k}")
            else:
                self.basis.append(f"a{k}")

    # ------------------------------------------------------------------
    def _cb(self):
        return [self.cost[b] for b in self.basis]

    def _zj(self, col_idx):
        cb = self._cb()
        z = ZERO
        for i, cval in enumerate(cb):
            a = self.tableau[i][col_idx]
            if a != 0:
                z = z + cval * a
        return z

    def _reduced_costs(self):
        rc = []
        for j, col in enumerate(self.columns):
            rc.append(self.cost[col] - self._zj(j))
        return rc

    def _z_value(self):
        cb = self._cb()
        z = ZERO
        for i, cval in enumerate(cb):
            z = z + cval * self.rhs[i]
        return z

    # ------------------------------------------------------------------
    def _print_table(self, table_no, rc):
        cols = self.columns
        print(f"\n--- TABLE {table_no} ---")

        # column width = widest entry in that column (name, cost, all rows, dev)
        widths = []
        for j, c in enumerate(cols):
            entries = [c, str(self.cost[c])] + [str(self.tableau[i][j]) for i in range(len(self.basis))] + [str(rc[j])]
            widths.append(max(len(e) for e in entries) + 2)
        basisw = max(6, max(len(b) for b in self.basis) + 2)
        rhsw = max(6, max(len(str(r)) for r in self.rhs) + 2)

        print(" " * basisw + "".join(str(self.cost[c]).rjust(w) for w, c in zip(widths, cols)))
        print("Basis".ljust(basisw) + "".join(c.rjust(w) for w, c in zip(widths, cols)) + "RHS".rjust(rhsw))
        for i, b in enumerate(self.basis):
            row = "".join(str(self.tableau[i][j]).rjust(w) for j, w in enumerate(widths))
            print(b.ljust(basisw) + row + str(self.rhs[i]).rjust(rhsw))
        dev = "".join(str(rc[j]).rjust(w) for j, w in enumerate(widths))
        print("Dev.".ljust(basisw) + dev)
        print(f"z = {self._z_value()}")

    # ------------------------------------------------------------------
    def solve(self, max_iter=200):
        table_no = 1
        status = None
        entering_when_unbounded = None

        for _ in range(max_iter):
            rc = self._reduced_costs()
            if self.verbose:
                self._print_table(table_no, rc)

            # ---- optimality check (maximize): all reduced costs <= 0 ----
            candidates = [j for j, v in enumerate(rc) if v > ZERO]
            if not candidates:
                status = "OPTIMAL_REACHED"
                break

            # ---- entering variable: most positive reduced cost ----------
            enter_j = max(candidates, key=lambda j: rc[j])
            enter_name = self.columns[enter_j]

            # ---- ratio test ----------------------------------------------
            ratios = []
            for i in range(len(self.basis)):
                a = self.tableau[i][enter_j]
                if a > 0:
                    ratios.append((self.rhs[i] / a, i))

            if not ratios:
                status = "UNBOUNDED"
                entering_when_unbounded = enter_name
                break

            min_ratio = min(r for r, i in ratios)
            # tie-break: prefer to kick out an artificial variable, else
            # the row with the smallest basic-variable name (Bland's rule)
            tied = [i for r, i in ratios if r == min_ratio]
            leave_i = min(
                tied,
                key=lambda i: (0 if self.basis[i] in self.artificial_names else 1,
                               self.basis[i])
            )
            leave_name = self.basis[leave_i]

            if self.verbose:
                print(f"Entering variable: {enter_name}   "
                      f"(most positive Dev.Row value = {rc[enter_j]})")
                print(f"Leaving variable : {leave_name}   "
                      f"(minimum ratio = {min_ratio})")

            # ---- pivot -----------------------------------------------------
            pivot = self.tableau[leave_i][enter_j]
            self.tableau[leave_i] = [v / pivot for v in self.tableau[leave_i]]
            self.rhs[leave_i] = self.rhs[leave_i] / pivot

            for i in range(len(self.basis)):
                if i == leave_i:
                    continue
                factor = self.tableau[i][enter_j]
                if factor != 0:
                    self.tableau[i] = [
                        self.tableau[i][j] - factor * self.tableau[leave_i][j]
                        for j in range(len(self.columns))
                    ]
                    self.rhs[i] = self.rhs[i] - factor * self.rhs[leave_i]

            self.basis[leave_i] = enter_name
            table_no += 1

        else:
            status = "MAX_ITER_REACHED"

        self.status = status
        self.entering_when_unbounded = entering_when_unbounded
        self.last_rc = self._reduced_costs()
        return self._report()

    # ------------------------------------------------------------------
    def _solution_dict(self):
        sol = {c: Fraction(0) for c in self.columns}
        for i, b in enumerate(self.basis):
            sol[b] = self.rhs[i]
        return sol

    def _report(self):
        n = self.lp.n_vars
        result = {"status": self.status}

        if self.status == "UNBOUNDED":
            result["message"] = (
                f"UNBOUNDED SOLUTION.\n"
                f"The entering variable '{self.entering_when_unbounded}' has a "
                f"positive Dev.Row (Cj-Zj) value, but every entry in its "
                f"column is <= 0, so no ratio test / leaving variable exists. "
                f"The objective function can be increased without bound."
            )
            print("\n" + result["message"])
            return result

        if self.status == "MAX_ITER_REACHED":
            result["message"] = "Iteration limit reached without termination."
            print("\n" + result["message"])
            return result

        # ---- optimal tableau reached: check feasibility -------------------
        sol = self._solution_dict()
        infeasible = any(
            a in self.basis and sol[a] != 0 for a in self.artificial_names
        )
        if infeasible:
            result["status"] = "INFEASIBLE"
            result["message"] = (
                "INFEASIBLE PROBLEM.\n"
                "The optimal Big-M tableau still has an artificial variable "
                "in the basis at a value > 0. Since M -> infinity, this can "
                "only happen if the original constraints have no feasible "
                "region — the problem is INFEASIBLE."
            )
            print("\n" + result["message"])
            return result

        # ---- classify unique vs multiple optimal solutions -----------------
        non_basic = [j for j, c in enumerate(self.columns) if c not in self.basis]
        zero_rc_non_artificial = [
            self.columns[j] for j in non_basic
            if self.last_rc[j].is_zero() and self.columns[j] not in self.artificial_names
        ]

        if zero_rc_non_artificial:
            result["status"] = "MULTIPLE_OPTIMA"
            result["message"] = (
                "MULTIPLE (ALTERNATE) OPTIMAL SOLUTIONS.\n"
                f"Non-basic variable(s) {zero_rc_non_artificial} have a "
                "Dev.Row (Cj-Zj) value of exactly 0 in the final optimal "
                "tableau. Bringing any of them into the basis (at ratio-test "
                "determined level) gives another optimal solution with the "
                "SAME objective value."
            )
        else:
            result["status"] = "UNIQUE_OPTIMAL"
            result["message"] = (
                "UNIQUE OPTIMAL SOLUTION.\n"
                "Every non-basic variable has a strictly negative Dev.Row "
                "(Cj-Zj) value in the final optimal tableau."
            )

        z_max = self._z_value()          # value of the MAXIMIZED objective
        z_report = -z_max.r if self.min_to_max else z_max.r
        result["z_max_form"] = z_max.r
        result["z_original"] = z_report
        result["x"] = {f"x{i}": sol.get(f"x{i}", Fraction(0)) for i in range(1, n + 1)}

        print("\n" + result["message"])
        print(f"\nOptimal solution: " +
              ", ".join(f"{k} = {v}" for k, v in result["x"].items()))
        sense_word = "Minimum" if self.min_to_max else "Maximum"
        print(f"{sense_word} value of original objective z = {z_report}")

        return result


# ==============================================================================
# 4. Public convenience function
# ==============================================================================
def solve_bigm(objective_tuples, constraints_tuples, verbose=True):
    lp = build_lp(objective_tuples, constraints_tuples)
    solver = BigMSimplex(lp, verbose=verbose)
    return solver.solve()


# ==============================================================================
# 5. GIve Input of Objective Function and Constraints 
# ==============================================================================
if __name__ == "__main__":
    # Minimize z = -3x1 + x2 + x3
    # s.t.  x1 - 2x2 + x3 <= 11
    #      -4x1 + x2 + 2x3 >= 3
    #       2x1 - x3 = -1
    #       x1, x2, x3 >= 0
    objective = [(1, 5), (2, 2), (3, 3),(4,-1),(5,1), (11,)]
    constraints = [
        [(1, 1), (2, 2), (3, 2), (4, 1), (8,), (0,)],
        [(1, 3), (2, 4), (3, 1), (5, 1), (7,), (0,)],
    ]

    solve_bigm(objective, constraints)