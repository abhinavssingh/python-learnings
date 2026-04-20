from __future__ import annotations


class MatrixHelper:
    import sympy as sp
    """
    Stateless learner-friendly Matrix Helper.

    All matrices/vectors are passed explicitly to each method.
    No ambiguity, safe for teaching and production.
    """

    # =============================
    # Binary Operations
    # =============================
    def add(self, A, B):
        A, B = self.sp.Matrix(A), self.sp.Matrix(B)
        C = A + B
        latex = f"{self.sp.latex(A)} + {self.sp.latex(B)} = {self.sp.latex(C)}"
        return {"result": C, "latex": latex}

    def subtract(self, A, B):
        A, B = self.sp.Matrix(A), self.sp.Matrix(B)
        C = A - B
        latex = f"{self.sp.latex(A)} - {self.sp.latex(B)} = {self.sp.latex(C)}"
        return {"result": C, "latex": latex}

    # =============================
    # Matrix Multiplication (A × B)
    # =============================
    def dot_product(self, A, B):
        A, B = self.sp.Matrix(A), self.sp.Matrix(B)
        C = A * B

        steps_latex = []

        for i in range(A.rows):
            for j in range(B.cols):
                terms = []
                values = []

                for k in range(A.cols):
                    terms.append(
                        f"({self.sp.latex(A[i, k])} \\times {self.sp.latex(B[k, j])})"
                    )
                    values.append(A[i, k] * B[k, j])

                step = (
                    "\\begin{aligned}"
                    f"C_{{{i + 1}{j + 1}}} &= " + " + ".join(terms) + " \\\\ "
                    f"&= " + " + ".join(self.sp.latex(v) for v in values) + " \\\\ "
                    f"&= {self.sp.latex(sum(values))}"
                    "\\end{aligned}"
                )

                steps_latex.append(step)

        latex = f"{self.sp.latex(A)} \\times {self.sp.latex(B)} = {self.sp.latex(C)}"

        return {
            "result": C,
            "steps_latex": steps_latex,
            "latex": latex
        }

    def _latex_matrix(self, M):
        return self.sp.latex(M, mat_str="pmatrix", mat_delim=None)

    # =============================
    # Cross Product (3D vectors ONLY)
    # =============================

    def cross_product(self, u, v):
        u, v = self.sp.Matrix(u), self.sp.Matrix(v)

        if u.shape not in [(3, 1), (1, 3)] or v.shape not in [(3, 1), (1, 3)]:
            raise ValueError("Cross product requires 3D vectors")

        u = u.reshape(3, 1)
        v = v.reshape(3, 1)

        C = u.cross(v)

        u_l = self._latex_matrix(u)
        v_l = self._latex_matrix(v)
        c_l = self._latex_matrix(C)

        steps = [
            r"\begin{aligned}"
            r"\vec{u} \times \vec{v}"
            r"&= \begin{vmatrix}"
            r"\mathbf{i} & \mathbf{j} & \mathbf{k} \\"
            f"{u[0]} & {u[1]} & {u[2]} \\"
            f"{v[0]} & {v[1]} & {v[2]}"
            r"\end{vmatrix} \\[6pt]"
            rf"&= \mathbf{{i}}({u[1]}\times{v[2]} - {u[2]}\times{v[1]})"
            rf" - \mathbf{{j}}({u[0]}\times{v[2]} - {u[2]}\times{v[0]})"
            rf" + \mathbf{{k}}({u[0]}\times{v[1]} - {u[1]}\times{v[0]}) \\[6pt]"
            rf"&= {c_l}"
            r"\end{aligned}"
        ]

        latex = f"{u_l} \\times {v_l} = {c_l}"

        return {
            "result": C,
            "steps_latex": steps,
            "latex": latex
        }

    # =============================
    # Single-Matrix Operations
    # =============================

    def determinant(self, A):
        A = self.sp.Matrix(A)
        detA = A.det()

        steps = []

        if A.shape == (3, 3):
            a, b, c = A[0, 0], A[0, 1], A[0, 2]
            d, e, f = A[1, 0], A[1, 1], A[1, 2]
            g, h, i = A[2, 0], A[2, 1], A[2, 2]

            steps = [
                r"\begin{aligned}"
                r"\text{Using cofactor expansion along the first row} \\[6pt]"
                rf"\det(A) &="
                rf" {a}({e}\times{i} - {f}\times{h})"
                rf" - {b}({d}\times{i} - {f}\times{g})"
                rf" + {c}({d}\times{h} - {e}\times{g}) \\[6pt]"
                rf"&= {self.sp.latex(detA)}"
                r"\end{aligned}"
            ]

        latex = rf"\det(A) = {self.sp.latex(detA)}"

        return {
            "result": detA,
            "steps_latex": steps,
            "latex": latex
        }

    def adjoint(self, A):
        A = self.sp.Matrix(A)
        adjA = A.adjugate()

        # Force MathJax-safe matrix rendering
        A_ltx = self._latex_matrix(A)
        adj_ltx = self._latex_matrix(adjA)

        steps = [
            r"\begin{aligned}"
            r"\text{The adjoint of a matrix is the tranself.spose of its cofactor matrix} \\[6pt]"
            rf"\operatorname{{adj}}(A) & = (\text{{Cofactor}}(A))^T \\[6pt]"
            rf"A & = {A_ltx} \\[6pt]"
            rf"\operatorname{{adj}}(A) & = {adj_ltx}"
            r"\end{aligned}"
        ]

        latex = rf"\operatorname{{adj}}(A) = {adj_ltx}"

        return {
            "result": adjA,
            "steps_latex": steps,
            "latex": latex,
        }

    def inverse(self, A):
        A = self.sp.Matrix(A)
        detA = A.det()
        adjA = A.adjugate()
        invA = A.inv()

        steps = [
            r"\begin{aligned}"
            r"A^{-1} & = \frac{1}{\det(A)}\,\operatorname{adj}(A) \\[6pt]"
            rf"\det(A) & = {self.sp.latex(detA)} \\[6pt]"
            rf"\operatorname{{adj}}(A) & = {self.sp.latex(adjA)} \\[6pt]"
            rf"A^{-1} & = {self.sp.latex(invA)}"
            r"\end{aligned}"
        ]

        latex = rf"{self.sp.latex(A)}^{{-1}} = {self.sp.latex(invA)}"

        return {
            "result": invA,
            "steps_latex": steps,
            "latex": latex
        }

    def rref(self, A):
        A = self.sp.Matrix(A)
        R, pivots = A.rref()

        # Force MathJax-safe matrix rendering
        R_ltx = self.sp.latex(R, mat_str="pmatrix", mat_delim=None)

        steps = [
            r"\begin{aligned}"
            r"\text{Convert matrix to Reduced Row Echelon Form (RREF)} \\[6pt]"
            rf"\text{{RREF}}(A) & = {R_ltx} \\[6pt]"
            rf"\text{{Pivot columns}} & = {list(pivots)}"
            r"\end{aligned}"
        ]

        return {
            "result": R,
            "pivots": pivots,
            "steps_latex": steps,
        }

    def rank(self, A):
        A = self.sp.Matrix(A)
        R, pivots = A.rref()

        steps = [
            r"\begin{aligned}"
            r"\text{Convert matrix to Reduced Row Echelon Form (RREF)} \\[6pt]"
            rf"\text{{RREF}}(A) & = {self._latex_matrix(R)} \\[6pt]"
            rf"\text{{Number of pivot columns}} & = {len(pivots)}"
            r"\end{aligned}"
        ]

        latex = rf"\text{{rank}}({self.sp.latex(A)}) = {len(pivots)}"

        return {
            "result": len(pivots),
            "steps_latex": steps,
            "latex": latex
        }

    def eigen(self, A):
        A = self.sp.Matrix(A)
        λ = self.sp.symbols("λ")
        char_poly = (A - λ * self.sp.eye(A.rows)).det()

        results = []
        for val, mult, vecs in A.eigenvects():
            results.append({
                "eigenvalue": val,
                "multiplicity": mult,
                "eigenvectors": vecs,
            })

        steps = [
            r"\begin{aligned}"
            r"\text{Solve the characteristic equation} \\[6pt]"
            rf"\det(A - \lambda I) &= {self.sp.latex(char_poly)} = 0"
            r"\end{aligned}"
        ]

        return {
            "result": results,
            "steps_latex": steps,
            "latex": r"\det(A - \lambda I) = 0"
        }
