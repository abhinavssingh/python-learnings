import numpy as np

from lib.html import HtmlBuilder
from lib.utility.matrixhelper import MatrixHelper as mh
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running basic matrix operation report...")
    # ...


matrix_a = np.array([
    [1, 2, 3],
    [0, 1, 4],
    [5, 6, 0]]
)
matrix_b = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]]
)

u = np.array([1, 4, 5])
v = np.array([2, -3, 6])

helper = mh()
builder = HtmlBuilder()
content = []

dotproduct_result = helper.dot_product(matrix_a, matrix_b)
dotproduct_steps = "".join(
    builder.renderers.render_latex_block(step, display=True)
    for step in dotproduct_result["steps_latex"]
)

crossproduct_result = helper.cross_product(u, v)
crossproduct_steps = "".join(
    builder.renderers.render_latex_block(step, display=True)
    for step in crossproduct_result["steps_latex"]
)

a_determinent = helper.determinant(matrix_a)
determinent_steps = "".join(
    builder.renderers.render_latex_block(step, display=True)
    for step in a_determinent["steps_latex"]
)

a_adj = helper.adjoint(matrix_a)
adj_steps = "".join(
    builder.renderers.render_latex_block(step, display=True)
    for step in a_adj["steps_latex"]
)

a_inverse = helper.inverse(matrix_a)
inverse_steps = "".join(
    builder.renderers.render_latex_block(step, display=True)
    for step in a_inverse["steps_latex"]
)

a_row_ech = helper.rref(matrix_a)
row_ech_steps = "".join(
    builder.renderers.render_latex_block(step, display=True)
    for step in a_row_ech["steps_latex"]
)

a_rank = helper.rank(matrix_a)
rank_steps = "".join(
    builder.renderers.render_latex_block(step, display=True)
    for step in a_rank["steps_latex"]
)

a_eigen = helper.eigen(matrix_a)
eigen_steps = "".join(
    builder.renderers.render_latex_block(step, display=True)
    for step in a_eigen["steps_latex"]
)

content.append(builder.grid([
    builder.card(" First matrix A for dotproduct is:", builder.render_array(matrix_a, display=False)),
    builder.card(" Second matrix B for dotproduct is:", builder.render_array(matrix_b, display=False)),
    builder.card(" Result of dotproduct A.B is:", builder.render_array(np.array(dotproduct_result["result"]), display=False)),
    builder.card(" First matrix U for cross product is:", builder.render_array(u, display=False)),
    builder.card(" Second matrix V for cross product is:", builder.render_array(v, display=False)),
    builder.card(" Result of cross product U x V is:", builder.render_array(np.array(crossproduct_result["result"]), display=False)),
    builder.card(" Inverse of the first matrix A is:", builder.render_array(np.array(a_inverse["result"]), display=False)),
    builder.card(" Rank of the first matrix A is:", builder.render_array(np.array(a_rank["result"]), display=False)),
    builder.card(" Determinent first matrix A is:", builder.render_array(np.array(a_determinent["result"]), display=False)),
    builder.math_card(" Eigen of the first matrix A is:", builder.render_eigen_results(a_eigen["result"])),
    builder.math_card("Dot Product steps:", dotproduct_steps),
    builder.math_card("Cross Product Steps", crossproduct_steps),
    builder.math_card("Determinent Steps", determinent_steps),
    builder.math_card("Adjoint steps:", adj_steps),
    builder.math_card("Inverse steps:", inverse_steps),
    builder.math_card("Row Echelon steps:", row_ech_steps),
    builder.math_card("Rank steps;", rank_steps),
    builder.math_card("Eigen steps", eigen_steps),
]))

html_doc = builder.build_page(
    "Basic Matrix Operation  Report",
    "\n".join(content)
)

# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "basic_matrix_operation_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")

if __name__ == "__main__":
    main()
    main()
