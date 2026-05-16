from lib.html import HtmlBuilder
from lib.mathshelper import FORMULA_REGISTRY
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running Linear Regression Mathshelper report...")
    # ...


builder = HtmlBuilder()
content = []

content.append(
    builder.grid([
        formula.render(builder)
        for formula in FORMULA_REGISTRY.by_category(["Linear Regression", "Regularization", "Derivation"])
    ]
    ))


html_doc = builder.build_page(
    "Linear Regression Mathshelper  Report",
    "\n".join(content)
)

# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "ml_linear_regression_maths_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")

if __name__ == "__main__":
    main()
