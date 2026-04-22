import math

import numpy as np
import pandas as pd
from numpy.random import randn
from scipy import stats
from scipy.stats import chi2, ttest_1samp
from statsmodels.stats.weightstats import ztest

from lib.html import HtmlBuilder, PlotRenderer
from lib.mathshelper import FORMULA_REGISTRY
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running advance statistics operation report...")
    # ...


builder = HtmlBuilder()
plotRenderer = PlotRenderer()
content = []

hypothesis_testing_steps = f"""
1. Create a null hypothesis (H₀) and an alternative hypothesis (H₁).

2. Decide on the level of significance (α), commonly:
   - α = 0.05 (5%) or
   - α = 0.01 (1%).

3. Choose the appropriate statistical test based on the sample data:
   - Z‑test
   - T‑test
   - Chi‑square test.

4. Calculate the test statistic (Z‑score or T‑score) using the formula
   corresponding to the chosen test.

5. Obtain the critical value from the sampling distribution to construct
   the rejection region of size α using:
   - Z‑table,
   - T‑table, or
   - Chi‑square table.

6. Compare the calculated test statistic with the critical value to determine
   whether it falls in the rejection region or the non‑rejection region.

7. Decision Rule:
   a. If the test statistic falls in the rejection region, reject H₀.
      This indicates that the sample data provides sufficient evidence
      against the null hypothesis, and the difference between the
      hypothesized value and observed value is statistically significant.

   b. If the test statistic falls in the non‑rejection region, do not reject H₀.
      This indicates that the sample data does not provide sufficient evidence
      against the null hypothesis, and the observed difference is likely due
      to random sampling variation.
"""
cutlets_df, cutlets_report = dl.read_dataset("Hypothesis Testing\\cutlets.csv", optimize=True, handle_unnamed="drop", return_report=True)
ages_df, ages_report = dl.read_dataset("Hypothesis Testing\\Ages.csv", optimize=True, handle_unnamed="drop", return_report=True)
plant_df, plant_report = dl.read_dataset("Hypothesis Testing\\plant.csv", optimize=True, handle_unnamed="drop", return_report=True)
blood_pressure_df, blood_pressure_report = dl.read_dataset(
    "Hypothesis Testing\\blood_pressure.csv", optimize=True, handle_unnamed="drop", return_report=True)
chi_test_df, chi_test_report = dl.read_dataset("Hypothesis Testing\\chi_test.csv", optimize=True, handle_unnamed="drop", return_report=True)
crop_yield_df, crop_yield_report = dl.read_dataset("Hypothesis Testing\\crop_yield.csv", optimize=True, handle_unnamed="drop", return_report=True)
employee_satisfaction_df, employee_satisfaction_report = dl.read_dataset(
    "Hypothesis Testing\\employee_satisfaction.csv", optimize=True, handle_unnamed="drop", return_report=True)
customer_orderForm_df, customer_orderForm_report = dl.read_dataset(
    "Hypothesis Testing\\Customer_OrderForm.csv", optimize=True, handle_unnamed="drop", return_report=True)

# T- Test
# For a particular organization, the average age of the employees was claimed to be 30 years.
# The authorities collected a random sample of 10 employees' age data to check the claim made by the organization.
# Construct a hypothesis test to validate the hypothesis at a significance level of 0.05.

age_column = ages_df['ages']
mean_age = age_column.mean()

# Perform one-sample t-test
t_statistic, p_value = ttest_1samp(age_column, 30)

# Decision-making
alpha = 0.05
t_result = {}
t_result["null hypothesis"] = "Ho = mean = 30"
t_result["alternate hypothesis"] = "Ha = mean != 30"
t_result["mean_age"] = mean_age
t_result["t_statistics"] = t_statistic
t_result["p_value"] = np.round(p_value, decimals=4)
t_result["alpha"] = alpha
t_result["criteria"] = "if alpha < .05 then reject the Ho else Failed to reject the Ho"
if np.round(p_value, decimals=4) < alpha:
    t_result["hypothesis testing result"] = "Reject the null hypothesis."
else:
    t_result["hypothesis testing result"] = "Fail to reject the null hypothesis."

# Paired T-Test
# For a particular hospital, it is advertised that a particular chemotherapy session does not affect the patient's health
# based on blood pressure. It is to be checked if the blood pressure before the treatment is equivalent to the blood pressure after the treatment.
# Perform a statistical test at the aplha 0.05 level to help validate the claim.

ttest, pval = stats.ttest_rel(blood_pressure_df['bp_before'], blood_pressure_df['bp_after'])
t_p_result = {}
t_p_result["null hypothesis"] = "Ho = mean difference between two samples is 0"
t_p_result["alternate hypothesis"] = "Ha = mean difference between two samples is not 0"
t_p_result["t_statistics"] = ttest
t_p_result["p_value"] = np.round(pval, decimals=4)
t_p_result["alpha"] = alpha
t_p_result["criteria"] = "if alpha < .05 then reject the Ho else Failed to reject the Ho"
if np.round(pval, decimals=4) < alpha:
    t_p_result["hypothesis testing result"] = "Reject the null hypothesis."
else:
    t_p_result["hypothesis testing result"] = "Fail to reject the null hypothesis."

# Two Sample T-Test
# Employee satisfaction is a crucial factor that can influence the productivity and success of a company. The Human Resources department
# wants to assess whether the satisfaction levels are consistent across different departments. For this analysis, we will focus on two key
# departments: Sales and Marketing
# Separate the satisfaction scores for each department

sales_scores = employee_satisfaction_df[employee_satisfaction_df['Department'] == 'Sales']['Satisfaction_Score']
marketing_scores = employee_satisfaction_df[employee_satisfaction_df['Department'] == 'Marketing']['Satisfaction_Score']

# Perform the independent two-sample t-test
t_stat, p_val = stats.ttest_ind(sales_scores, marketing_scores, equal_var=False)

t_2s_result = {}
t_2s_result["null hypothesis"] = "Ho = Mean_Sales = Mean_Marketing"
t_2s_result["alternate hypothesis"] = "Ha = Mean_Sales ≠ Mean_Marketing"
t_2s_result["t_statistics"] = t_stat
t_2s_result["p_value"] = np.round(p_val, decimals=4)
t_2s_result["alpha"] = alpha
t_2s_result["criteria"] = "if alpha < .05 then reject the Ho else Failed to reject the Ho"
if np.round(p_val, decimals=4) < alpha:
    t_2s_result["hypothesis testing result"] = "Reject the null hypothesis."
else:
    t_2s_result["hypothesis testing result"] = "Fail to reject the null hypothesis."

# Z-Test
# A school principal claims that the students in their school are more intelligent than those of other schools.
# A random sample of 50 students' IQ scores has a mean score of 110. The mean population IQ is 100, with a standard deviation of 15.
# State whether the claim of the principal is right or not at a 5% significance level.

# Generate a random array of 50 numbers having mean 110 and standard deviation of 15
# similar to the IQ scores data
mean_iq = 110
sd_iq = 15 / math.sqrt(50)
alpha = 0.05
null_mean = 100
data = sd_iq * randn(50) + mean_iq

# Now you perform the test, and in this function, you passed data in the value parameter
# You passed mean value in the null hypothesis and will check if the mean is larger in the
# alternative hypothesis

ztest_Score, z_p_value = ztest(data, value=null_mean, alternative='larger')

z_result = {}
z_result["null hypothesis"] = "Ho = average population IQ score is 100"
z_result["alternate hypothesis"] = "Ha = average population IQ score is above 100"
z_result["mean_iq"] = mean_iq
z_result["sd_iq"] = sd_iq
z_result["null mean"] = null_mean
z_result["random data"] = data
z_result["mean of random data"] = np.round(np.mean(data), decimals=4)
z_result["standard deviation of random data"] = np.round(np.std(data), decimals=4)
z_result["z_statistics"] = ztest_Score
z_result["p_value"] = np.round(z_p_value, decimals=4)
z_result["alpha"] = alpha
z_result["criteria"] = "if alpha < .05 then reject the Ho else Failed to reject the Ho"
if np.round(z_p_value, decimals=4) < alpha:
    z_result["hypothesis testing result"] = "Reject the null hypothesis."
else:
    z_result["hypothesis testing result"] = "Fail to reject the null hypothesis."

# Chi-Test
# In a study about the election survey, voters might be classified by gender (male or female) and voting preference (democrat, republican, or independent).
# Using alpha = 0.05, perform a chi - square test for independence to determine whether gender is related to voting preference.

contingency_table = pd.crosstab(chi_test_df["Gender"], chi_test_df["Shopping"])
# Observed Values
Observed_Values = contingency_table.values
b = stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
no_of_rows = len(contingency_table.iloc[0:2, 0])
no_of_columns = len(contingency_table.iloc[0, 0:2])
ddof = (no_of_rows - 1) * (no_of_columns - 1)
alpha = 0.05
chi_square = sum([(o - e)**2. / e for o, e in zip(Observed_Values, Expected_Values)])
chi_square_statistic = chi_square[0] + chi_square[1]
critical_value = chi2.ppf(q=1 - alpha, df=ddof)
# P-value
chi_p_value = 1 - chi2.cdf(x=chi_square_statistic, df=ddof)

chi_result = {}
chi_result["null hypothesis"] = "Ho = There is a relationship between 2 categorical variables."
chi_result["alternate hypothesis"] = "Ha = There is no relationship between 2 categorical variables."
chi_result["contingency_table"] = contingency_table
chi_result["Observed_Values"] = Observed_Values
chi_result["Expected_Values"] = Expected_Values
chi_result["Degree of Freedom"] = ddof
chi_result["chi_square_statistic"] = chi_square_statistic
chi_result["critical_value"] = critical_value
chi_result["p_value"] = chi_p_value
chi_result["alpha"] = alpha
chi_result["criteria"] = "if alpha < .05 then reject the Ho else Failed to reject the Ho"
if chi_square_statistic >= critical_value:
    chi_result["hypothesis testing result based on critical value"] = "Reject Ho, There is a relationship between 2 categorical variables."
else:
    chi_result["hypothesis testing result based on critical value"] = "Retain Ho, There is no relationship between 2 categorical variables."

if chi_p_value <= alpha:
    chi_result["hypothesis testing result based on p value"] = "Reject Ho, There is a relationship between 2 categorical variables."
else:
    chi_result["hypothesis testing result based on p value"] = "Retain Ho, There is no relationship between 2 categorical variables"

# One Way Anova Test
plant_df = plant_df[['weight', 'group']]
groups = pd.unique(plant_df.group.values)
data = {grp: plant_df['weight'][plant_df.group == grp] for grp in groups}

F, p = stats.f_oneway(data['ctrl'], data['trt1'], data['trt2'])
one_way_anova_result = {}
one_way_anova_result["f_stats"] = F
one_way_anova_result["p_value"] = p
one_way_anova_result["alpha"] = alpha
one_way_anova_result["criteria"] = "if alpha < .05 then reject the Ho else Failed to reject the Ho"
if p < 0.05:
    one_way_anova_result["hypothesis testing result"] = "Reject Null Hypothesis."
else:
    one_way_anova_result["hypothesis testing result"] = "Accept Null Hypothesis."


content.append(
    builder.grid([
        formula.render(builder)
        for formula in FORMULA_REGISTRY.by_category("Hypothesis Testing")
    ]
    ))

content.append(
    builder.grid([
        builder.card("Cutlets Dataframe", builder.render_dataframe(cutlets_df)),
        builder.card("Plant Dataframe", builder.render_dataframe(plant_df)),
        builder.card("Ages Dataframe", builder.render_dataframe(ages_df)),
        builder.card("Crop Yield Dataframe", builder.render_dataframe(crop_yield_df)),
        builder.card("Chi-Test Dataframe", builder.render_dataframe(chi_test_df)),
        builder.card("Blood Pressure Dataframe", builder.render_dataframe(blood_pressure_df)),
        builder.card("Customer OrderForm Dataframe", builder.render_dataframe(customer_orderForm_df)),
        builder.card("Employee Satisfaction Dataframe", builder.render_dataframe(employee_satisfaction_df)),
        builder.card("Hypothesis Testing Steps", builder.render_pre(hypothesis_testing_steps)),
        builder.card("T-Test result for Age", builder.render_dict(t_result)),
        builder.card("Paired T-Test result for Blood Pressure", builder.render_dict(t_p_result)),
        builder.card("Two T-Test Sample result for Employee Satisfaction", builder.render_dict(t_2s_result)),
        builder.card("Z-Test result", builder.render_dict(z_result)),
        builder.card("Chi-Test result", builder.render_dict(chi_result)),
        builder.card("One Way Anova result", builder.render_dict(one_way_anova_result)),
    ])
)

html_doc = builder.build_page(
    "Advance Statistics Operation  Report",
    "\n".join(content)
)

# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "advance_statistics_operation_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")

if __name__ == "__main__":
    main()
    main()
    main()
    main()
