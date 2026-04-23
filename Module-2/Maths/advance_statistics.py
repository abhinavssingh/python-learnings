import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.random import randn
from scipy import stats
from scipy.stats import chi2, f_oneway, ttest_1samp
from statsmodels.formula.api import ols
from statsmodels.stats.weightstats import ztest

from lib.html import HtmlBuilder, PlotRenderer
from lib.mathshelper import FORMULA_REGISTRY
from lib.plothelper.PlotHelper import DistributionPlotHelper as dph
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running advance statistics operation report...")
    # ...


builder = HtmlBuilder()
plotRenderer = PlotRenderer()
content = []

df_pairs = [
    (1, 12),     # Cutlets
    (2, 27),     # Plant Weights
    (3, 16),     # Crop Yield
]
dof_list = [1, 2, 5, 10]

hypothesis_testing_steps = """
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


chi_test_fig = dph.plot_chi_square_distribution(
    observed_chi2=chi_square_statistic,
    dof=ddof,
    title="Chi‑Square Distribution: Chi‑Test Dataset"
)

multi_chi_fig = dph.plot_multiple_chi_square_distributions(
    dof_list=dof_list,
    observed_chi2=chi_square_statistic,
    title="Chi‑Square Distribution: Chi‑Test Dataset"
)


anova_summary = """
1. Statistical tests such as the T-Test and Z-Test are suitable
 for univariate analysis, whereas ANOVA is used to compare the means
 of multiple groups in multivariate scenarios.

2. ANOVA is widely used in bio-chemical, pharmaceutical,
and scientific research to study the effect of multiple independent
variables on a dependent variable.

3. In a medical experiment, such as testing different hypertension
medicines, the sample population is divided into groups, each receiving
a specific treatment. After measuring the outcome (e.g., blood pressure),
ANOVA compares group means to determine whether they are statistically different.

4. ANOVA evaluates sample means and the grand mean, examines dependent
and independent variables (factors and their levels), and produces an F-Statistic.
This ratio compares intergroup and intragroup variation to decide whether to accept
or reject the null hypothesis.
"""

# One Way Anova Test
# Three different categories of plants can be differentiated on the basis of their weights. A dataset with various plants and
# their weights is given. Construct a hypothetical test to determine the category of a plant at a significance level of 0.05.
plant_df = plant_df[['weight', 'group']]
groups = pd.unique(plant_df.group.values)
data = {grp: plant_df['weight'][plant_df.group == grp] for grp in groups}

F, p = stats.f_oneway(data['ctrl'], data['trt1'], data['trt2'])
one_way_anova_result = {}
one_way_anova_result["f_stats"] = F
one_way_anova_result["p_value"] = np.round(p, decimals=4)
one_way_anova_result["alpha"] = alpha
one_way_anova_result["criteria"] = "if alpha < .05 then reject the Ho else Failed to reject the Ho"
if np.round(p, decimals=4) < 0.05:
    one_way_anova_result["hypothesis testing result"] = "Reject Null Hypothesis."
else:
    one_way_anova_result["hypothesis testing result"] = "Accept Null Hypothesis."

# F-Distribution Plot for One-Way ANOVA
groups = plant_df["group"].nunique()
total_samples = len(plant_df)

dfn = groups - 1
dfd = total_samples - groups

plant_f_fig = dph.plot_f_distribution(
    observed_f=F,
    dfn=dfn,
    dfd=dfd,
    title="F‑Distribution: Plant Weights ANOVA"
)

multi_plant_f_fig = dph.plot_multiple_f_distributions(
    df_pairs=df_pairs,
    observed_f=F,
    title="F‑Distribution Comparison Across ANOVA Tests for Plant Weights"
)

# Two Way Anova Test
# The crop yield of a particular region is affected by the amount of fertilizer used and the type of crop.
# A dataset with various crops, the amount of fertilizer used, and their corresponding yields is given.

# Fit the model
model = ols('Yield ~ C(Fert)*C(Water)', crop_yield_df).fit()

# Perform ANOVA and print the table
res = sm.stats.anova_lm(model, typ=2)

model_summary = """
The overall model result, F(3, 16) = 4.112, p = 0.0243, tells you about the fit of the entire model,
including all the predictors (Fert, Water, and the interaction term). An F-statistic of 4.112 and p-value of 0.0243
(which is less than the usual significance level of 0.05) indicates that the model as a whole is statistically significant,
 meaning there is evidence that at least one of the predictors has a non-zero effect.
"""
anova_table = {}
anova_table["Fertilizer"] = """
The p-value for Fert is 0.028847, which is less than 0.05. This indicate that there is a statistically
significant difference in crop yield between the different types of fertilizer.
"""
anova_table["Water"] = """
The p-value for Water is 0.035386, which is also less than 0.05. This indicates that there is a statistically
significant difference in crop yield between different water levels.
"""
anova_table["Interaction between Fertilizer and Water"] = """
The p-value for the interaction is 0.272656, which is greater than 0.05. This suggests that there is no statistically significant interaction between Fert and Water.
In other words, the effect of the type of fertilizer on crop yield does not depend on the level of water, and vice versa.
"""
anova_table["*Residual"] = """
The Residual row provides the sum of squares of the residuals, which are the differences between the observed and predicted values.
This row does not provide an F-statistic or p-value.
"""
anova_table["Final Result"] = """
These results suggest that both types of fertilizer and water levels have a significant effect on crop yield when considered individually,
but there's no evidence to suggest that the effect of one depends on the level of the other.
"""

two_way_anova_result = {}
two_way_anova_result["Overall model"] = np.round(model.df_model, decimals=0)
two_way_anova_result["Overall Residuals"] = np.round(model.df_resid, decimals=0)
two_way_anova_result["F-Statistic"] = np.round(model.fvalue, decimals=3)
two_way_anova_result["p_value"] = np.round(model.f_pvalue, decimals=4)
two_way_anova_result["Residuals"] = res
two_way_anova_result["Observation"] = model_summary
two_way_anova_result["Anova table"] = anova_table

# F-Distribution Plot for Two-Way ANOVA
dfn = int(model.df_model)
dfd = int(model.df_resid)

crop_yield_f_fig = dph.plot_f_distribution(
    observed_f=model.fvalue,
    dfn=dfn,
    dfd=dfd,
    title="F‑Distribution: Crop Yield Two‑Way ANOVA"
)

multi_crop_f_fig = dph.plot_multiple_f_distributions(
    df_pairs=df_pairs,
    observed_f=model.fvalue,
    title="F‑Distribution Comparison Across ANOVA Tests for Crop Yield"
)


# F-Test
f_test_notes = """
The F-test is a statistical method used to compare the variances of two or more groups to determine
whether they are significantly different. It is most commonly applied in ANOVA (Analysis of Variance)
to evaluate whether differences in group means are statistically significant.

The test computes an F-statistic, defined as the ratio of variance between groups to variance within groups.
A larger F-statistic indicates greater evidence against the null hypothesis that all group means are equal.
If the p-value associated with the F-statistic is less than the chosen significance level (commonly α = 0.05),
the null hypothesis is rejected.

Key assumptions and guidelines for the F-test include:
1. The population should be approximately normally distributed.
2. Samples must be independent.
3. The larger variance is placed in the numerator, making the test right-tailed.
4. For two-tailed tests, the significance level α is split between both tails.
5. Variances are obtained by squaring standard deviations when necessary.
6. If degrees of freedom are not available in the F-table, the larger critical value is used to reduce Type I error.

Steps to perform an F-test:
1. State the null hypothesis (equal variances) and the alternative hypothesis.
2. Select a significance level (α).
3. Compute the F-statistic: F = Variance between groups / Variance within groups.
4. Compare the calculated F-statistic to the critical value from the F-distribution to make a decision.

The F-test is widely used in fields such as biology, psychology, and social sciences for analyzing
experimental data and drawing conclusions about group variabilit

"""

# TeleCall uses four centers around the globe to process customer order forms. They audit a certain percentage of the
# customer order forms. Any error in the order form renders it defective and has to be reworked before processing. The manager
# wants to check whether the defective percentage varies by center. Analyze the data at the 5 % significance level and help
# the manager draw appropriate inferences.
# Given dataset is categorical in nature, so we will perform the chi-square test for independence to determine
# if there is a relationship between the center and the defective percentage.

# Recommended: Chi - Square Test
customer_contingency = pd.DataFrame({
    "Phillippines": customer_orderForm_df["Phillippines"].value_counts(),
    "Indonesia": customer_orderForm_df["Indonesia"].value_counts(),
    "Malta": customer_orderForm_df["Malta"].value_counts(),
    "India": customer_orderForm_df["India"].value_counts()
}).fillna(0)
b = stats.chi2_contingency(customer_contingency)
customer_chi_square_statistic = b[0]
customer_chi_p_value = b[1]
customer_ddof = b[2]
customer_expected_values = b[3]
customer_chi_result = {}
customer_chi_result["null hypothesis"] = "Ho = There is no relationship between center and defective percentage."
customer_chi_result["alternate hypothesis"] = "Ha = There is a relationship between center and defective percentage."
customer_chi_result["contingency_table"] = customer_contingency
customer_chi_result["F-statistic"] = customer_chi_square_statistic
customer_chi_result["p_value"] = customer_chi_p_value
customer_chi_result["degrees_of_freedom"] = customer_ddof
customer_chi_result["expected_values"] = customer_expected_values
customer_chi_result["alpha"] = alpha
customer_chi_result["criteria"] = "if alpha < .05 then reject the Ho else Failed to reject the Ho"
if customer_chi_p_value < alpha:
    customer_chi_result["hypothesis testing result"] = "Reject the null hypothesis."
else:
    customer_chi_result["hypothesis testing result"] = "Fail to reject the null hypothesis."

# Charts for chi-square distribution
customer_chi_fig = dph.plot_chi_square_distribution(
    observed_chi2=customer_chi_square_statistic,
    dof=customer_ddof,
    title="Chi‑Square Distribution: Customer Order Form"
)

multi_customer_chi_fig = dph.plot_multiple_chi_square_distributions(
    dof_list=dof_list,
    observed_chi2=customer_chi_square_statistic,
    title="Chi‑Square Distribution: Customer Order Form"
)

# Alternatively, One-Way ANOVA (F-Test) across countries
co_mapped_df = customer_orderForm_df.replace(
    {
        "Phillippines": {
            "Error Free": 1, "Defective": 0}, "Indonesia": {
                "Error Free": 1, "Defective": 0}, "Malta": {
                    "Error Free": 1, "Defective": 0}, "India": {
                        "Error Free": 1, "Defective": 0}})


f_stat, f_p_value = f_oneway(co_mapped_df["Phillippines"], co_mapped_df["Indonesia"], co_mapped_df["Malta"], co_mapped_df["India"])
f_test_result = {}
f_test_result["null hypothesis"] = "Ho = There is no relationship between center and defective percentage."
f_test_result["alternate hypothesis"] = "Ha = There is a relationship between center and defective percentage."
f_test_result["F-statistic"] = f_stat
f_test_result["p_value"] = f_p_value
f_test_result["alpha"] = alpha
f_test_result["criteria"] = "if alpha < .05 then reject the Ho else Failed to reject the Ho"
if f_p_value < alpha:
    f_test_result["hypothesis testing result"] = "Reject the null hypothesis."
else:
    f_test_result["hypothesis testing result"] = "Fail to reject the null hypothesis."

# An administrator wants to determine whether there is any significant
# difference in the diameter of cutlets between two units. A randomly
# selected sample of cutlets was collected from both units and measured.
# Analyze the data and draw inferences at a 5% significance level.


# One-way ANOVA
cutlest_f_stat, cutlest_p_value = f_oneway(cutlets_df["Unit A"], cutlets_df["Unit B"])
cutlet_anova_result = {}
cutlet_anova_result["null hypothesis"] = "Ho = There is no significant difference in the diameter of cutlets between two units."
cutlet_anova_result["alternate hypothesis"] = "Ha = There is a significant difference in the diameter of cutlets between two units."
cutlet_anova_result["F-statistic"] = cutlest_f_stat
cutlet_anova_result["p_value"] = cutlest_p_value
cutlet_anova_result["alpha"] = alpha
cutlet_anova_result["criteria"] = "if alpha < .05 then reject the Ho else Failed to reject the Ho"
if cutlest_p_value < alpha:
    cutlet_anova_result["hypothesis testing result"] = "Reject the null hypothesis."
else:
    cutlet_anova_result["hypothesis testing result"] = "Fail to reject the null hypothesis."

# F-Distribution Plot for Cutlets Diameter ANOVA
dfn = 1  # number of groups - 1
dfd = len(cutlets_df) * 2 - 2  # total samples - number of groups

cutlets_f_fig = dph.plot_f_distribution(
    observed_f=cutlest_f_stat,
    dfn=dfn,
    dfd=dfd,
    title="F‑Distribution: Cutlets Diameter ANOVA"
)

multi_cutlets_f_fig = dph.plot_multiple_f_distributions(
    df_pairs=df_pairs,
    observed_f=cutlest_f_stat,
    title="F‑Distribution Comparison Across ANOVA Tests for Cutlets Diameter"
)


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
        builder.card("Chi-Test for Categorical Column result", builder.render_dict(chi_result)),
        builder.card("ANOVA Summary", builder.render_pre(anova_summary)),
        builder.card("One Way Anova Test for Plant", builder.render_dict(one_way_anova_result)),
        builder.card("Two Way Anova Test for Fertilizer and Water", builder.render_dict(two_way_anova_result)),
        builder.card("F-Test Notes", builder.render_pre(f_test_notes)),
        builder.card("Chi-Square Test for Customer Order Form", builder.render_dict(customer_chi_result)),
        builder.card("F-Test for Customer Order Form", builder.render_dict(f_test_result)),
        builder.card("Cutlets One-Way ANOVA Test", builder.render_dict(cutlet_anova_result)),
    ])
)

content.append(builder.chart_grid([
    plotRenderer.plot_to_card(cutlets_f_fig, "F-Distribution Plot for Cutlets Diameter ANOVA"),
    plotRenderer.plot_to_card(plant_f_fig, "F-Distribution Plot for Plant Weights ANOVA"),
    plotRenderer.plot_to_card(crop_yield_f_fig, "F-Distribution Plot for Crop Yield Two-Way ANOVA"),
    plotRenderer.plot_to_card(chi_test_fig, "Chi-Square Distribution Plot for Customer Order Form Chi-Test"),
    plotRenderer.plot_to_card(customer_chi_fig, "Chi-Square Distribution Plot for Customer Order Form Chi-Test"),
    plotRenderer.plot_to_card(multi_cutlets_f_fig, "F-Distribution Comparison Across ANOVA Tests for Cutlets Diameter"),
    plotRenderer.plot_to_card(multi_plant_f_fig, "F-Distribution Comparison Across ANOVA Tests for Plant Weights"),
    plotRenderer.plot_to_card(multi_crop_f_fig, "F-Distribution Comparison Across ANOVA Tests for Crop Yield"),
    plotRenderer.plot_to_card(multi_customer_chi_fig, "Chi-Square Distribution Comparison Across ANOVA Tests for Customer Order Form Chi-Test"),
    plotRenderer.plot_to_card(multi_chi_fig, "Chi-Square Distribution Comparison Across Chi-Test Datasets"),
]))

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
