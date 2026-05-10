import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from scipy import stats
from scipy.stats import spearmanr
from sklearn.preprocessing import OneHotEncoder
from statsmodels.formula.api import ols

from lib.html import HtmlBuilder, PlotRenderer
from lib.plothelper.PlotHelper import DistributionPlotHelper as dph
from lib.utility.dataframe.data_loader import DataLoader as dl
from lib.utility.dataframe.df_helper import DataFrameHelper as dfh
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running marketing campaign report...")
    # ...


# initialization and set variable
content = []
builder = HtmlBuilder()
plotRenderer = PlotRenderer()
alpha = .05

# read daraframe
df, report = dl.read_dataset("marketing_data.csv", optimize=True, handle_unnamed="drop", return_report=True)

# Income column has white space to trim the white space
df.columns = df.columns.str.strip()
df_copy = df.copy()

# Ensure Income is numeric (remove $ and commas if needed)
df_copy['Income'] = df_copy['Income'].replace('[\\$,]', '', regex=True).astype(float)

# Group-based mean imputation
df_copy['Income'] = df_copy['Income'].fillna(df_copy.groupby(['Education', 'Marital_Status'])['Income'].transform('mean'))
log_income = df_copy["Income"].transform(lambda x: np.round(np.log10(x), 2) if x > 0 else np.nan)
df_copy = dfh.insert_column_after(
    df_copy, after_col="Income", new_col="Log_Income", values=log_income, inplace=True)

df_copy['Dt_Customer'] = pd.to_datetime(df_copy['Dt_Customer'])

# Create variables to represent the total number of children, age, and total spending.
df_copy = dfh.insert_column_after(
    df_copy, after_col="Teenhome", new_col="Totalchildren", values=df_copy['Kidhome'] + df_copy['Teenhome'], inplace=True)

df_copy = dfh.insert_column_after(
    df_copy, after_col="Year_Birth", new_col="Age", values=df_copy["Dt_Customer"].dt.year - df_copy['Year_Birth'], inplace=True)

# Select columns that contain 'Mnt' and calculate row-wise sum
Total_Mnt = df_copy.loc[:, df_copy.columns.str.contains('Mnt')].sum(axis=1)

df_copy = dfh.insert_column_after(
    df_copy, after_col="MntGoldProds", new_col="TotalSpend", values=Total_Mnt, inplace=True)


age_category = pd.qcut(df_copy['Age'], q=4, labels=['Young', 'Mid-Age', 'Senior', 'Elder'])
df_copy = dfh.insert_column_after(
    df_copy, after_col="Age", new_col="Age_Category", values=age_category, inplace=True)
age_map = {'Young': 1, 'Mid-Age': 2, 'Senior': 3, 'Elder': 4}
age_category_code = df_copy['Age_Category'].map(age_map)
df_copy = dfh.insert_column_after(
    df_copy, after_col="Age_Category", new_col="Age_Category_Code", values=age_category_code, inplace=True)

income_category = pd.qcut(df_copy['Income'], q=4, labels=['Low Income', 'Lower-Middle', 'Upper-Middle', 'High Income'])
df_copy = dfh.insert_column_after(
    df_copy, after_col="Income", new_col="Income_Category", values=income_category, inplace=True)
income_map = {'Low Income': 1, 'Lower-Middle': 2, 'Upper-Middle': 3, 'High Income': 4}
income_category_code = df_copy['Income_Category'].map(income_map)
df_copy = dfh.insert_column_after(
    df_copy, after_col="Income_Category", new_col="Income_Category_Code", values=income_category_code, inplace=True)

# there are 3 channels web, catalogue and store
channel_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']

Total_Purchases_per_Customer = df_copy[channel_cols].sum(axis=1)
df_copy = dfh.insert_column_after(
    df_copy, after_col="NumStorePurchases", new_col="TotPurchase", values=Total_Purchases_per_Customer, inplace=True)

total_purchases_by_channel = df_copy[channel_cols].sum()

# Get information about the DataFrame
df_info_str = dfh.get_dataframe_info_str(df)
df_copy_info_str = dfh.get_dataframe_info_str(df_copy)

# apply one hot encoding through scikit preprocessing
ohe = OneHotEncoder(
    drop='first',        # avoids dummy variable trap
    sparse_output=False  # returns NumPy array (not sparse matrix)
)

encoded_array = ohe.fit_transform(
    df_copy[['Education', 'Marital_Status']]
)

encoded_df = pd.DataFrame(
    encoded_array,
    columns=ohe.get_feature_names_out(['Education', 'Marital_Status']),
    index=df_copy.index
)

df_copy = pd.concat(
    [df_copy.drop(columns=['Education', 'Marital_Status']), encoded_df],
    axis=1
)

# identify outliers and remove outliers
df_outliers = dfh.find_iqr_outliers(df_copy, columns=["Log_Income", "Age"], groupby="Country")
df_clean = dfh.remove_outliers(df_outliers)
total_purchases_by_channel_without_outliers = df_clean[channel_cols].sum()

corr_columns = ['Age', 'Income', 'Totalchildren', 'TotalSpend']

product_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

product_revenue = (df_clean.melt(value_vars=product_cols, var_name='Product', value_name='Revenue')
                   .groupby('Product', as_index=False)['Revenue'].sum())

product_revenue = product_revenue.sort_values(by='Revenue', ascending=False)

campaign_country = (df_clean[df_clean['AcceptedCmp5'] == 1].groupby('Country')
                    .size().reset_index(name='Accepted_Customers').sort_values(by='Accepted_Customers', ascending=False))

children_spend = (df_clean.groupby('Totalchildren', as_index=False)['TotalSpend'].mean())


education_cols = df_clean.columns[df_clean.columns.str.startswith('Education_')]


education_complaints = (df_clean.loc[df_clean['Complain'] == 1, education_cols]
                        .sum().reset_index().rename(columns={'index': 'Education', 0: 'Complaint_Count'}))

education_complaints['Education'] = (education_complaints['Education'].str.replace('Education_', '', regex=False))


# t and ANOVA Test | Older individuals may not possess the same level
# of technological proficiency and may, therefore, lean toward traditional
# in-store shopping preferences.
young_group = df_clean.loc[df_clean['Age_Category'].isin(['Young', 'Mid-Age']), 'NumStorePurchases']

older_group = df_clean.loc[df_clean['Age_Category'].isin(['Senior', 'Elder']), 'NumStorePurchases']

ttest, pval = stats.ttest_ind(older_group, young_group, equal_var=False, alternative='greater')
t_result = {}
t_result["null hypothesis"] = "Ho = There is no difference in the average number of store purchases between younger and older individuals."
t_result["alternate hypothesis"] = "Ha = Older individuals have a higher average number of in‑store purchases than younger individuals."
t_result["t_statistics"] = ttest
t_result["p_value"] = np.round(pval, decimals=4)
t_result["alpha"] = alpha
t_result["criteria"] = "if alpha < .05 then reject the Ho else Failed to reject the Ho"
if np.round(pval, decimals=4) < alpha:
    t_result["hypothesis testing result"] = "Reject the null hypothesis."
else:
    t_result["hypothesis testing result"] = "Fail to reject the null hypothesis."

# Two-Way ANOVA TEST

two_way_anova_df = df_clean.melt(id_vars=['Age_Category'], value_vars=['NumStorePurchases', 'NumWebPurchases'],
                                 var_name='Channel', value_name='Purchases')

two_way_anova_df['Channel'] = two_way_anova_df['Channel'].replace({'NumStorePurchases': 'Store', 'NumWebPurchases': 'Web'})


anova_mean_df = (two_way_anova_df.groupby(['Age_Category', 'Channel'])['Purchases'].mean().reset_index())


model = ols(
    'Purchases ~ C(Age_Category) * C(Channel)',
    data=two_way_anova_df
).fit()

anova_table = sm.stats.anova_lm(model, typ=2)
two_way_anova_result = {}
two_way_anova_result["null hypothesis"] = """Ho = Mean purchases are equal across all age categories.
"Mean purchases are equal across Store and Web channels."""
two_way_anova_result["alternate hypothesis"] = """Ha = There is NO interaction between age and channel preference.
Channel preference differs by age group."""
two_way_anova_result["Overall model"] = np.round(model.df_model, decimals=0)
two_way_anova_result["Overall Residuals"] = np.round(model.df_resid, decimals=0)
two_way_anova_result["F-Statistic"] = np.round(model.fvalue, decimals=3)
two_way_anova_result["p_value"] = np.round(model.f_pvalue, decimals=4)
two_way_anova_result["Residuals"] = anova_table
interaction_value = anova_table.loc['C(Age_Category):C(Channel)', 'PR(>F)']
if np.round(interaction_value, decimals=4) < alpha:
    two_way_anova_result["hypothesis testing result"] = "Reject the null hypothesis."
else:
    two_way_anova_result["hypothesis testing result"] = "Fail to reject the null hypothesis."

# F-Distribution Plot for Two-Way ANOVA
dfn = int(model.df_model)
dfd = int(model.df_resid)

anova_line_fig = px.line(anova_mean_df, x='Channel', y='Purchases', color='Age_Category', markers=True,
                         title='Interaction Effect: Age Category × Purchase Channel')

anova_f_fig = dph.plot_f_distribution(observed_f=model.fvalue, dfn=dfn, dfd=dfd,
                                      title="F‑Distribution:  Two‑Way ANOVA")

# T-Test | Customers with children likely experience time constraints, making online shopping a more convenient option.

group_no_children = df_clean.loc[df_clean['Totalchildren'] == 0, 'NumWebPurchases']

group_with_children = df_clean.loc[df_clean['Totalchildren'] > 0, 'NumWebPurchases']

t_child_stat, p_child_value = stats.ttest_ind(group_with_children, group_no_children, equal_var=False)

t_child_result = {}
t_child_result[
    "null hypothesis"] = "Ho = Customers with children do not differ from customers without children in their use of online (web) purchasing."
t_child_result["alternate hypothesis"] = "Ha = Customers with children make significantly more online (web) purchases than customers without children."
t_child_result["t_statistics"] = t_child_stat
t_child_result["p_value"] = np.round(p_child_value, decimals=4)
t_child_result["alpha"] = alpha
t_child_result["criteria"] = "if alpha < .05 then reject the Ho else Failed to reject the Ho"
if np.round(p_child_value, decimals=4) < alpha:
    t_child_result["hypothesis testing result"] = "Reject the null hypothesis."
else:
    t_child_result["hypothesis testing result"] = "Fail to reject the null hypothesis."

# Correlation test | Sales at physical stores may face the risk of cannibalization by alternative distribution channels.

df_clean['Alt_Channel_Purchases'] = (df_clean['NumWebPurchases'] + df_clean['NumCatalogPurchases'])

corr_alt_store, p_value_alt_store = spearmanr(df_clean['Alt_Channel_Purchases'], df_clean['NumStorePurchases'])
corr_result_alt_store = {}
corr_result_alt_store[
    "null hypothesis"] = "Ho = There is no relationship between purchases made through alternative channels (Web and Catalog) and purchases made at physical stores."
corr_result_alt_store[
    "alternate hypothesis"] = "Ha = There is a negative relationship between purchases made through alternative channels (Web and Catalog) and purchases made at physical stores."
corr_result_alt_store["corr"] = corr_alt_store
corr_result_alt_store["p_value"] = np.round(p_value_alt_store, decimals=4)
corr_result_alt_store["alpha"] = alpha
corr_result_alt_store["criteria"] = "if alpha < .05 then reject the Ho else Failed to reject the Ho"
if np.round(p_value_alt_store, decimals=4) < alpha:
    corr_result_alt_store["hypothesis testing result"] = "Reject the null hypothesis."
else:
    corr_result_alt_store["hypothesis testing result"] = "Fail to reject the null hypothesis."

# T-Test | Does the United States significantly outperform the rest of the world in total purchase volumes?

us_purchases = df_clean.loc[df_clean['Country'] == 'United States', 'TotPurchase']
non_us_purchases = df_clean.loc[df_clean['Country'] != 'United States', 'TotPurchase']

u_stat, p_stat_value = stats.ttest_ind(us_purchases, non_us_purchases, alternative='greater')

country_stat_result = {}
country_stat_result[
    "null hypothesis"] = "Ho = Customers from the United States do not have higher total purchase volumes than customers from the rest of the world."
country_stat_result[
    "alternate hypothesis"] = "Ha = Customers from the United States have significantly higher total purchase volumes than customers from the rest of the world."
country_stat_result["t-statistic"] = u_stat
country_stat_result["p_value"] = np.round(p_stat_value, decimals=4)
country_stat_result["alpha"] = alpha
country_stat_result["criteria"] = "if alpha < .05 then reject the Ho else Failed to reject the Ho"
if np.round(p_stat_value, decimals=4) < alpha:
    country_stat_result["hypothesis testing result"] = "Reject the null hypothesis."
else:
    country_stat_result["hypothesis testing result"] = "Fail to reject the null hypothesis."
country_stat_result["remarks"] = """SmallSampleWarning: One or more sample arguments is too small; all returned values will be NaN.
See documentation for sample size requirements."""

# data visualization
hist_box_fig_1 = px.histogram(df_copy, x="Log_Income", color="Country", marginal="box", opacity=0.7, barmode="overlay",
                              title="Histogram Box Graph Country as category with Outliers",
                              labels={"Log_Income": "Income"}, hover_data=df_copy.columns)
hist_box_fig_2 = px.histogram(df_clean, x="Log_Income", color="Country", marginal="box", opacity=0.7, barmode="overlay",
                              title="Histogram Box Graph Country as category without Outliers",
                              labels={"Log_Income": "Income"}, hover_data=df_clean.columns)
hist_box_fig_3 = px.histogram(df_copy, x="Log_Income", color="Age_Category", marginal="box", opacity=0.7, barmode="overlay",
                              title="Histogram Box Graph Age as category with Outliers",
                              labels={"Log_Income": "Income"}, hover_data=df_copy.columns)
hist_box_fig_4 = px.histogram(df_clean, x="Log_Income", color="Age_Category", marginal="box", opacity=0.7, barmode="overlay",
                              title="Histogram Box Graph Age as category without Outliers",
                              labels={"Log_Income": "Income"}, hover_data=df_clean.columns)
corr_with_outliers_fig = px.imshow(df_copy[corr_columns].corr(), text_auto='.2f',
                                   color_continuous_scale='RdBu', title='Correlation Matrix with Outliers')

corr_without_ouliers_fig = px.imshow(df_clean[corr_columns].corr(), text_auto='.2f',
                                     color_continuous_scale='Oxy', title='Correlation Matrix without Outliers')

corr_age_acceptance_fig = px.imshow(df_clean[['Age', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].corr(),
                                    text_auto='.2f', color_continuous_scale='viridis', title='Correlation Matrix Age vs Campaign Acceptance')


revenue_bar_fig = px.bar(product_revenue, x='Product', y='Revenue', color='Revenue', text='Revenue',
                         title='Product-wise Revenue Performance', color_continuous_scale='Viridis')

revenue_bar_fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

revenue_bar_fig.update_layout(xaxis_title='Product Category', yaxis_title='Total Revenue', showlegend=False)


campaign_accepted_fig = px.bar(campaign_country, x='Country', y='Accepted_Customers',
                               title='Customers Who Accepted the Last Campaign by Country', color='Country')

children_spend_fig = px.line(children_spend, x='Totalchildren', y='TotalSpend', markers=True,
                             title='Average Total Expenditure vs Number of Children at Home')

children_spend_fig.update_layout(xaxis_title='Number of Children at Home', yaxis_title='Average Total Expenditure')


complaint_edu_fig = px.bar(education_complaints, x='Education', y='Complaint_Count',
                           title='Complaint Count by Education Level', text='Complaint Count')


insights_pre = """
1. During the dataframe read only, i have optimized the dataframe
means changing the datatype.
2. There was an empty space in the Income column that i removed.
3. Removed $ and comma from Income column value.
4. Converted Dt_Customer column to datetime.
5. There were 24 places where Income vale was missing. So, I have
updated as per their education/marital_status groups average.
6. Performed Outlier techniques and removed outliears.
7. Visualize the data with and without outliers
8. All Hypothesis test results are published as PRE-TEXT into report.
9. For last hypothesis testing data is insufficient that's why NAN in
the report. We are getting SmallSampleWarning: One or more sample arguments
is too small; all returned values will be NaN. See documentation for sample
size requirements.
10. All required visualization is in the HTML report.
"""

# use for the large dataset
content.append(
    builder.full_width_card(
        "Original Marketing Campaign Interactive Preview",
        builder.render_dataframe_collapsible(df, initial_rows=15)
    )
)

content.append(
    builder.full_width_card(
        "Modified Marketing Campaign Interactive Preview",
        builder.render_dataframe_collapsible(df_copy, initial_rows=15)
    )
)

content.append(
    builder.full_width_card(
        "Outliers Dataframe",
        builder.render_dataframe_collapsible(df_outliers, initial_rows=15)
    )
)

content.append(
    builder.full_width_card(
        "Cleaned Dataframe after feature engineering operations",
        builder.render_dataframe_collapsible(df_clean, initial_rows=15)
    )
)


content.append(
    builder.grid(
        [
            builder.card("Information of the Original Marketing Dataframe is:", builder.render_pre(df_info_str)),
            builder.card("Optimized Dataframe report:", builder.render_pre(report)),
            builder.card("Information of the Modified Marketing Dataframe is:", builder.render_pre(df_copy_info_str)),
            builder.card("Description of the Modified Marketing Dataframe is:", builder.render_dict(df_copy.select_dtypes(
                include=["number"]).describe().to_dict())),
            builder.card("Total Spend channel wise with outliers", builder.render_series(total_purchases_by_channel)),
            builder.card("Description of the Cleaned Marketing Dataframe is:", builder.render_dict(df_clean.select_dtypes(
                include=["number"]).describe().to_dict())),
            builder.card("Total Spend channel wise without outliers", builder.render_series(total_purchases_by_channel_without_outliers)),
            builder.card("Data Insights after analysis of the Marketing campaign data", builder.render_pre(insights_pre)),
            builder.card("T-Test result for Age vs Store", builder.render_dict(t_result)),
            builder.card("Two Way Anova Test Result for Age vs Store", builder.render_dict(two_way_anova_result)),
            builder.card("T-Test Result for Children vs Web", builder.render_dict(t_child_result)),
            builder.card("Correlation Test Result for Store vs Alternate Channel", builder.render_dict(corr_result_alt_store)),
            builder.card("T-Test Result for US vs Non-US", builder.render_dict(country_stat_result)),
        ])
)

content.append(builder.chart_grid([
    plotRenderer.plot_to_card(hist_box_fig_1, " Histogram Box Graph Country as category with Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_2, " Histogram Box Graph Country as category without Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_3, " Histogram Box Graph Age as category with Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_4, " Histogram Box Graph Age as category without Outliers"),
    plotRenderer.plot_to_card(corr_with_outliers_fig, " Correlation Heatmap with Outliers"),
    plotRenderer.plot_to_card(corr_without_ouliers_fig, " Correlation Heatmap without Outliers"),
    plotRenderer.plot_to_card(anova_line_fig, " Two Way ANOVA test Line Graph"),
    plotRenderer.plot_to_card(anova_f_fig, " Two Way ANOVA F-Distribution"),
    plotRenderer.plot_to_card(revenue_bar_fig, " Top Revenue Products vs lowest Revenue products"),
    plotRenderer.plot_to_card(corr_age_acceptance_fig, " Correlation Heatmap Age vs Camapiagn Acceptance"),
    plotRenderer.plot_to_card(campaign_accepted_fig, " Customers Who Accepted the Last Campaign by Country"),
    plotRenderer.plot_to_card(children_spend_fig, " Average Total Expenditure vs Number of Children at Home"),
    plotRenderer.plot_to_card(complaint_edu_fig, " Complaint Count by Education Level"),
]))

html_doc = builder.build_page(
    "Marketing Campaign  Report",
    "\n".join(content)
)


# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "marketing_campaign_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
