import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
dashboard = []
builder = HtmlBuilder()
plotRenderer = PlotRenderer()
alpha = .05

# read daraframe
df, report = dl.read_dataset("marketing_data.csv", optimize=True, handle_unnamed="drop", return_report=True)

df_copy = df.copy()

# Ensure Income is numeric (remove $ and commas if needed)
df_copy['Income'] = df_copy['Income'].replace('[\\$,]', '', regex=True).astype(float)

# Group-based mean imputation
df_copy['Income'] = df_copy['Income'].fillna(df_copy.groupby(['Education', 'Marital_Status'])['Income'].transform('mean'))
log_income = df_copy["Income"].transform(lambda x: np.round(np.log10(x), 2) if x > 0 else np.nan)
df_copy = dfh.insert_column_after(
    df_copy, after_col="Income", new_col="Log_Income", values=log_income, inplace=True)

df_copy['Dt_Customer'] = pd.to_datetime(df_copy['Dt_Customer'])
df_copy["ISO3"] = df_copy["Country"].apply(dfh.country_to_iso3)

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

country_revenue_df = (df_clean.groupby("ISO3", as_index=False).agg(Total_Revenue=("TotalSpend", "sum")))

campaign_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]

df_clean["AcceptedAny"] = df_clean[campaign_cols].sum(axis=1).gt(0)

campaign_country_df = (df_clean.groupby("ISO3", as_index=False).agg(Acceptance_Rate=("AcceptedAny", "mean")))

campaign_country_df["Acceptance_Rate"] *= 100

visual_recommednation = """
    Plotly was chosen over Seaborn because:
    - Interactive dashboards improve decision-making
    - Executives can drill down by State, Group, Time
    - Suitable for large datasets (7478 rows)
    - Only those countries are mapped on plot whose ISO
      Code is correct.
    """

marketing_kpi_df = pd.DataFrame([
    {
        "KPI": "Total Customers",
        "Value": f"{len(df_clean):,}",
        "Business Meaning": "Number of customers included after data cleaning"
    },
    {
        "KPI": "Average Customer Income",
        "Value": f"${df_clean['Income'].mean():,.0f}",
        "Business Meaning": "Mean household income used for customer segmentation"
    },
    {
        "KPI": "Average Total Spend per Customer",
        "Value": f"${df_clean['TotalSpend'].mean():,.0f}",
        "Business Meaning": "Average total customer lifetime spending during analysis period"
    },
    {
        "KPI": "Average Purchases per Customer",
        "Value": f"{df_clean['TotPurchase'].mean():.1f}",
        "Business Meaning": "Overall customer purchasing engagement"
    },
    {
        "KPI": "Campaign Acceptance Rate",
        "Value": f"{(df_clean[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']]
                     .sum(axis=1).gt(0).mean() * 100):.1f}%",
        "Business Meaning": "Percentage of customers who accepted at least one marketing campaign"
    },
    {
        "KPI": "Top Revenue Product",
        "Value": product_revenue.iloc[0]['Product'],
        "Business Meaning": "Product category generating the highest total revenue"
    },
    {
        "KPI": "Most Effective Sales Channel",
        "Value": total_purchases_by_channel_without_outliers
        .idxmax().replace('Num', '').replace('Purchases', ''),
        "Business Meaning": "Channel contributing the highest share of total purchases"
    },
    {
        "KPI": "Age-Based Store Purchasing Impact",
        "Value": "Significant",
        "Business Meaning": "Older customers make significantly more in-store purchases than younger customers"
    },
    {
        "KPI": "Family Status vs Online Purchases",
        "Value": "Significant",
        "Business Meaning": "Customers with children purchase less via web, indicating convenience constraints"
    },
    {
        "KPI": "Channel Interaction Effect (Age × Channel)",
        "Value": "Not Significant",
        "Business Meaning": "No interaction effect found; channel preference does not significantly differ by age"
    },
    {
        "KPI": "Store vs Alternate Channel Relationship",
        "Value": "Strong Positive",
        "Business Meaning": "Web and catalog channels complement rather than cannibalize in-store purchases"
    },
    {
        "KPI": "US vs Non‑US Purchase Performance",
        "Value": "Inconclusive",
        "Business Meaning": "Sample size insufficient to confirm higher purchase volume in the US market"
    },
    {
        "KPI": "Outliers Removed (%)",
        "Value": f"{round(100 * (1 - len(df_clean) / len(df_copy)), 2)}%",
        "Business Meaning": "Data cleaned to improve statistical reliability and modeling accuracy"
    }
])

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
                           title='Complaint Count by Education Level', text='Complaint_Count')

marketing_kpi_fig = go.Figure(data=[go.Table(
    columnwidth=[0.25, 0.15, 0.60],
    header=dict(
        values=marketing_kpi_df.columns.tolist(),
        fill_color="#1f3d4f", font=dict(color="white", size=13), align="left", height=38
    ),
    cells=dict(
        values=[marketing_kpi_df[col].tolist() for col in marketing_kpi_df.columns],
        fill_color=[["#f7f9fb", "#ffffff"] * len(marketing_kpi_df)],
        font=dict(color="#1f2933", size=12), align="left", height=34
    ))])

marketing_kpi_fig.update_layout(
    title="Marketing Campaign – Key Performance Indicators", margin=dict(l=10, r=10, t=30, b=10), height=450)

country_revenue_map = px.choropleth(country_revenue_df, locations="ISO3", locationmode="ISO-3", color="Total_Revenue",
                                    color_continuous_scale="Viridis", title="Total Revenue by Country", labels={"Total_Revenue": "Total Revenue"})

country_revenue_map.update_layout(template="none", geo=dict(showframe=False, showcoastlines=True)
                                  )

campaign_map = px.choropleth(campaign_country_df, locations="ISO3", locationmode="ISO-3",
                             color="Acceptance_Rate", color_continuous_scale="Plasma", title="Campaign Acceptance Rate (%) by Country",
                             labels={"Acceptance_Rate": "Acceptance Rate (%)"})

campaign_map.update_layout(template="none")


country_purchase_map = px.choropleth(df_clean.groupby("ISO3", as_index=False).agg(Total_Purchases=("TotPurchase", "sum")),
                                     locations="ISO3", locationmode="ISO-3", color="Total_Purchases", color_continuous_scale="Blues",
                                     title="Total Purchases by Country")

# use for the large dataset
content.append(
    builder.full_width_card("Original Marketing Campaign Interactive Preview", builder.render_dataframe_collapsible(df, initial_rows=15)
                            ))

content.append(
    builder.full_width_card("Modified Marketing Campaign Interactive Preview", builder.render_dataframe_collapsible(df_copy, initial_rows=15)
                            ))

content.append(builder.full_width_card("Outliers Dataframe", builder.render_dataframe_collapsible(df_outliers, initial_rows=15)
                                       ))

content.append(
    builder.full_width_card("Cleaned Dataframe after feature engineering operations", builder.render_dataframe_collapsible(df_clean, initial_rows=15)
                            ))

content.append(
    builder.grid([
        builder.card("Information of the Original Marketing Dataframe is:", builder.render_pre(df_info_str)),
        builder.card("Optimized Dataframe report:", builder.render_pre(report)),
        builder.card("Information of the Modified Marketing Dataframe is:", builder.render_pre(df_copy_info_str)),
        builder.card("Description of the Modified Marketing Dataframe is:", builder.render_dict(df_copy.select_dtypes(
            include=["number"]).describe().to_dict())),
        builder.card("Total Spend channel wise with outliers", builder.render_series(total_purchases_by_channel)),
        builder.card("Description of the Cleaned Marketing Dataframe is:", builder.render_dict(df_clean.select_dtypes(
            include=["number"]).describe().to_dict())),
        builder.card("Total Spend channel wise without outliers", builder.render_series(total_purchases_by_channel_without_outliers)),
        builder.card("Dashboard Navigation",
                     """<a href="marketing_campaign_executive_dashboard.html" target="_blank" style="display:inline-block;padding:12px 18px;background:#1f3b4d;color:white;
               text-decoration:none;border-radius:6px;font-weight:600;">📈 View Interactive Marketing Executive Dashboard
        </a>""")
    ])
)

dashboard.append(
    builder.full_width_card("Execute Summary of the Marketing Campaign Analysis", builder.render_dataframe_collapsible(marketing_kpi_df, 10))
)

dashboard.append(
    builder.grid([
        builder.card("T-Test result for Age vs Store", builder.render_dict(t_result)),
        builder.card("Two Way Anova Test Result for Age vs Store", builder.render_dict(two_way_anova_result)),
        builder.card("T-Test Result for Children vs Web", builder.render_dict(t_child_result)),
        builder.card("Correlation Test Result for Store vs Alternate Channel", builder.render_dict(corr_result_alt_store)),
        builder.card("T-Test Result for US vs Non-US", builder.render_dict(country_stat_result)),
        builder.card("Visualization Recommendation", builder.render_pre(visual_recommednation)),
        builder.card("Dashboard Navigation",
                     """<a href="marketing_campaign_report.html" target="_blank" style="display:inline-block;padding:12px 18px;background:#1f3b4d;color:white;
               text-decoration:none;border-radius:6px;font-weight:600;">📈 View Interactive Marketing Dashboard
        </a>""")
    ]))

content.append(builder.chart_grid([
    plotRenderer.plot_to_card(hist_box_fig_1, " Histogram Box Graph Country as category with Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_2, " Histogram Box Graph Country as category without Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_3, " Histogram Box Graph Age as category with Outliers"),
    plotRenderer.plot_to_card(hist_box_fig_4, " Histogram Box Graph Age as category without Outliers"),
    plotRenderer.plot_to_card(corr_with_outliers_fig, " Correlation Heatmap with Outliers"),
    plotRenderer.plot_to_card(corr_without_ouliers_fig, " Correlation Heatmap without Outliers"),
]))

dashboard.append(
    builder.chart_grid([
        plotRenderer.plot_to_card(country_revenue_map, "Total Revenue by Country"),
        plotRenderer.plot_to_card(campaign_map, "Campaign Acceptance Rate by Country"),
        plotRenderer.plot_to_card(country_purchase_map, "Total Purchases by Country"),
        plotRenderer.plot_to_card(revenue_bar_fig, " Top Revenue Products vs lowest Revenue products"),
        plotRenderer.plot_to_card(corr_age_acceptance_fig, " Correlation Heatmap Age vs Camapiagn Acceptance"),
        plotRenderer.plot_to_card(campaign_accepted_fig, " Customers Who Accepted the Last Campaign by Country"),
        plotRenderer.plot_to_card(children_spend_fig, " Average Total Expenditure vs Number of Children at Home"),
        plotRenderer.plot_to_card(complaint_edu_fig, " Complaint Count by Education Level"),
        plotRenderer.plot_to_card(anova_line_fig, " Two Way ANOVA test Line Graph"),
        plotRenderer.plot_to_card(anova_f_fig, " Two Way ANOVA F-Distribution"),
        plotRenderer.plot_to_card(marketing_kpi_fig, " Marketing Campaign – Key Performance Indicator"),
    ])
)

html_doc = builder.build_page(
    "Marketing Campaign  Report",
    "\n".join(content)
)

dashboard_doc = builder.build_page(
    "Marketing Campaign Execute  Dashboard",
    "\n".join(dashboard)
)

# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "marketing_campaign_report.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)

dashboard_path = ru.save_html_report(
    __file__,
    "marketing_campaign_executive_dashboard.html",   # file name
    dashboard_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=False
)

print(f"Wrote report to: {output_path}")
print(f"Wrote report to: {dashboard_path}")

if __name__ == "__main__":
    main()
