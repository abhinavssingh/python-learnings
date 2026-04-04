import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.html import HtmlBuilder, PlotRenderer
from lib.utility.reports.report_utils import ReportUtils as ru


def main():
    # your current script code goes here
    print("Running Pandas DataFrame Data Visualization report...")
    # ...


# Build full column-wise page
builder = HtmlBuilder()
plotRenderer = PlotRenderer()

df_canada = px.data.gapminder().query("country=='Canada'")
line_fig_1 = px.line(df_canada, x="year", y="lifeExp", title='Life expectancy in Canada')
bar_fig_1 = px.bar(df_canada, x='year', y='pop')
bar_fig_2 = px.bar(df_canada, x='year', y='pop',
                   hover_data=['lifeExp', 'gdpPercap'], color='lifeExp',
                   labels={'pop': 'population of Canada'}, height=400)
bar_fig_3 = px.bar(df_canada, x='year', y='pop', orientation='h')

df_population = px.data.gapminder()
area_fig = px.area(df_population, x="year", y="pop", color="continent", line_group="country")

df_europe = px.data.gapminder().query("year == 2007").query("continent == 'Europe'")
df_europe.loc[df_europe['pop'] < 2.e6, 'country'] = 'Other countries'  # Represent only large countries
pie_fig_1 = px.pie(df_europe, values='pop', names='country', title='Population of European continent')

df_continent = px.data.gapminder().query("continent=='Oceania'")
line_fig_2 = px.line(df_continent, x="year", y="lifeExp", color='country')

df_year_2007 = px.data.gapminder().query("year == 2007")
heatmap_fig = px.treemap(df_year_2007, path=[px.Constant("world"), 'continent', 'country'], values='pop',
                         color='lifeExp', hover_data=['iso_alpha'],
                         color_continuous_scale='RdBu',
                         color_continuous_midpoint=np.average(df_year_2007['lifeExp'], weights=df_year_2007['pop']))
heatmap_fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

df_flower = px.data.iris()
scatter_fig = px.scatter(df_flower, x="sepal_width", y="sepal_length", color="species", size='petal_length', hover_data=['petal_width'])
scatter_fig_2 = px.scatter(df_flower, x="sepal_width", y="sepal_length", color='petal_length')

df_hotel = px.data.tips()
pie_fig_2 = px.pie(df_hotel, values='tip', names='day')
box_fig = px.box(df_hotel, x="day", y="total_bill", color="smoker")
box_fig.update_traces(quartilemethod="exclusive")  # or "inclusive", or "linear" by default
hist_fig = px.histogram(df_hotel, x="total_bill", color="sex")
violin_fig = px.violin(df_hotel, y="tip", x="smoker", color="sex", box=True, points="all",
                       hover_data=df_hotel.columns)

df_wind = px.data.wind()
polar_fig = px.line_polar(df_wind, r="frequency", theta="direction", color="strength", line_close=True,
                          color_discrete_sequence=px.colors.sequential.Plasma_r,
                          template="plotly_dark",)

fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'polar'}] * 2] * 2)

fig.add_trace(go.Scatterpolar(
    name="angular categories",
    r=[5, 4, 2, 4, 5],
    theta=["a", "b", "c", "d", "a"],
), 1, 1)
fig.add_trace(go.Scatterpolar(
    name="radial categories",
    r=["a", "b", "c", "d", "b", "f", "a"],
    theta=[1, 4, 2, 1.5, 1.5, 6, 5],
    thetaunit="radians",
), 1, 2)
fig.add_trace(go.Scatterpolar(
    name="angular categories (w/ categoryarray)",
    r=[5, 4, 2, 4, 5],
    theta=["a", "b", "c", "d", "a"],
), 2, 1)
fig.add_trace(go.Scatterpolar(
    name="radial categories (w/ category descending)",
    r=["a", "b", "c", "d", "b", "f", "a", "a"],
    theta=[45, 90, 180, 200, 300, 15, 20, 45],
), 2, 2)

fig.update_traces(fill='toself')
fig.update_layout(
    polar=dict(
        radialaxis_angle=-45,
        angularaxis=dict(
            direction="clockwise",
            period=6)
    ),
    polar2=dict(
        radialaxis=dict(
            angle=180,
            tickangle=-180  # so that tick labels are not upside down
        )
    ),
    polar3=dict(
        sector=[80, 400],
        radialaxis_angle=-45,
        angularaxis_categoryarray=["d", "a", "c", "b"]
    ),
    polar4=dict(
        radialaxis_categoryorder="category descending",
        angularaxis=dict(
            thetaunit="radians",
            dtick=0.3141592653589793
        ))
)

html_doc = builder.build_page(
    "Pandas Data Visulaization through Various chart",
    builder.chart_grid([
        plotRenderer.plot_to_card(line_fig_1, "Life Expectancy in Canada Over Time"),
        plotRenderer.plot_to_card(line_fig_2, "Life Expectancy Trends in Oceania by Country"),
        plotRenderer.plot_to_card(bar_fig_1, "Population of Canada by Year"),
        plotRenderer.plot_to_card(bar_fig_2, "Population of Canada by Year (Colored by Life Expectancy)"),
        plotRenderer.plot_to_card(bar_fig_3, "Population of Canada by Year (Horizontal Bar Chart)"),
        plotRenderer.plot_to_card(area_fig, "Population Growth by Continent Over Time"),
        plotRenderer.plot_to_card(pie_fig_1, "Population Distribution in Europe (2007)"),
        plotRenderer.plot_to_card(heatmap_fig, "Global Population and Life Expectancy by Continent and Country (2007)"),
        plotRenderer.plot_to_card(scatter_fig, "Iris Dataset: Sepal Width vs Sepal Length"),
        plotRenderer.plot_to_card(scatter_fig_2, "Iris Dataset: Sepal Dimensions Colored by Petal Length"),
        plotRenderer.plot_to_card(box_fig, "Total Bill Distribution by Day and Smoker Status"),
        plotRenderer.plot_to_card(pie_fig_2, "Tip Distribution by Day"),
        plotRenderer.plot_to_card(hist_fig, "Total Bill Distribution by Gender"),
        plotRenderer.plot_to_card(violin_fig, "Tip Distribution by Gender and Smoker Status"),
        plotRenderer.plot_to_card(polar_fig, "Wind Frequency by Direction and Strength"),
        plotRenderer.plot_to_card(fig, "Categorical Polar Chart"),
    ])
)

# html_doc is the string you already have
output_path = ru.save_html_report(
    __file__,
    "Pandas Data Visulaization through Various chart.html",   # file name
    html_doc,
    subfolder="reports",                # or 'reports' to keep files in a subdir
    open_in_browser=True
)
print(f"Wrote report to: {output_path}")


if __name__ == "__main__":
    main()
