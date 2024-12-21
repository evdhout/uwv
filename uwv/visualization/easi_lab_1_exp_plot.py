import pandas as pd
import plotly.express as px

from uwv.config import OUTPUT_DIR
from uwv.visualization.get_slp_data import get_slp_data


def easi_lab_1_exp_plot(order: str = "descending"):
    slp_year = get_slp_data(period="JJ", max_year=2022)
    slp_year.sbi_title = slp_year.apply(
        lambda row: "Q Health care" if row.sbi_title.startswith("Q") else row.sbi_title, axis=1
    )
    slp_year.sbi_title = slp_year.sbi_title.astype("category")
    slp_quarter = get_slp_data(period="KW", max_year=2022)
    slp_quarter.sbi_title = slp_quarter.apply(
        lambda row: "Q Health care" if row.sbi_title.startswith("Q") else row.sbi_title, axis=1
    )
    slp_quarter.sbi_title = slp_quarter.sbi_title.astype("category")

    for df, label in zip((slp_year, slp_quarter), ("year", "quarter")):
        fig = px.box(
            df,
            x="sick_leave_percentage",
            y="sbi_title",
            orientation="h",
            color="parenttitle",
            title="No relation to sick leave percentage within business sectors",
            labels={
                "sbi_title": "Branch",
                "parenttitle": "Sector",
                "sick_leave_percentage": "Sick leave percentage",
            },
        )
        fig.update_yaxes(
            categoryorder=f"median {order}",
        )
        fig.update_traces(
            # marker_line_width=1,
            # marker_line_color="black",
            width=0.5,
        )
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=1000,
            height=600,
            xaxis=dict(
                title=dict(font=dict(size=12)),
                gridcolor="lightgray",
                showgrid=True,
            ),
            yaxis=dict(
                title=dict(font=dict(size=12)),
                showgrid=False,
            ),
            title_font=dict(size=14),
            boxgap=0,
            boxgroupgap=0,
            legend=dict(
                valign="middle",
                bgcolor="rgba(0,0,0,0)",
                title="",
                traceorder="normal",
                orientation="h",  # Horizontal orientation
                x=0.5,  # Center horizontally
                y=-0.2,  # Position below the plot, adjust the value as needed
                xanchor="center",  # Horizontally center the legend
                yanchor="top",  # Align the legend’s top with the specified `y` position,
            ),
        )
        fig.write_html(OUTPUT_DIR / f"easi_lab_1_exp_plot_{label}_{order}.html")

    for q in [1, 2, 3, 4]:
        fig2 = px.box(
            slp_quarter[slp_quarter.period_quarter_number == q],
            x="sick_leave_percentage",
            y="sbi_title",
            orientation="h",
            color="parenttitle",
            title=f"Quarter {q}",
            labels={
                "sbi_title": "Branch",
                "parenttitle": "Sector",
                "sick_leave_percentage": "Sick leave percentage",
            },
        )
        fig2.update_yaxes(categoryorder="median ascending")
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                title="",
                orientation="h",  # Horizontal orientation
                x=0.5,  # Center horizontally
                y=-0.2,  # Position below the plot, adjust the value as needed
                xanchor="center",  # Horizontally center the legend
                yanchor="top",  # Align the legend’s top with the specified `y` position
            ),
        )
        fig2.write_html(OUTPUT_DIR / f"easi_lab_1_exp_plot_quarter_{q}.html")


if __name__ == "__main__":
    easi_lab_1_exp_plot()
