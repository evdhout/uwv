import pandas as pd
import plotly.express as px

from uwv.config import OUTPUT_DIR
from uwv.visualization.get_slp_data import get_slp_data


def easi_lab_1_exp_plot(order: str = "descending"):
    slp_year = get_slp_data(period="JJ", max_year=2022)
    slp_quarter = get_slp_data(period="KW", max_year=2022)

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
        fig.update_yaxes(categoryorder=f"median {order}")
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
        fig2.write_html(OUTPUT_DIR / f"easi_lab_1_exp_plot_quarter_{q}.html")


if __name__ == "__main__":
    easi_lab_1_exp_plot()
