import math

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import typer
from loguru import logger

from uwv.config import BASELINE_PARQUET, BASELINE_MEAN_ERRORS_PARQUET, OUTPUT_DIR
from uwv.translations.sbi_translate import sbi_translate

app = typer.Typer(no_args_is_help=True)
baseline: pd.DataFrame
baseline_errors: pd.DataFrame

TOTAL_SBI = "T001081"
SBI_LIST = ["307500", "354200", "422400"]
SBI_LIST_WITH_TOTAL = SBI_LIST + [TOTAL_SBI]

SBI_TITLE_DICT = {}


def get_sbi_title(sbi: str):
    return baseline[baseline.sbi == sbi]["sbi_title"].values[0]


def translate_sbi_title(sbi: str, sbi_title: str):
    return sbi_translate(sbi_title) if sbi != TOTAL_SBI else "All economical activities"


def load_dataframe():
    global baseline
    global baseline_errors
    global SBI_TITLE_DICT

    try:
        baseline = pd.read_parquet(BASELINE_PARQUET)
    except FileNotFoundError:
        logger.error(f"File {BASELINE_PARQUET} not found.")
        raise typer.Abort()

    try:
        baseline_errors = pd.read_parquet(BASELINE_MEAN_ERRORS_PARQUET)
    except FileNotFoundError:
        logger.error(f"File {BASELINE_MEAN_ERRORS_PARQUET} not found.")
        raise typer.Abort()

    # translate the sbi descriptions to dutch
    logger.debug("Translating SBI titles to Dutch")
    baseline.sbi_title = baseline.apply(
        lambda row: translate_sbi_title(row.sbi, row.sbi_title), axis=1
    )
    baseline.sbi_title = baseline.sbi_title.astype("category")

    baseline_errors.sbi_title = baseline_errors.apply(
        lambda row: translate_sbi_title(row.sbi, row.sbi_title), axis=1
    )
    baseline_errors.sbi_title = baseline_errors.sbi_title.astype("category")

    # create a text string for the mean's
    baseline_errors["formatted_mean_absolute_error"] = (
        baseline_errors["mean_absolute_error_ra"].round(3).astype(str)
    )

    # create the title dictionary for easy reference when creating viz
    logger.debug("Creating SBI title dictionary")
    SBI_TITLE_DICT = dict(zip(baseline.sbi, baseline.sbi_title))


def viz_predictions(sbi: str, viz_type: str = "KW"):
    baseline_branches = baseline[(baseline.sbi == sbi) & (baseline.period_type == viz_type)]

    # Filter periods for the first quarter of each year
    first_quarter_periods = baseline_branches[
        baseline_branches["period"].astype(str).str.endswith("1")
    ]["period"].unique()

    fig = px.line(
        baseline_branches,
        x="period",
        y="ra_prediction",
        color="sbi_title",
        title=f"Rolling averages for branch {SBI_TITLE_DICT[sbi]}",
    )

    fig.add_scatter(
        x=baseline_branches.period,
        y=baseline_branches.sick_leave_percentage,
        mode="markers",
        name="Actual values",
    )

    # Customize x-axis to show ticks only for the first quarters
    fig.update_layout(
        xaxis=dict(
            tickvals=first_quarter_periods,
            ticktext=[str(period)[:4] for period in first_quarter_periods],
        ),
        legend=dict(
            title="",
            orientation="h",  # Horizontal orientation
            x=0.5,  # Center horizontally
            y=-0.2,  # Position below the plot, adjust the value as needed
            xanchor="center",  # Horizontally center the legend
            yanchor="top",  # Align the legend’s top with the specified `y` position
        ),
    )

    fig.show()


def viz_multiple_branches(sbi_list: list, viz_type: str = "KW"):
    fig = go.Figure()

    baseline_all_branches = baseline[(baseline.sbi.isin(sbi_list)) & (baseline.period_type == viz_type)]

    # Ensure the period column is formatted correctly
    first_quarter_periods = baseline_all_branches[
        baseline_all_branches["period"].astype(str).str.endswith("1")
    ]["period"].unique()

    for sbi in sbi_list:
        baseline_branches = baseline[(baseline.sbi == sbi) & (baseline.period_type == viz_type)]

        # Add line for ra_prediction
        fig.add_trace(
            go.Scatter(
                x=baseline_branches["period"],
                y=baseline_branches["ra_prediction"],
                mode="lines",
                name=f"{SBI_TITLE_DICT[sbi]} - Predicted",
                line=dict(color=px.colors.qualitative.Plotly[sbi_list.index(sbi) % 10]),
            )
        )

        # Add markers for actual values
        fig.add_trace(
            go.Scatter(
                x=baseline_branches["period"],
                y=baseline_branches["sick_leave_percentage"],
                mode="markers",
                name=f"{SBI_TITLE_DICT[sbi]} - Actual",
                marker=dict(color=px.colors.qualitative.Plotly[sbi_list.index(sbi) % 10]),
            )
        )

    # Customize the layout
    fig.update_layout(
        plot_bgcolor="rgb(0,0,0,0)",
        paper_bgcolor="rgb(0,0,0,0)",
        title="Sick leave percentage of predections and actual values",
        xaxis_title="Period",
        yaxis_title="Prediction/Actual Values",
        xaxis=dict(
            tickvals=first_quarter_periods,
            ticktext=[str(period)[:4] for period in first_quarter_periods],
            # Display the year only for the first quarter
            showgrid=False,
        ),
        yaxis=dict(
            showgrid=False,
        ),
        legend=dict(
            title="",
            orientation="h",  # Horizontal orientation
            x=0.5,  # Center horizontally
            y=-0.2,  # Position below the plot, adjust the value as needed
            xanchor="center",  # Horizontally center the legend
            yanchor="top",  # Align the legend’s top with the specified `y` position
        ),
    )

    fig.show()


def viz_mean_absolute_errors(sbi_list: list):
    baseline_branch_errors = baseline_errors[
        (baseline_errors.sbi.isin(sbi_list)) & (baseline_errors.period_quarter_number > 0)
    ]

    fig = px.scatter(
        baseline_branch_errors,
        x="period_quarter_number",
        y="mean_absolute_error_ra",
        color="sbi_title",
        title="Mean Absolute Errors for Rolling Average",
        text="formatted_mean_absolute_error",  # Add this line to label markers with their Y values
    )

    # Adjust positions of text labels by trace index
    for i, trace in enumerate(fig.data):
        # Set text position based on the trace index
        position = "bottom center" if i % 2 == 0 else "top center"

        trace.update(
            mode="lines+markers+text",
            textposition=position,
        )

        # Ensure text font color matches the marker color
        trace.textfont.color = trace.marker.color

    fig.update_layout(
        plot_bgcolor="rgb(0,0,0,0)",
        paper_bgcolor="rgb(0,0,0,0)",
        yaxis_title="Mean Absolute Error",
        xaxis_title="Quarter Number",
        xaxis_type="category",  # Treat the x-axis as categorical
        legend=dict(
            title="",
            orientation="h",  # Horizontal orientation
            x=0.5,  # Center horizontally
            y=-0.2,  # Position below the plot, adjust the value as needed
            xanchor="center",  # Horizontally center the legend
            yanchor="top",  # Align the legend’s top with the specified `y` position
        ),
        xaxis=dict(
            showgrid=False,
        ),
        yaxis=dict(
            showgrid=False,
        ),
    )

    fig.show()


@app.command()
def main(
    force: bool = False,
    with_total: bool = False,
    viz_type: str = "KW",
    sbi_individual_plots: bool = False,
):
    load_dataframe()

    sbi_list = SBI_LIST if not with_total else SBI_LIST_WITH_TOTAL

    if sbi_individual_plots:
        for sbi in sbi_list:
            viz_predictions(sbi=sbi, viz_type=viz_type)

    viz_multiple_branches(sbi_list=sbi_list, viz_type=viz_type)
    viz_mean_absolute_errors(sbi_list=sbi_list)


if __name__ == "__main__":
    app()
