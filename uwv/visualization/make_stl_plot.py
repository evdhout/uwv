import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from statsmodels.tsa.seasonal import STL

from uwv.config import CBS80072NED, CBS_OPENDATA_PROCESSED_DATA_DIR
from uwv.visualization.helpers import save_fig


def get_series(start_year: int, end_year: int, dataset: str = "T001081") -> pd.Series:
    """Get the series for the STL plot from the complete dataframe

    Args:
        start_year (int): start year
        end_year (int): end year
        dataset (str, optional): Which dataset to use.
            Defaults to "T001081" wich is the aggregated dataset.

    Returns:
        pd.Series: _description_
    """
    slp: pd.DataFrame = pd.read_parquet(CBS_OPENDATA_PROCESSED_DATA_DIR / f"{CBS80072NED}.parquet")

    slp_frame = slp[
        (slp.period_year >= start_year)
        & (slp.period_year <= end_year)
        & (slp.period_type == "KW")
        & (slp.sbi == dataset)
    ]

    slp_series = slp_frame.sick_leave_percentage
    slp_series.index = slp_frame.period
    logger.trace(slp_series)

    return slp_series


def make_stl_plot(start_year: int = 1996, end_year: int = 2022, dataset: str = "T001081"):
    """Generate season trend decomposition plot

    Args:
        start_year (int, optional): Year to start with. Defaults to 2012.
        end_year (int, optional): Year to end with. Defaults to 2022.
    """
    slp_series = get_series(start_year=start_year, end_year=end_year, dataset=dataset)

    plt.rc("font", size=6)

    stl = STL(slp_series, period=4)
    res = stl.fit()
    fig: Figure = res.plot()
    slp, trend, season, residual = fig.get_axes()

    slp.set_title(
        f"Season-Trend decomposition using LOESS of "
        f"quartely sick leave percentages ({start_year} - {end_year})",
        fontsize=8,
        fontweight="bold",
    )

    # Format the x-axis to show years only on major ticks
    # minor ticks are quaters
    tick_font_size = 5 if end_year - start_year > 20 else 6
    residual.axes.xaxis.set_major_locator(MultipleLocator(4))
    residual.axes.xaxis.set_major_formatter(formatter=lambda _, pos: f"{start_year + pos - 1}")
    residual.axes.xaxis.set_minor_locator(MultipleLocator(1))
    for tick in residual.xaxis.get_ticklabels():
        tick.set_horizontalalignment("left")
        tick.set_rotation("horizontal")
        tick.set_fontsize(tick_font_size)
        tick.set_text(tick.get_text()[:4])

    # The title of the residual subplot is "Resid", complete it
    residual.axes.set_ylabel("Residual")

    # Make the titles of the subplots horizontal at the top
    for subplot in [trend, season, residual]:
        subplot.yaxis.set_label_coords(-0.08, 1)
        label = subplot.yaxis.get_label()
        subplot.axes.set_ylabel(
            label.get_text(),
            rotation="horizontal",
            horizontalalignment="left",
            x=-0.08,
            y=1,
        )

    save_fig(fig, f"slp_stl_{dataset}_{start_year}_{end_year}", with_timestamp=False)


if __name__ == "__main__":
    make_stl_plot()
