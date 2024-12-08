import pandas as pd
import altair as alt

from uwv.config import CBS80072NED, CBS_OPENDATA_PROCESSED_DATA_DIR, OUTPUT_DIR, RAW_DATA_DIR
from uwv.translations.sbi_translate import sbi_translate
from uwv.visualization.get_slp_data import get_slp_data


def boxplot_slp_quarterly():
    slp_quarter = get_slp_data(period="KW", max_year=2022)

    slp_ordered = (
        slp_quarter.groupby("sbi_title",observed=True)["sick_leave_percentage"]
        .median()
        .sort_values(ascending=True)
        .index
        .tolist()
    )

    print("Ordered Index:", slp_ordered)
    print(slp_quarter[['sbi_title', 'sick_leave_percentage']].head())

    # Create a multi-selection for highlighting
    selection = alt.selection_point(
        fields=['sbi_title'],
        bind='legend'
    )

    alt.Chart(data=slp_quarter).mark_boxplot().encode(
        x=alt.X('sick_leave_percentage:Q',
                axis=alt.Axis(title='Sick leave percentage')
                ),
        y=alt.Y('sbi_title:N',
                sort=slp_ordered,
                axis=alt.Axis(title='Branch')
                ),
    ).properties(
        title='Boxplot with quarterly sick leave percentages (1996-2022) sorted on median values'
    ).save(
        str(OUTPUT_DIR / "boxplot_slp_quarterly.png")
    )





if __name__ == "__main__":
    boxplot_slp_quarterly()
