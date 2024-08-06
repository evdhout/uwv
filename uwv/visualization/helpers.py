import datetime
from pathlib import Path

from matplotlib.figure import Figure

from uwv.config import OUTPUT_DIR

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def get_graph_path(
    title: str,
    extension: str = "png",
    with_timestamp: bool = True,
    base_dir: Path = OUTPUT_DIR,
    sub_dir: str | None = None,
) -> Path:
    """Construct the graph filename

    Args:
        title (str): The name of the graph
        extension (str, optional): The filetype of the graph. Defaults to "png".
        with_timestamp (bool, optional): Add timestamp in filename. Defaults to True.
        base_dir (Path, optional): the root of the path to save the file. Defaults to OUTPUT_DIR
        sub_dir (str | None, optional): the subdirectory to put the graph in

    Returns:
        Path: Path to the graph file
    """
    if sub_dir is not None:
        base_dir = base_dir / sub_dir

    if with_timestamp:
        return base_dir / f"{title}_{TIMESTAMP}.{extension}"
    else:
        return base_dir / f"{title}.{extension}"


def save_fig(figure: Figure, title: str, with_timestamp: bool = True):
    figure.savefig(get_graph_path(title=title, extension="png", with_timestamp=with_timestamp))
    figure.savefig(get_graph_path(title=title, extension="svg", with_timestamp=with_timestamp))
