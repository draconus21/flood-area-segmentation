import os
import logging
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from functools import partial
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from floodsegment.utils import misc

from typing import Tuple, List

logger = logging.getLogger(__name__)


def _saveFig(
    fig,
    figName: str,
    figDir: str = None,
    extra: str = None,
    descStr: str = None,
    saveSize: list = [18.5, 10.5],
    **kwargs,
):
    if extra is None:
        extra = ""
    else:
        extra = f"_{extra}"
    if descStr is None:
        descStr = ""
    else:
        descStr = f"{descStr}"
    fig.set_size_inches(saveSize)

    if figDir is None:
        figDir = misc.getParent(figName)
    misc.mkdirs(figDir, deleteIfExists=False)
    if not figName.endswith(".png"):
        figName = figName + ".png"
    figName = figName.replace(".png", f"{extra}{descStr}.png")
    saveName = os.path.join(figDir, figName)
    fig.tight_layout()
    fig.savefig(saveName, bbox_inches="tight", **kwargs)
    logger.debug(f"saved fig as {saveName}")


def saveFigs(figDict: dict, figDir: str, extra: str = None, descStr: str = None, **kwargs):
    logger.info(f"saving figures in {figDir}")
    for figName, fig in figDict.items():
        _saveFig(fig=fig, figName=figName, figDir=figDir, extra=extra, descStr=descStr, **kwargs)


def _setAxProp(ax, setter, propertyName, default=None, **kwargs):
    pval = kwargs.pop(propertyName, default)
    if pval is not None:
        setter(pval)
    return kwargs


def plot(ax, *args, scalex=True, scaley=True, data=None, **kwargs):
    kwargs = _setAxProp(ax, ax.set_title, "title", **kwargs)
    kwargs = _setAxProp(ax, ax.set_xlabel, "xlabel", **kwargs)
    kwargs = _setAxProp(ax, ax.set_ylabel, "ylabel", **kwargs)
    kwargs = _setAxProp(ax, ax.set_xlim, "xlim", **kwargs)
    kwargs = _setAxProp(ax, ax.set_ylim, "ylim", **kwargs)

    ax.plot(*args, scalex=scalex, scaley=scaley, data=data, **kwargs)


def fill_between(ax, x, y1, y2=0, **kwargs):
    ax.fill_between(x=x, y1=y1, y2=y2, **kwargs)


def axvspan(ax, ymin, ymax, cmap_y, cmap, vmin=None, vmax=None, **kwargs):
    if cmap == "seismic":
        if vmin is None and vmax is None:
            r = np.nanmax(np.abs(cmap_y))
        elif vmax is not None:
            r = np.abs(vmax)
        else:
            r = np.abs(vmin)
        vmin, vmax = -r, r
    else:
        vmin = np.nanmin(cmap_y) if vmin is None else vmin
        vmax = np.nanmax(cmap_y) if vmax is None else vmax

    cmap = cm.ScalarMappable(None, cmap)
    cmap.set_clim(vmin, vmax)
    for i in range(len(ymin)):
        ax.axvspan(ymin[i], ymax[i], color=cmap.to_rgba(cmap_y[i]), **kwargs)


def grid(ax, *args, **kwargs):
    try:
        ax.grid(*args, **kwargs)
    except:
        for a in ax.ravel():
            a.grid(*args, **kwargs)


def legend(ax):
    try:
        ax.legend()
    except:
        for a in ax.ravel():
            a.legend()


def get_vmin_vmax(
    mat: np.ndarray | np.ma.MaskedArray, remove_outliers: bool = True, low_p=2, high_p=98
) -> Tuple[float, float]:
    if isinstance(mat, np.ma.MaskedArray):
        if mat.mask.all():  # all values are masked out
            return 0, 0
        else:
            _data = mat.data[mat.mask]
    else:
        _data = mat

    if remove_outliers:
        return float(np.nanpercentile(_data, low_p)), float(np.nanpercentile(_data, high_p))
    else:
        return np.nanmin(_data), np.nanmax(_data)


def matshow(mat: np.ndarray | np.ma.MaskedArray, ax=None, title: str = "", cbar: bool = True, **kwargs):
    if mat is None:
        logger.debug(f"no mat provided")
        return
    if ax is None:
        suptitle = kwargs.pop("suptitle", f"fig for {title}")
        _, ax = subplots(1, 1, title=suptitle)
        ax = ax.ravel()[0]
    if title:
        ax.set_title(title)

    show_axis = kwargs.pop("show_axis", True)

    cmap = kwargs.get("cmap", "gray")
    if isinstance(cmap, str):
        cmap_obj = cm.get_cmap(cmap)

    # set nans to black for seismic cmap
    if cmap_obj.name == "seismic":
        nancolor = kwargs.pop("nan_color", "black")
        cmap_obj = cm.get_cmap(cmap_obj.name).copy()
        cmap_obj.set_bad(color=nancolor)
        kwargs["cmap"] = cmap_obj

    if "vmin" not in kwargs or "vmax" not in kwargs:
        _vmin, _vmax = get_vmin_vmax(
            mat,
            remove_outliers=kwargs.pop("remove_outliers", True),
            low_p=kwargs.pop("low_p", 2),
            high_p=kwargs.pop("high_p", 98),
        )

        if cmap_obj.name == "seismic":
            r = np.max(np.abs(_vmin), np.abs(_vmax))
            _vmin, _vmax = -r, r
        vmin = kwargs.get("vmin", _vmin)
        vmax = kwargs.get("vmax", _vmax)

    else:
        vmin, vmax = kwargs["vmin"], kwargs["vmax"]
    kwargs["vmin"] = vmin
    kwargs["vmax"] = vmax

    # if nothing is plotted, create a new matshow
    if len(ax.images) == 0:
        ax.matshow(mat, **kwargs)
    else:  # otherwise update existing plot
        ax.images[0].set_data(mat)

    if cbar:
        t = ax.images[0]
        # if cbar is not displayed, show it
        if ax.images[0].colorbar is None:
            ax_div = make_axes_locatable(ax)
            # add an axis to the right of main axis
            cax = ax_div.append_axes("right", size="7%", pad="2%")
            ax.get_figure().colorbar(t, cax=cax)
        else:  # otherwise update it
            t.set_clim([kwargs["vmin"], kwargs["vmax"]])

    if not show_axis:
        ax.axis("off")


def show(*args, **kwargs):
    plt.show(*args, **kwargs)


def pause(interval):
    plt.pause(interval)


def close(*args, **kwargs):
    plt.close(*args, **kwargs)


def subplots_n(n: int, aspect_ratio: float = 1, title: str = "", **kwargs):
    """
    create n subplots while trying to respect the aspect ratio
    aspect ratio := nr/nc
    """
    nc = np.sqrt(float(n / aspect_ratio))
    nr = nc * aspect_ratio
    nc = int(nc)
    nr = int(nr)

    rows_and_cols = lambda a: None if kwargs.get(a, None) is None else not kwargs.pop(a, False)
    odd_rows = rows_and_cols("even_rows")
    odd_cols = rows_and_cols("even_cols")
    if nr * nc < n:
        if odd_rows is None and odd_cols is None:
            if nr * nc < n:  # try adding an extra col
                nc = nc + 1
            if nr * nc < n:  # try adding an extra row
                nr = nr + 1
            assert nr * nc >= n, f"nr [{nr}] and nc [{nc}] that was found do not match n [{n}]"
        elif odd_cols is None:
            odd_cols = False
            nc = nc + 1
        elif odd_rows is None:
            odd_rows = False
            nr = nr + 1
        elif (nr < nc or nc % 2 == odd_cols) and nr % 2 != odd_rows:
            nr = nr + 1
        elif (nc < nr or nr % 2 == odd_rows) and nc % 2 != odd_cols:
            nc = nc + 1
        elif nr < nc:
            nr = nr + 1
        else:
            nc = nc + 1

    if odd_rows is not None:
        nr = nr + (nr % 2 + odd_rows) % 2
    if odd_cols is not None:
        nc = nc + (nc % 2 + odd_cols) % 2

    if nr * nc < n:
        raise ValueError(f"nr [{nr}] and nc [{nc}] that was found do not match n [{n}]")

    fig, ax = subplots(nr, nc, title, **kwargs)
    return fig, ax, nr, nc


def subplots(nrows: int, ncols: int, title: str = "", **kwargs):
    kwargs.setdefault("sharex", True)
    kwargs.setdefault("sharey", True)
    kwargs.setdefault("squeeze", False)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)
    if title:
        fig.suptitle(title)
    return fig, ax


def quickmatshow(
    mats: List[np.ndarray | np.ma.MaskedArray],
    aspect_ratio: float = 1,
    title: str = "",
    subtitles: List[str] = list(),
    **kwargs,
):
    """
    matshow all mats in list
    """
    # remove cmap to avoid conflict with subplots
    cmap = kwargs.pop("cmap", "gray")

    n = len(mats)
    fig, axList, nr, nc = subplots_n(n, aspect_ratio=aspect_ratio, title=title)
    # to avoid conflict with axis title
    kwargs.pop("title", "")

    # add cmap back for matshow
    kwargs["cmap"] = cmap
    for i, mat in enumerate(mats):
        ax = axList[i // nc, i % nc]
        subtitle = subtitles[i] if i < len(subtitles) else ""
        matshow(mat, ax, title=subtitle, **kwargs)


def addHSlider(fig, triggerFunc, sliderCount, **kwargs):
    # make room for sliders
    w, h = 0.65, 0.03
    tab = 0.0
    sliderSpace = h + tab
    fig.subplots_adjust(bottom=0.25 + tab + sliderSpace)
    axHSlider = fig.add_axes([(1 - w) / 2, 0.1 + sliderCount * sliderSpace, w, h])

    def addDefault(key, value):
        if key not in kwargs:
            kwargs[key] = value

    addDefault("valmin", 0)
    addDefault("valmax", 1)
    addDefault("valinit", 0.5)
    addDefault("label", "hslider")

    kwargs["orientation"] = "horizontal"
    hSlider = Slider(ax=axHSlider, **kwargs)
    hSlider.on_changed(partial(triggerFunc, hSlider))
    return hSlider, sliderCount + 1
