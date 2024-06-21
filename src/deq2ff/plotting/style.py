import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, pathlib

# andreas-burger/EquilibriumEquiFormer
entity = "andreas-burger"
project = "EquilibriumEquiFormer"

# parent folder of the plot
plotfolder = pathlib.Path(__file__).parent.absolute()
plotfolder = os.path.join(plotfolder, "plots")

myrc = {
    "figure.figsize": (8, 6),  # Adjust the figure size as needed
    "font.size": 20,  # Increase font size
    "lines.linewidth": 3.5,  # Thicker lines
    "lines.markersize": 8,  # Thicker lines
    "legend.fontsize": 15,  # Legend font size
    "legend.frameon": False,  # Display legend frame
    "legend.loc": "upper right",  # Adjust legend position
    "axes.spines.right": False,
    "axes.spines.top": False,  # top and right border
    "text.usetex": True,
}

pkmn_type_colors = [
    "#6890F0",  # Water
    "#F08030",  # Fire
    "#78C850",  # Grass
    "#A8B820",  # Bug
    "#A8A878",  # Normal
    "#A040A0",  # Poison
    "#F8D030",  # Electric
    "#E0C068",  # Ground
    "#EE99AC",  # Fairy
    "#C03028",  # Fighting
    "#F85888",  # Psychic
    "#B8A038",  # Rock
    "#705898",  # Ghost
    "#98D8D8",  # Ice
    "#7038F8",  # Dragon
]

# dark, muterd, deep
PALETTE = "dark"
# sns.color_palette("Set3", 10)

# sns.color_palette("crest", as_cmap=True)
# sns.color_palette("dark:b", as_cmap=True)
# sns.color_palette("dark:#5A9_r", as_cmap=True)
# https://content.codecademy.com/programs/dataviz-python/unit-5/seaborn-design-2/article2_image9.png
# https://colorbrewer2.org/#type=sequential&scheme=Oranges&n=4


def set_seaborn_style(
    style="whitegrid", palette=PALETTE, context="poster", figsize=(8, 6), font_scale=0.8
):
    sns.set_style(style=style)  # whitegrid white
    sns.set_palette(palette)
    myrc["figure.figsize"] = figsize
    sns.set_context(
        context,  # {paper, notebook, talk, poster}
        font_scale=font_scale,
        rc=myrc,
    )


def set_style_after(ax, fs=15, legend=True):
    plt.grid(False)
    plt.grid(which="major", axis="y", linestyle="-", linewidth="1.0", color="lightgray")

    # removes axes spines top and right
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if legend is True:
        # increase legend fontsize
        ax.legend(fontsize=fs)

        # remove legend border
        # ax.legend(frameon=False)
        ax.get_legend().get_frame().set_linewidth(0.0)
        # plt.legend().get_frame().set_linewidth(0.0)
    elif legend is None:
        pass
    else:
        ax.get_legend().remove()


timelabels = {
    "time_test": "Test time for 1000 samples [s]",
    "time_forward_per_batch_test": "Forward time per batch [s]",
    "time_forward_total_test": "Total forward time for 1000 samples [s]",
    "nfe": "NFE",
}
acclabels = {
    "best_test_f_mae": r"Best force MAE [kcal/mol/$\AA$]",
    "test_f_mae": r"Force MAE [kcal/mol/$\AA$]",
    "best_test_e_mae": "Best energy MAE [kcal/mol]",
    "test_e_mae": "Energy MAE [kcal/mol]",
    "nfe": "NFE",
}


def combine_legend(ax, colorstyle_dict, markerstyle):
    # https://stackoverflow.com/questions/68591271/how-can-i-combine-hue-and-style-groups-in-a-seaborn-legend

    # seaborn used to have the legend title as element 0, but it seems to have been removed
    offset = 1

    # create a dictionary mapping the colorstyle to their color
    handles, labels = ax.get_legend_handles_labels()
    index_item_title = labels.index(markerstyle)
    color_dict = {
        label: handle.get_color()
        for handle, label in zip(
            handles[offset:index_item_title], labels[offset:index_item_title]
        )
    }
    print("color_dict:", color_dict)

    # loop through the items, assign color via the colorstyle of the item
    for handle, label in zip(
        handles[index_item_title + offset :], labels[index_item_title + offset :]
    ):
        handle.set_color(color_dict[colorstyle_dict[label]])

    # create a legend only using the items
    ax.legend(
        handles=handles[index_item_title + 1 :],
        labels=labels[index_item_title + 1 :],
        # title='Item',
        # bbox_to_anchor=(1.03, 1.02), fontsize=10
    )


# Kevin Xi


def prep_plot_style(do_legend=True, xrange=None, legend_framealpha=0, legend_loc=None):
    # A function that sets some simple matplotlib style options re-used in each plot
    ax = plt.gca()

    if do_legend:
        plt.legend(framealpha=legend_framealpha, loc=legend_loc)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False,
        left=False,
    )  # labels along the bottom edge are off

    if xrange:
        ax.set_xticks([xrange[0], (xrange[0] + xrange[1]) / 2, xrange[1]])
    # plt.tight_layout()


import matplotlib.style as style

# Define custom style parameters
custom_style = {
    "figure.figsize": (8, 6),  # Adjust the figure size as needed
    "font.size": 20,  # Increase font size
    "lines.linewidth": 3.5,  # Thicker lines
    "lines.markersize": 8,  # Thicker lines
    "legend.fontsize": 12,  # Legend font size
    "legend.frameon": True,  # Display legend frame
    "legend.loc": "upper right",  # Adjust legend position
}

if __name__ == "__main__":
    # Apply the custom style
    style.use(custom_style)
