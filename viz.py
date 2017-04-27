import sys
USE_GPU = "--cpu" not in sys.argv

import matplotlib.pyplot as plt

OUTPUT_FOLDER = "output/"

# Subplots helper: hide axes, minimize space between, maximize window
def cleanSubplots(r, c, pad=0.05, axes=False):
    f, ax = plt.subplots(r, c)
    if not axes:
        if r == 1 and c == 1:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        elif r == 1 or c == 1:
            for a in ax:
                a.get_xaxis().set_visible(False)
                a.get_yaxis().set_visible(False)
        else:
            for aRow in ax:
                for a in aRow:
                    a.get_xaxis().set_visible(False)
                    a.get_yaxis().set_visible(False)

    f.subplots_adjust(left=pad, right=1.0-pad, top=1.0-pad, bottom=pad, hspace=pad)
    try:
        plt.get_current_fig_manager().window.showMaximized()
    except AttributeError:
        pass # Can't maximize, sorry :(
    return ax

# Visualization helper: Show results, or write to file if running on AWS:
def saveOrShow(path):
    plt.savefig(OUTPUT_FOLDER + path)
    if not USE_GPU:
        plt.show()
