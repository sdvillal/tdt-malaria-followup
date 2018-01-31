from future.utils import string_types

import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from IPython import get_ipython

# Some notebook extras, to catch your attention

get_ipython().magic('matplotlib notebook')


def info(text):
    # noinspection PyTypeChecker
    display(HTML('<div class="alert alert-block alert-success alert-text-normal">' + text + '</div>'))


def warning(text):
    # noinspection PyTypeChecker
    display(HTML('<div class="alert alert-block alert-warning alert-text-normal">' + text + '</div>'))


def danger(text):
    # noinspection PyTypeChecker
    display(HTML('<div class="alert alert-block alert-danger alert-text-normal">' + text + '</div>'))


# And some help for nicer dataframe display

pd.set_option('display.max_colwidth', -1)


def to_html(df, index=False):
    return df.to_html(index=index, escape=False)


def show_df(df, index=False):
    # noinspection PyTypeChecker
    display(HTML(to_html(df, index=index)))


# Show inline images

def show_image(image, bgr=False):
    if isinstance(image, string_types):
        image = mpimg.imread(image)
    if bgr:
        image = image[:, :, ::-1]
    plt.imshow(image)
    plt.show()
