import cv2
import numpy as np
import plotly.express as px
from plotly import graph_objects as go


def draw_masks(img, masks, opacity=0.3, border_opacity=0.7, border_size=3):
    # here masks is a dict having category names as keys
    # associated to a list of binary masks
    masked_image = img.copy()
    masked_gradient = img.copy()
    kernel = np.ones((border_size, border_size), np.uint8)
    colors = {}
    if not isinstance(masks, list):
        masks = [masks]

    for i, cat in enumerate(masks):
        colors[i] = np.random.randint(0, 255, size=(3,)).tolist()

    for i, mask in enumerate(masks):
        masked_image = np.where(
            np.repeat(mask[:, :, np.newaxis], 3, axis=2), np.asarray(colors[i], dtype="uint8"), masked_image
        )
        gradient = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)

        masked_gradient = np.where(
            np.repeat(gradient[:, :, np.newaxis], 3, axis=2), np.asarray(colors[i], dtype="uint8"), masked_gradient
        )

    masked_image = masked_image.astype(np.uint8)
    img_with_mask = cv2.addWeighted(img, 1 - opacity, masked_image, opacity, 0)
    img_with_mask = cv2.addWeighted(img_with_mask, 1 - border_opacity, masked_gradient, border_opacity, 0)
    return img_with_mask


def imshow(
    image,
    mask=None,
    title=None,
    width=None,
    height=None,
    keypoints=None,
    labels={},
    opacity=0.5,
    border_opacity=0.7,
    border_size=3,
    show=True,
):
    """Plot an image using Plotly.

    Args:
        image (numpy.ndarray): Image to plot.
        title (str, optional): Plot title. Defaults to None.
        width (int, optional): Plot width. Defaults to None.
        height (int, optional): Plot height. Defaults to None.

    Returns:
        plotly.graph_objs.Figure: Plotly figure.
    """
    np.random.seed(1)
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)

    if mask is not None:
        image = draw_masks(image, mask, opacity=opacity, border_opacity=border_opacity, border_size=border_size)

    fig = px.imshow(
        image,
        title=title,
        x=np.arange(image.shape[1]),
        y=np.arange(image.shape[0]),
    )

    if keypoints is not None:
        fig.add_trace(
            go.Scatter(
                x=keypoints[:, 1], y=keypoints[:, 0], mode="markers", marker=dict(size=4, opacity=0.8), showlegend=False
            )
        )

    fig.update_xaxes(visible=False, range=[0, image.shape[1]])
    fig.update_yaxes(visible=False, range=[0, image.shape[0]])

    fig.update_layout(width=width, height=height, title=title, margin=dict(l=0, r=0, b=0, t=0))

    config = {"scrollZoom": True, "displaylogo": False}
    fig.update_layout(dragmode="pan")
    if show:
        fig.show(config=config)
    return fig


def imshows(images, masks):
    figs = []
    for i, (image, mask) in enumerate(zip(images, masks)):
        fig = imshow(image, mask, title=f"Image {i}")
        figs.append(fig)

    layout = go.Layout()
    fig = go.Figure(data=figs, layout=layout)
    fig.show()
    return figs
