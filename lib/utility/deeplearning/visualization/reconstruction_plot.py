from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ReconstructionPlot:
    @staticmethod
    def _prepare_image(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)

        if image.ndim == 2:
            return np.clip(image, 0.0, 1.0)

        if image.ndim == 3 and image.shape[-1] == 1:
            return np.clip(image[:, :, 0], 0.0, 1.0)

        return np.clip(image, 0.0, 1.0)

    @staticmethod
    def _normalize_image(img: np.ndarray) -> np.ndarray:
        img = np.asarray(img, dtype=np.float32)

        # Robust normalization
        img_min = np.min(img)
        img_max = np.max(img)

        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

        return img

    @staticmethod
    def _add_grayscale_image(fig, img, row, col):
        fig.add_trace(
            go.Heatmap(
                z=img,
                colorscale="Gray",
                showscale=False,
            ),
            row=row,
            col=col,
        )

    @staticmethod
    def create_grid(
        images: np.ndarray,
        title: str,
        max_images: int = 5,
        columns: int = 5,
    ):
        count = min(max_images, len(images))
        columns = max(1, min(columns, count))
        rows = int(np.ceil(count / columns))

        fig = make_subplots(
            rows=rows,
            cols=columns,
        )

        idx = 0

        for row in range(1, rows + 1):
            for col in range(1, columns + 1):
                if idx >= count:
                    break

                img = ReconstructionPlot._prepare_image(images[idx])
                img = ReconstructionPlot._normalize_image(img)

                ReconstructionPlot._add_grayscale_image(
                    fig,
                    img,
                    row,
                    col,
                )

                idx += 1

        fig.update_layout(
            title=title,
            height=240 * rows,
            width=240 * columns,
            margin=dict(l=10, r=10, t=40, b=10),
        )

        fig.update_xaxes(
            visible=False,
            showgrid=False,
            zeroline=False,
        )

        fig.update_yaxes(
            visible=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
        )

        return fig

    @staticmethod
    def create_noisy_vs_reconstructed(
        noisy_images: np.ndarray,
        reconstructed_images: np.ndarray,
        max_images: int = 10,
    ):
        count = min(
            max_images,
            len(noisy_images),
            len(reconstructed_images),
        )

        if count == 0:
            return go.Figure()

        columns = min(5, count)

        noisy_rows = int(np.ceil(count / columns))
        total_rows = noisy_rows * 2

        fig = make_subplots(
            rows=total_rows,
            cols=columns,
            vertical_spacing=0.08,
            horizontal_spacing=0.03,
        )

        for idx in range(count):

            col = (idx % columns) + 1
            noisy_row = (idx // columns) + 1
            recon_row = noisy_row + noisy_rows

            noisy = ReconstructionPlot._normalize_image(
                ReconstructionPlot._prepare_image(
                    noisy_images[idx]
                )
            )

            recon = ReconstructionPlot._normalize_image(
                ReconstructionPlot._prepare_image(
                    reconstructed_images[idx]
                )
            )

            fig.add_trace(
                go.Heatmap(
                    z=noisy,
                    colorscale="Gray",
                    showscale=False,
                    hoverinfo="skip",
                ),
                row=noisy_row,
                col=col,
            )

            fig.add_trace(
                go.Heatmap(
                    z=recon,
                    colorscale="Gray",
                    showscale=False,
                    hoverinfo="skip",
                ),
                row=recon_row,
                col=col,
            )

        # Remove axes
        fig.update_xaxes(
            visible=False,
            showgrid=False,
            zeroline=False,
        )

        fig.update_yaxes(
            visible=False,
            showgrid=False,
            zeroline=False,
        )

        # Main title
        fig.update_layout(
            title={
                "text": "Noisy vs Reconstructed Dental X-rays",
                "x": 0.5,
                "xanchor": "center",
            },

            autosize=True,

            # Important for card rendering
            width=None,

            # Dynamic height
            height=max(
                800,
                total_rows * 260,
            ),

            margin=dict(
                l=20,
                r=20,
                t=100,
                b=20,
            ),

            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        # Section headers
        fig.add_annotation(
            text="<b>Noisy Images</b>",
            x=0,
            y=1.02,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font=dict(size=18),
        )

        fig.add_annotation(
            text="<b>Reconstructed Images</b>",
            x=0,
            y=0.48,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font=dict(size=18),
        )

        return fig
