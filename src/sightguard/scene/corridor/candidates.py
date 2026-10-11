"""Small reproducible widening image bands, independent of evidence selection."""

from dataclasses import asdict

from sightguard.scene.corridor.schema import CandidateConfig, CandidateCorridor
from sightguard.scene.spatial_grid.schema import SpatialGridFrame


class CandidateGenerator:
    def __init__(self, config: CandidateConfig | None = None) -> None:
        self.config = config or CandidateConfig()

    def generate(self, frame: SpatialGridFrame) -> tuple[CandidateCorridor, ...]:
        """Templates always retain geometry; eligibility is a separate operation."""
        config = self.config
        start_row = min(frame.rows - 1, int(frame.rows * config.start_row_fraction))
        near_row = min(frame.rows - 1, int(frame.rows * config.near_row_fraction))
        results = []
        for index, name in enumerate(config.names):
            ids, near_ids, bounds = [], [], []
            for row in range(start_row, frame.rows):
                position = (row - start_row) / max(1, frame.rows - start_row - 1)
                center = (
                    (1 - position) * config.top_centers[index]
                    + position * config.bottom_centers[index]
                ) * (frame.cols - 1)
                width = max(
                    1,
                    round(
                        frame.cols
                        * (
                            (1 - position) * config.top_width_fraction
                            + position * config.bottom_width_fraction
                        )
                    ),
                )
                first = max(0, min(frame.cols - width, round(center - (width - 1) / 2)))
                row_ids = list(range(row * frame.cols + first, row * frame.cols + first + width))
                ids.extend(row_ids)
                if row >= near_row:
                    near_ids.extend(row_ids)
                left, right = frame.cells[row_ids[0]], frame.cells[row_ids[-1]]
                bounds.append(
                    (
                        left.bounds_xyxy[0],
                        left.bounds_xyxy[1],
                        right.bounds_xyxy[2],
                        right.bounds_xyxy[3],
                    )
                )
            polygon = tuple(
                [point for x1, y1, _, y2 in bounds for point in ((x1, y1), (x1, y2))]
                + [point for _, y1, x2, y2 in reversed(bounds) for point in ((x2, y2), (x2, y1))]
            )
            results.append(
                CandidateCorridor(
                    name,
                    frame.key,
                    tuple(ids),
                    tuple(bounds),
                    polygon,
                    tuple(near_ids),
                    asdict(config),
                )
            )
        return tuple(results)
