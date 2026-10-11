"""Independent image evidence evaluation with explicit eligibility and abstention."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from sightguard.scene.corridor.artifacts import validate_grid
from sightguard.scene.corridor.candidates import CandidateGenerator
from sightguard.scene.corridor.schema import (
    CandidateAssessment,
    CandidateCorridor,
    CorridorAssessment,
    CorridorConfig,
    CorridorEvidence,
    ObjectCorridorRelation,
)
from sightguard.scene.spatial_grid.schema import SpatialGridCell, SpatialGridFrame

ABLATIONS = {
    "A": "geometry only",
    "B": "semantics and detections",
    "C": "semantics, detections and relative depth",
    "D": "all evidence with uncertainty gates",
}


def area(cell: SpatialGridCell) -> int:
    x1, y1, x2, y2 = cell.bounds_xyxy
    return (x2 - x1) * (y2 - y1)


def bbox_proxy(cell: SpatialGridCell) -> float:
    # Max box coverage is a proxy, not union area or instance occupancy probability.
    return max((x.cell_fraction for x in cell.detection_coverage if x.candidate), default=0.0)


class CorridorEvaluator:
    """Stateless, exact-frame observer; never changes risk or decision state."""

    def __init__(self, config: CorridorConfig | None = None) -> None:
        self.config = config or CorridorConfig()
        self.generator = CandidateGenerator(self.config.candidates)

    def aggregate(
        self, frame: SpatialGridFrame, candidate: CandidateCorridor, ablation: str
    ) -> CorridorEvidence:
        if ablation not in ABLATIONS or candidate.key != frame.key:
            raise ValueError("invalid ablation or candidate frame identity")
        cells = [frame.cells[x] for x in candidate.cell_ids]
        start = cells[0].row
        denominator = max(1, frame.rows - 1 - start)
        weights = {
            x.cell_id: area(x)
            * (1 + self.config.candidates.row_weight_gain * (x.row - start) / denominator)
            for x in cells
        }

        def mean(values: dict[int, float], subset: list[SpatialGridCell] = cells) -> float:
            total = sum(weights[x.cell_id] for x in subset)
            return sum(weights[x.cell_id] * values.get(x.cell_id, 0) for x in subset) / total

        semantics = ablation != "A" and frame.source_status.get("semantic") == "valid"
        detections = ablation != "A" and frame.source_status.get("detection") == "valid"
        depth = (
            ablation in ("C", "D")
            and frame.source_status.get("depth") == "valid"
            and len(frame.frame_depth_quantiles) >= 2
            and max(frame.frame_depth_quantiles) > min(frame.frame_depth_quantiles)
        )
        ground = {x.cell_id: (x.ground.support_fraction or 0) if semantics else 0 for x in cells}
        classes = {
            name: mean(
                {
                    x.cell_id: x.ground.class_fractions.get(name, 0) if semantics else 0
                    for x in cells
                }
            )
            for name in ("floor", "road", "sidewalk")
        }
        factors = {
            "floor": self.config.evaluation.floor_support_factor,
            "road": self.config.evaluation.road_support_factor,
            "sidewalk": self.config.evaluation.sidewalk_support_factor,
        }
        adjusted = {}
        for cell in cells:
            fractions = cell.ground.class_fractions
            total = sum(fractions.get(name, 0) for name in factors)
            factor = sum(fractions.get(name, 0) * factor for name, factor in factors.items())
            adjusted[cell.cell_id] = ground[cell.cell_id] * factor / total if total else 0
        bottom = [x for x in cells if x.row == frame.rows - 1]
        near = [x for x in cells if x.cell_id in candidate.near_cell_ids]
        reachable: dict[int, set[int]] = {}
        connected_rows = 0
        interrupted = []
        for row in range(frame.rows - 1, start - 1, -1):
            current = {}
            for cell in (x for x in cells if x.row == row):
                components = set(cell.ground.bottom_connected_component_ids)
                if not semantics or cell.ground.status != "supported" or not components:
                    continue
                if row == frame.rows - 1:
                    current[cell.col] = components
                else:
                    matched = set().union(
                        *(
                            ids & components
                            for col, ids in reachable.items()
                            if abs(col - cell.col) <= 1
                        )
                    )
                    if matched:
                        current[cell.col] = matched
            reachable = current
            if current:
                connected_rows += 1
            else:
                interrupted.append(row)
        coverage = {x.cell_id: bbox_proxy(x) if detections else 0 for x in cells}
        valid_depth = {
            x.cell_id: x.depth.frame_percentile
            for x in cells
            if depth and x.depth.valid_count and x.depth.frame_percentile is not None
        }
        depth_coverage = mean(
            {x.cell_id: x.depth.valid_count / area(x) for x in cells if x.cell_id in valid_depth}
        )
        valid_near = [x for x in near if x.cell_id in valid_depth]
        near_depth = (
            mean(
                {x.cell_id: coverage[x.cell_id] * valid_depth[x.cell_id] for x in valid_near},
                valid_near,
            )
            if valid_near
            else None
        )
        variations = {
            x.cell_id: min(1, x.depth.spread_fraction)
            for x in cells
            if x.cell_id in valid_depth and x.depth.spread_fraction is not None
        }
        variation = (
            mean(variations, [x for x in cells if x.cell_id in variations]) if variations else None
        )
        semantic_unknown = {
            x.cell_id: float(not semantics or x.ground.status == "unknown") for x in cells
        }
        # Excluded sources do not contribute unknown exposure to a smaller ablation.
        unknown = {
            x.cell_id: max(
                semantic_unknown[x.cell_id],
                float(not detections),
                1 - x.depth.valid_count / area(x) if depth else 0,
                float(ablation in ("C", "D") and not depth),
            )
            for x in cells
        }
        conflicts = {
            x.cell_id: float(
                any(
                    reason == "ground_and_bbox_candidate_overlap"
                    or (depth and reason == "relative_depth_variation_not_object_boundary")
                    for reason in x.conflict_reasons
                )
            )
            if semantics and detections
            else 0
            for x in cells
        }
        reasons = set()
        # SpatialGridFrame v1 has no measured pose field. Absence of a limitation
        # string is not affirmative evidence of an observed camera orientation.
        reasons.add("camera_pose_unobserved")
        if depth_coverage < 1 and ablation in ("C", "D"):
            reasons.add("partial_or_missing_relative_depth")
        if detections and any(x.detection_coverage for x in cells):
            reasons.add("bbox_contains_background_not_instance_mask")
        if any(x.depth.quality != "valid" for x in cells) and depth:
            reasons.add("depth_quality_limited")
        if mean(semantic_unknown):
            reasons.add("ground_evidence_unknown")
        statuses = dict(frame.source_status)
        if ablation == "A":
            statuses = {name: "excluded" for name in statuses}
        elif ablation == "B":
            statuses["depth"] = "excluded"
        return CorridorEvidence(
            mean(ground),
            classes,
            mean(adjusted),
            mean(ground, bottom),
            connected_rows / (frame.rows - start),
            tuple(interrupted),
            mean(unknown),
            mean(semantic_unknown),
            mean(coverage),
            mean(coverage, near),
            near_depth,
            variation,
            depth_coverage,
            mean(conflicts),
            mean({x.cell_id: (x.ground.mean_confidence or 0) if semantics else 0 for x in cells}),
            statuses,
            "unknown",
            tuple(sorted(reasons)),
        )

    def evaluate_candidate(
        self,
        frame: SpatialGridFrame,
        candidate: CandidateCorridor,
        ablation: str,
        evidence: CorridorEvidence,
        *,
        include_relations: bool = True,
    ) -> CandidateAssessment:
        cfg = self.config.evaluation
        reasons = []
        if ablation == "A":
            reasons.append("geometry_has_no_passability_evidence")
        else:
            if evidence.source_status.get("semantic") != "valid":
                reasons.append("semantic_source_unavailable")
            if evidence.source_status.get("detection") != "valid":
                reasons.append("detection_source_unavailable")
            for actual, minimum, reason in (
                (evidence.ground_support, cfg.min_ground_support, "insufficient_ground_support"),
                (
                    evidence.bottom_ground_support,
                    cfg.min_bottom_ground,
                    "insufficient_bottom_ground",
                ),
                (evidence.ground_continuity, cfg.min_continuity, "interrupted_bottom_ground"),
            ):
                if actual < minimum:
                    reasons.append(reason)
        if ablation == "D":
            for actual, maximum, reason in (
                (evidence.unknown_exposure, cfg.max_unknown, "excessive_unknown_evidence"),
                (evidence.conflict_coverage, cfg.max_conflict, "conflicting_evidence"),
                (
                    evidence.near_bbox_proxy_coverage,
                    cfg.max_near_bbox_coverage,
                    "near_bbox_coverage_gate",
                ),
            ):
                if actual > maximum:
                    reasons.append(reason)
            if evidence.semantic_confidence < cfg.min_scene_confidence:
                reasons.append("semantic_confidence_gate")
            if evidence.depth_coverage < cfg.min_depth_coverage:
                reasons.append("relative_depth_coverage_gate")
            if cfg.require_observed_orientation and evidence.camera_orientation != "observed":
                reasons.append("camera_orientation_unobserved")
        components: dict[str, float | None] = {
            "ground": 1 - evidence.class_adjusted_ground_support,
            "continuity": 1 - evidence.ground_continuity,
            "obstacle": evidence.bbox_proxy_coverage,
            "depth": (
                cfg.depth_near_fraction * evidence.near_relative_depth_proxy
                + (1 - cfg.depth_near_fraction) * evidence.relative_depth_variation
                if ablation in ("C", "D")
                and evidence.near_relative_depth_proxy is not None
                and evidence.relative_depth_variation is not None
                else None
            ),
            "unknown": evidence.unknown_exposure,
            "conflict": evidence.conflict_coverage,
        }
        if ablation == "A":
            components = dict.fromkeys(components)
        contributions = {
            name: value * getattr(cfg, f"{name}_weight") if value is not None else None
            for name, value in components.items()
        }
        total = sum(x for x in contributions.values() if x is not None) if not reasons else None
        relations = self.object_relations(frame, candidate, ablation) if include_relations else ()
        return CandidateAssessment(
            candidate,
            evidence,
            "INELIGIBLE" if reasons else "ELIGIBLE",
            tuple(reasons),
            components,
            contributions,
            total,
            tuple(name for name, value in components.items() if value is None),
            relations,
        )

    def object_relations(
        self, frame: SpatialGridFrame, candidate: CandidateCorridor, ablation: str
    ) -> tuple[ObjectCorridorRelation, ...]:
        if ablation == "A" or frame.source_status.get("detection") != "valid":
            return ()
        result = []
        selected = set(candidate.cell_ids)
        near = set(candidate.near_cell_ids)
        candidate_area = sum(area(frame.cells[x]) for x in selected)
        near_area = sum(area(frame.cells[x]) for x in near)
        for obj in frame.object_relations:
            overlaps = [x for x in obj.cell_coverage if x["cell_id"] in selected]
            if not overlaps:
                continue
            cells = tuple(int(x["cell_id"]) for x in overlaps)
            covered_area = sum(
                x["cell_fraction"] * area(frame.cells[int(x["cell_id"])]) for x in overlaps
            )
            near_covered = sum(
                x["cell_fraction"] * area(frame.cells[int(x["cell_id"])])
                for x in overlaps
                if x["cell_id"] in near
            )
            depth_enabled = (
                ablation in ("C", "D")
                and frame.source_status.get("depth") == "valid"
                and len(frame.frame_depth_quantiles) >= 2
                and max(frame.frame_depth_quantiles) > min(frame.frame_depth_quantiles)
            )
            neighbors = set()
            object_cells = {x["cell_id"] for x in obj.cell_coverage}
            for index in cells:
                row, col = divmod(index, frame.cols)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = row + dr, col + dc
                    if 0 <= rr < frame.rows and 0 <= cc < frame.cols:
                        other = rr * frame.cols + cc
                        if other not in object_cells and not frame.cells[other].detection_coverage:
                            neighbors.add(other)
            background = [
                frame.cells[x].depth.frame_percentile
                for x in sorted(neighbors)
                if depth_enabled and frame.cells[x].depth.frame_percentile is not None
            ]
            background_mean = sum(background) / len(background) if background else None
            percentile = obj.depth.frame_percentile if depth_enabled else None
            coverage = next(
                x
                for x in frame.cells[cells[0]].detection_coverage
                if x.detection_index == obj.detection_index
            )
            reasons = set(obj.reasons) | {
                "bbox_contains_background_not_instance_mask",
                "same_frame_only",
            }
            if background_mean is None:
                reasons.add("background_depth_unavailable")
            result.append(
                ObjectCorridorRelation(
                    frame.key,
                    candidate.candidate_id,
                    obj.detection_index,
                    obj.track_id,
                    obj.class_id,
                    coverage.class_name,
                    obj.confidence,
                    cells,
                    min(1, sum(x["bbox_fraction"] for x in overlaps)),
                    covered_area / candidate_area,
                    near_covered / near_area,
                    obj.ground_fraction if ablation != "A" else None,
                    percentile,
                    background_mean,
                    percentile - background_mean
                    if percentile is not None and background_mean is not None
                    else None,
                    "limited",
                    tuple(sorted(reasons)),
                    obj.detection_reference,
                )
            )
        return tuple(result)

    def assess(
        self,
        frame: SpatialGridFrame,
        ablation: str = "D",
        *,
        provenance: dict[str, Any] | None = None,
        validate: bool = True,
    ) -> CorridorAssessment:
        if ablation not in ABLATIONS:
            raise ValueError("unknown ablation")
        if validate:
            validate_grid(frame)
        candidates = self.generator.generate(frame)
        evaluated = tuple(
            self.evaluate_candidate(
                frame, candidate, ablation, self.aggregate(frame, candidate, ablation)
            )
            for candidate in candidates
        )
        return self.assemble(frame, ablation, evaluated, provenance)

    def assemble(
        self,
        frame: SpatialGridFrame,
        ablation: str,
        evaluated: tuple[CandidateAssessment, ...],
        provenance: dict[str, Any] | None = None,
    ) -> CorridorAssessment:
        """Rank already evaluated bands without assigning a navigation action."""
        eligible = sorted(
            (x for x in evaluated if x.total_cost is not None),
            key=lambda x: (x.total_cost, x.candidate.candidate_id),
        )
        tied = tuple(
            x.candidate.candidate_id
            for x in eligible
            if x.total_cost - eligible[0].total_cost <= self.config.evaluation.tie_margin
        )
        status = "ABSTAIN" if not eligible else "TIED" if len(tied) > 1 else "BEST_SUPPORTED"
        return CorridorAssessment(
            frame.key,
            ablation,
            evaluated,
            status,
            eligible[0].candidate.candidate_id if status == "BEST_SUPPORTED" else None,
            tied if len(tied) > 1 else (),
            tuple(x.candidate.candidate_id for x in eligible),
            tuple(sorted({reason for x in evaluated for reason in x.eligibility_reasons}))
            if not eligible
            else ("cost_tie_within_margin",)
            if len(tied) > 1
            else (),
            {
                "config": asdict(self.config),
                "sources": {
                    name: asdict(source) if source is not None else None
                    for name, source in frame.sources.items()
                },
                **(provenance or {}),
            },
        )
