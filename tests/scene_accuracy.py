"""Helpers for evaluating scene-cut detections against annotated frame numbers."""


def score_cut_frames(reference_frames, detected_frames, tolerance_frames=1):
    """Return one-to-one precision/recall/F1 metrics for frame-level cut points."""
    if tolerance_frames < 0:
        raise ValueError("tolerance_frames must be non-negative")

    references = sorted(reference_frames)
    detections = sorted(detected_frames)
    reference_index = 0
    detection_index = 0
    true_positives = 0

    while reference_index < len(references) and detection_index < len(detections):
        reference = references[reference_index]
        detection = detections[detection_index]
        if abs(reference - detection) <= tolerance_frames:
            true_positives += 1
            reference_index += 1
            detection_index += 1
        elif detection < reference - tolerance_frames:
            detection_index += 1
        else:
            reference_index += 1

    false_positives = len(detections) - true_positives
    false_negatives = len(references) - true_positives
    precision = (
        true_positives / len(detections)
        if detections
        else float(not references)
    )
    recall = (
        true_positives / len(references)
        if references
        else float(not detections)
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
