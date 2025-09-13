import os
from pathlib import Path
from typing import List, Tuple

# Labels to exclude
PROHIBITED_LABELS = ["p0_20", "p20_50", "p99", "p100"]

N: int = 1
def _select_best_frames(frame_infos, N=N):
    """
    frame_infos: list of dicts with keys: frame_id, img_file, label_file, n_valid_labels, n_total_labels, only_valid_labels
    Returns a list of selected frame dicts.
    """
    if N >= len(frame_infos):
        return frame_infos
    # Sort by: only_valid_labels desc, n_valid_labels desc, n_total_labels desc, then middle frame
    sorted_frames = sorted(
        frame_infos,
        key=lambda x: (
            x['only_valid_labels'],
            x['n_valid_labels'],
            x['n_total_labels']
        ),
        reverse=True
    )
    if N == 1:
        # If tie, pick the middle frame among the best
        best_score = (
            sorted_frames[0]['only_valid_labels'],
            sorted_frames[0]['n_valid_labels'],
            sorted_frames[0]['n_total_labels']
        )
        tied = [f for f in sorted_frames if (
            f['only_valid_labels'], f['n_valid_labels'], f['n_total_labels']) == best_score]
        if len(tied) > 1:
            mid = len(tied) // 2
            return [tied[mid]]
        else:
            return [sorted_frames[0]]
    else:
        return sorted_frames[:N]

def collect_image_label_pairs(input_dir: str, logger=None, N: int = N) -> List[Tuple[str, str, str]]:
    """
    Return list of (image_path, label_path, unique_name) for CADICA detection dataset.
    Only includes frames from selectedVideos, lesionVideos, and selectedFrames, and filters out frames with only prohibited labels.
    unique_name: CADICAp{pnum}v{vnum}frame{framenum}_{unique}
    If N is given, only select up to N frames per video, using the following rules:
      1. Prefer frames with only valid labels
      2. Then, most valid labels
      3. Then, most total labels
      4. Then, middle frame among ties
    """
    input_path = Path(input_dir) / "selectedVideos"
    pairs = []
    unique_counter = 1
    for patient_dir in sorted(input_path.iterdir()):
        if not patient_dir.is_dir() or not patient_dir.name.startswith("p"):
            continue
        pnum = patient_dir.name[1:]
        lesion_videos_file = patient_dir / "lesionVideos.txt"
        if not lesion_videos_file.exists():
            if logger:
                logger.warning(f"No lesionVideos.txt in {patient_dir}")
            continue
        with open(lesion_videos_file) as f:
            lesion_videos = set(line.strip() for line in f if line.strip())
        for video_dir in sorted(patient_dir.iterdir()):
            if not video_dir.is_dir() or not video_dir.name.startswith("v"):
                continue
            vnum = video_dir.name[1:]
            if video_dir.name not in lesion_videos:
                continue
            selected_frames_file = video_dir / f"p{pnum}_v{vnum}_selectedFrames.txt"
            input_frames_dir = video_dir / "input"
            gt_dir = video_dir / "groundtruth"
            if not (selected_frames_file.exists() and input_frames_dir.exists() and gt_dir.exists()):
                if logger:
                    logger.warning(f"Missing selectedFrames, input, or groundtruth in {video_dir}")
                continue
            with open(selected_frames_file) as f:
                selected_frames = [line.strip() for line in f if line.strip()]
            frame_infos = []
            for frame_id in selected_frames:
                img_file = input_frames_dir / f"{frame_id}.png"
                label_file = gt_dir / f"{frame_id}.txt"
                if not (img_file.exists() and label_file.exists()):
                    if logger:
                        logger.warning(f"Missing image or label for frame {frame_id} in {video_dir}")
                    continue
                n_valid_labels = 0
                n_total_labels = 0
                only_valid_labels = True
                with open(label_file) as lf:
                    lines = [line.strip() for line in lf if line.strip()]
                    for line in lines:
                        parts = line.split()
                        if len(parts) == 5:
                            n_total_labels += 1
                            if parts[4] not in PROHIBITED_LABELS:
                                n_valid_labels += 1
                            else:
                                only_valid_labels = False
                if n_valid_labels == 0:
                    continue
                frame_infos.append({
                    'frame_id': frame_id,
                    'img_file': img_file,
                    'label_file': label_file,
                    'n_valid_labels': n_valid_labels,
                    'n_total_labels': n_total_labels,
                    'only_valid_labels': only_valid_labels
                })
            if N is not None and N > 0:
                selected_frames = _select_best_frames(frame_infos, N)
            else:
                selected_frames = frame_infos
            for frame in selected_frames:
                unique_name = f"CADICAp{pnum}v{vnum}frame{frame['frame_id'].split('_')[-1]}_{unique_counter:06d}"
                pairs.append((str(frame['img_file']), str(frame['label_file']), unique_name))
                unique_counter += 1
    if logger:
        logger.info(f"Total CADICA pairs: {len(pairs)}")
    return pairs
