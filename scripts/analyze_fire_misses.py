"""Validation-only fire FN review and confidence sweep; never trains or relabels.

One-to-one, confidence-ordered box matching at IoU >= 0.5. These fixed-threshold
counts are NOT Ultralytics' mAP or its best-F1 operating point.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from training_workflow import (
    copy_verified, make_id, read_json, sha256, verify_prepared_dataset, write_json,
)

THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.35, 0.50)
BASELINE_CONF = 0.25
PREDICTION_FLOOR = 0.05
MATCH_IOU = 0.50


def box_iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(
        0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.


def match_fire(truth, predictions, confidence, match_iou=MATCH_IOU):
    """Fire-only matching; each prediction and GT can be used at most once."""
    selected = sorted(
        [p for p in predictions if p['class_id'] == 1 and p['score'] >= confidence],
        key=lambda p: p['score'], reverse=True,
    )
    unmatched = set(range(len(truth)))
    matches = []
    for pred in selected:
        if not unmatched:
            break
        index = max(sorted(unmatched), key=lambda i: box_iou(truth[i], pred['xyxy']))
        overlap = box_iou(truth[index], pred['xyxy'])
        if overlap >= match_iou:
            unmatched.remove(index)
            matches.append((index, pred, overlap))
    return dict(tp=len(matches), fn=len(unmatched), fp=len(selected)-len(matches),
                missed_indices=sorted(unmatched), matches=matches)


def missed_box_hint(box, predictions):
    """Review hints, not a causal diagnosis or a claim of annotation error."""
    fire = [p for p in predictions if p['class_id'] == 1]
    overlaps = [(box_iou(box, p['xyxy']), p['score']) for p in fire]
    aligned = [p for p in fire if box_iou(box, p['xyxy']) >= MATCH_IOU]
    smoke_overlap = max((box_iou(box, p['xyxy']) for p in predictions
                         if p['class_id'] == 0 and p['score'] >= BASELINE_CONF), default=0.)
    if aligned:
        best_score = max(p['score'] for p in aligned)
        hint = ('low_confidence_candidate' if best_score < BASELINE_CONF
                else 'matching_competition_review')
    elif smoke_overlap >= MATCH_IOU:
        best_score, hint = None, 'smoke_overlap_review'
    elif overlaps and max(x[0] for x in overlaps) > 0:
        best_score, hint = None, 'localization_or_weak_overlap'
    else:
        best_score, hint = None, 'no_overlapping_fire_at_conf005'
    return dict(hint=hint, aligned_fire_score=best_score,
                best_fire_iou=max((x[0] for x in overlaps), default=0.),
                best_smoke_iou=smoke_overlap)


def read_fire_truth(label_path, shape):
    h, w = shape
    boxes = []
    for line in label_path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        cls, x, y, bw, bh = map(float, line.split())
        if int(cls) == 1:
            boxes.append([(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h])
    return boxes


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw_review(image_path, truth, predictions, missed, target, display_conf=BASELINE_CONF):
    import cv2
    import numpy as np
    # imdecode/imencode also work with Korean Windows paths.
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode {image_path}')
    missed = set(missed)
    for index, box in enumerate(truth):
        color = (0, 0, 255) if index in missed else (0, 180, 0)
        p1, p2 = tuple(map(round, box[:2])), tuple(map(round, box[2:]))
        cv2.rectangle(image, p1, p2, color, 2)
        cv2.putText(image, 'GT fire MISS' if index in missed else 'GT fire matched',
                    (p1[0], max(15, p1[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    for pred in predictions:
        if pred['score'] < display_conf:
            continue
        color = (255, 120, 0) if pred['class_id'] == 1 else (0, 180, 255)
        p1, p2 = tuple(map(round, pred['xyxy'][:2])), tuple(map(round, pred['xyxy'][2:]))
        cv2.rectangle(image, p1, p2, color, 1)
        name = 'fire' if pred['class_id'] == 1 else 'smoke'
        cv2.putText(image, f'PRED {name} {pred["score"]:.2f}',
                    (p1[0], min(image.shape[0]-5, p2[1]+15)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise IOError('JPEG output failed')
    data.tofile(target)


def analyze(run_dir, dataset, local_output, *, device='0', batch=16,
            max_examples=30, expected_model_sha256=None):
    """All validation images, including backgrounds, to quantify extra false alarms."""
    from ultralytics import YOLO
    run_dir, dataset = Path(run_dir), Path(dataset)
    if batch < 1 or max_examples < 0:
        raise ValueError('batch >= 1 and max_examples >= 0 required')
    summary = verify_prepared_dataset(dataset)
    saved = read_json(run_dir / 'run.json')
    for key, actual in (
        ('split_sha256', sha256(dataset / 'split_v001.csv')),
        ('source_sha256', summary['source_sha256']),
        ('dataset_manifest_sha256', sha256(dataset / 'dataset_manifest_v001.csv')),
    ):
        if saved[key] != actual:
            raise ValueError(f'Data differs from training: {key}')
    weights = run_dir / 'final' / 'best.pt'
    model_hash = sha256(weights)
    if expected_model_sha256 and model_hash != expected_model_sha256:
        raise ValueError('Selected model hash differs from expected model')
    name = make_id('fire_fn_val')
    target = Path(local_output).resolve() / name
    target.mkdir(parents=True, exist_ok=False)
    local_weights = target / 'weights' / 'best.pt'
    copy_verified(weights, local_weights)
    model = YOLO(str(local_weights))
    if model.names != {0: 'smoke', 1: 'fire'}:
        raise ValueError('Expected trained D-Fire model: 0=smoke, 1=fire')
    with (dataset / 'dataset_manifest_v001.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = {r['file_name']: r for r in csv.DictReader(f) if r['split'] == 'val'}
    totals = {c: Counter(tp=0, fp=0, fn=0, fire_images_missed_all=0,
                         images_with_fire_fp=0, no_fire_images_with_fire_fp=0) for c in THRESHOLDS}
    missed_boxes, missed_images, seen = [], [], set()
    gt_count = fire_image_count = 0
    with (target / 'predictions.jsonl').open('w', encoding='utf-8') as log:
        for result in model.predict(source=str(dataset / 'val' / 'images'),
                stream=True, conf=PREDICTION_FLOOR, iou=0.7, imgsz=640,
                rect=False, max_det=300, batch=batch, device=device, save=False, verbose=False):
            file_name = Path(result.path).name
            if file_name not in rows or file_name in seen:
                raise ValueError(f'Unexpected/duplicate image: {file_name}')
            seen.add(file_name)
            row = rows[file_name]
            truth = read_fire_truth(dataset / row['label_path'], result.orig_shape)
            predictions = [dict(class_id=int(b.cls.item()), score=float(b.conf.item()),
                                xyxy=b.xyxy[0].tolist()) for b in result.boxes]
            log.write(json.dumps(dict(file_name=file_name, shape=list(result.orig_shape),
                                      gt_fire=truth, predictions=predictions)) + '\n')
            gt_count += len(truth)
            fire_image_count += bool(truth)
            baseline = None
            for conf in THRESHOLDS:
                matched = match_fire(truth, predictions, conf)
                totals[conf].update({k: matched[k] for k in ('tp', 'fp', 'fn')})
                totals[conf]['fire_images_missed_all'] += bool(truth) and matched['tp'] == 0
                totals[conf]['images_with_fire_fp'] += matched['fp'] > 0
                totals[conf]['no_fire_images_with_fire_fp'] += not truth and matched['fp'] > 0
                if conf == BASELINE_CONF:
                    baseline = matched
            if baseline['fn']:
                missed_images.append(dict(file_name=file_name, gt_fire=truth,
                                          predictions=predictions,
                                          missed_indices=baseline['missed_indices']))
                for index in baseline['missed_indices']:
                    box = truth[index]
                    h, w = result.orig_shape
                    missed_boxes.append(dict(file_name=file_name, gt_index=index,
                        x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                        area_fraction=(box[2]-box[0])*(box[3]-box[1])/(w*h),
                        **missed_box_hint(box, predictions)))
            if len(seen) % 200 == 0:
                print(f'Validation images: {len(seen)}/{len(rows)}', flush=True)
    if seen != set(rows):
        raise ValueError('Not all validation images were processed')
    sweep = []
    for conf, counts in totals.items():
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        sweep.append(dict(confidence=conf, **dict(counts),
                          precision=tp/(tp+fp) if tp+fp else None,
                          recall=tp/(tp+fn) if tp+fn else None))
    write_csv(target / 'threshold_sweep.csv', sweep, list(sweep[0]))
    fields = ['file_name', 'gt_index', 'x1', 'y1', 'x2', 'y2', 'area_fraction',
              'hint', 'aligned_fire_score', 'best_fire_iou', 'best_smoke_iou']
    write_csv(target / 'fire_missed_boxes.csv', missed_boxes, fields)
    selected = random.Random(42).sample(sorted(missed_images, key=lambda r: r['file_name']),
                                        min(max_examples, len(missed_images)))
    examples = target / 'examples'
    examples.mkdir()
    for row in selected:
        draw_review(dataset / rows[row['file_name']]['image_path'], row['gt_fire'],
                    row['predictions'], row['missed_indices'], examples / row['file_name'])
    report = dict(split='val', run=str(run_dir), model_sha256=model_hash,
        dataset_manifest_sha256=saved['dataset_manifest_sha256'],
        images=len(seen), fire_images=fire_image_count, gt_fire_boxes=gt_count,
        baseline_conf=BASELINE_CONF, prediction_floor=PREDICTION_FLOOR,
        match_iou=MATCH_IOU, nms_iou=0.7, imgsz=640, rect=False, batch=batch,
        device=str(device), max_det=300, seed=42,
        baseline_missed_images=len(missed_images), baseline_missed_boxes=len(missed_boxes),
        hint_counts=dict(Counter(r['hint'] for r in missed_boxes)), threshold_sweep=sweep,
        saved_examples=[r['file_name'] for r in selected],
        note='Fixed-confidence greedy box matching; NOT mAP or official best-F1 recall. '
             'FP includes unmatched/duplicate fire predictions. Hints require visual review. '
             'Smoke can physically overlap fire; smoke overlap is not proof of wrong classification. '
             'Threshold sweep reuses predictions at conf=0.05 with the same NMS settings. '
             'No training, relabeling, test evaluation, or model changes occurred.')
    write_json(target / 'summary.json', report)
    (target / 'review_notes.md').write_text(
        '# 불꽃 미탐 검토\n\n'
        '빨강: 놓친 fire 정답 박스, 초록: 검출과 매칭된 fire 정답 박스, '
        '파랑: fire 예측, 노랑: smoke 예측. 예측은 conf >= 0.25만 표시합니다.\n\n'
        'threshold_sweep.csv에서 recall 증가와 fp 증가를 함께 비교하세요. '
        '이 recall은 고정 conf/IoU의 별도 분석이며 기존 val 지표와 동일하지 않습니다.\n\n'
        '힌트는 원인 확정이 아닙니다. low_confidence_candidate는 낮은 점수 후보가 있고, '
        'smoke_overlap_review는 연기 박스와 겹침, localization_or_weak_overlap은 위치 검토 대상입니다. '
        '매칭 경쟁이나 작은 불꽃, 가림, 야간, 밝은 반사, 라벨 상태를 사람이 확인하세요.\n\n'
        '원인별 개선은 train 자료로만 진행하세요. validation 이미지를 train으로 이동하거나 '
        'test를 보며 threshold/학습 설정을 선택하지 마세요. 이미 확인한 test 점수는 기존 모델의 '
        '기준 결과로 보관하고, 개선 모델의 최종 주장은 새로운 미사용 평가 자료로 검증하세요.\n',
        encoding='utf-8')
    # Drive stores only reports/predictions and capped examples, not dataset or model duplicates.
    destination = run_dir / 'analysis' / name
    destination.mkdir(parents=True, exist_ok=False)
    for source in target.rglob('*'):
        if source.is_file() and 'weights' not in source.relative_to(target).parts:
            copy_verified(source, destination / source.relative_to(target))
    print(f'Completed {len(seen)} images. Report saved: {destination}', flush=True)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--max-examples', type=int, default=30)
    parser.add_argument('--expected-model-sha256')
    args = parser.parse_args()
    analyze(args.run, args.dataset, args.output, device=args.device, batch=args.batch,
            max_examples=args.max_examples, expected_model_sha256=args.expected_model_sha256)
