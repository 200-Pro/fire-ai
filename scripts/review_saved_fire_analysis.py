"""Audit saved validation predictions only; no new inference or source modification."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from analyze_fire_misses import box_iou, match_fire, missed_box_hint, THRESHOLDS
from training_workflow import read_json, sha256, write_json


def size_bin(side):
    return '<8px' if side < 8 else '8-<16px' if side < 16 else '16-<32px' if side < 32 else '>=32px'


def review(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = read_json(source / 'summary.json')
    records = [json.loads(line) for line in (source / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(records) == summary['images']
    assert len({r['file_name'] for r in records}) == len(records)
    stat = {c: Counter(tp=0, fp=0, fn=0, fire_images_missed_all=0,
                      images_with_fire_fp=0, no_fire_images_with_fire_fp=0,
                      fire_image_alarm_tp=0, fire_image_alarm_fn=0,
                      no_fire_image_alarm_fp=0, no_fire_image_alarm_tn=0) for c in THRESHOLDS}
    sizes, fp_types, iou_bands, hints = defaultdict(Counter), Counter(), Counter(), Counter()
    boxes, image_details = [], []
    fire_images = 0
    for rec in records:
        gt, preds = rec['gt_fire'], rec['predictions']
        fire_images += bool(gt)
        baseline = match_fire(gt, preds, .25)
        baseline_ids = {id(p) for _, p, _ in baseline['matches']}
        for p in [p for p in preds if p['class_id'] == 1 and p['score'] >= .25]:
            if id(p) in baseline_ids:
                continue
            best = max((box_iou(p['xyxy'], b) for b in gt), default=0.)
            kind = ('no_fire_ground_truth_image' if not gt else
                    'duplicate_or_matching_competition' if best >= .5 else
                    'overlap_but_iou_below_05' if best > 0 else
                    'no_overlap_with_annotated_fire')
            fp_types[kind] += 1
        for conf in THRESHOLDS:
            m = match_fire(gt, preds, conf)
            alarm = any(p['class_id'] == 1 and p['score'] >= conf for p in preds)
            stat[conf].update({k: m[k] for k in ('tp', 'fp', 'fn')})
            stat[conf]['fire_images_missed_all'] += bool(gt) and m['tp'] == 0
            stat[conf]['images_with_fire_fp'] += m['fp'] > 0
            stat[conf]['no_fire_images_with_fire_fp'] += not gt and m['fp'] > 0
            if gt:
                stat[conf]['fire_image_alarm_tp' if alarm else 'fire_image_alarm_fn'] += 1
            else:
                stat[conf]['no_fire_image_alarm_fp' if alarm else 'no_fire_image_alarm_tn'] += 1
        detail = dict(file_name=rec['file_name'], gt_fire_boxes=len(gt),
                      missed_boxes_conf025=baseline['fn'],
                      fire_predictions_conf025=sum(p['class_id'] == 1 and p['score'] >= .25 for p in preds),
                      recovered_gt_at_conf010=[], missed_gt_at_conf025=baseline['missed_indices'])
        low_missed = set(match_fire(gt, preds, .1)['missed_indices'])
        scale = 640 / max(rec['shape'])
        for index, box in enumerate(gt):
            missed = index in baseline['missed_indices']
            side = min(box[2]-box[0], box[3]-box[1]) * scale
            group = size_bin(side)
            sizes[group]['gt_boxes'] += 1
            sizes[group]['missed_boxes'] += missed
            candidates = [p for p in preds if p['class_id'] == 1 and p['score'] >= .25]
            best = max(candidates, key=lambda p: box_iou(box, p['xyxy']), default=None)
            best_iou = box_iou(box, best['xyxy']) if best else 0.
            row = dict(file_name=rec['file_name'], gt_index=index, gt_xyxy=box,
                       image_shape=rec['shape'], min_side_at_imgsz640=side, size_bin=group,
                       missed_at_conf025=missed, missed_at_conf010=index in low_missed,
                       best_iou_at_conf025=best_iou,
                       best_iou_conf025_score=best['score'] if best else None)
            if missed:
                hint = missed_box_hint(box, preds)
                hints[hint['hint']] += 1
                row.update(hint)
                band = ('0' if best_iou == 0 else '0-<0.25' if best_iou < .25 else
                        '0.25-<0.4' if best_iou < .4 else '0.4-<0.5' if best_iou < .5 else '>=0.5')
                iou_bands[band] += 1
                if index not in low_missed:
                    detail['recovered_gt_at_conf010'].append(index)
            boxes.append(row)
        image_details.append(detail)
    assert fire_images == summary['fire_images']
    assert len(boxes) == summary['gt_fire_boxes']
    assert dict(hints) == summary['hint_counts']
    for row in summary['threshold_sweep']:
        for key in ('tp', 'fp', 'fn', 'fire_images_missed_all', 'images_with_fire_fp', 'no_fire_images_with_fire_fp'):
            assert stat[row['confidence']][key] == row[key], (row['confidence'], key)
    for counts in sizes.values():
        counts['miss_rate'] = counts['missed_boxes'] / counts['gt_boxes']
    for c, counts in stat.items():
        counts['fire_image_alarm_recall'] = counts['fire_image_alarm_tp'] / fire_images
        counts['no_fire_image_alarm_rate'] = counts['no_fire_image_alarm_fp'] / (len(records)-fire_images)
    write_json(output / 'derived_metrics.json', dict(
        source=str(source), predictions_sha256=sha256(source / 'predictions.jsonl'),
        model_sha256=summary['model_sha256'], images=len(records), fire_images=fire_images,
        no_fire_images=len(records)-fire_images, boxes=len(boxes),
        hint_counts=dict(hints), size_analysis=dict(sizes), fp_types_conf025=dict(fp_types),
        missed_gt_best_iou_conf025=dict(iou_bands),
        threshold_metrics=[dict(confidence=c, **dict(counts)) for c, counts in stat.items()],
        note='Image alarm = any fire prediction, ignoring location. NOT incident recall or FP/hour. '
             'Size bins use GT minimum side after 640 longest-side resize. No causal claim.'))
    write_json(output / 'all_gt_box_audit.json', boxes)
    write_json(output / 'all_image_audit.json', image_details)
    # Read-only diagnostic montages of already annotated examples, not new model output.
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 24)
    names = sorted(summary['saved_examples'])
    for start in range(0, len(names), 6):
        sheet = Image.new('RGB', (1440, 1380), '#202020')
        draw = ImageDraw.Draw(sheet)
        for offset, name in enumerate(names[start:start+6]):
            x, y = (offset % 2)*720, (offset // 2)*460
            image = Image.open(source / 'examples' / name).convert('RGB')
            image = ImageOps.contain(image, (710, 414))
            sheet.paste(image, (x+(720-image.width)//2, y+40+(414-image.height)//2))
            draw.text((x+10, y+6), name, fill='white', font=font)
        sheet.save(output / f'contact_{start//6+1:02}.jpg', quality=95)
    print(json.dumps(read_json(output / 'derived_metrics.json'), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review(args.source, args.output)
