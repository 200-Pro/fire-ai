"""Tune conservative postprocessing on saved VAL predictions; never read test.

Choose once on stratified 70% calibration; audit remaining 30% once. Audit/all
guard failure yields no deployable config. Shared validation is NOT new test.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from analyze_fire_misses import match_fire
from fire_policy import filter_predictions
from training_workflow import make_id, read_json, sha256, write_json


BASELINE = dict(strong_conf=.25, weak_conf=.25, weak_support='none',
                duplicate_iou=1., smoke_conf=.25)


def metrics(records, policy, reference=None):
    out = Counter(tp=0, fp=0, fn=0, image_alarm_tp=0, image_alarm_fn=0,
                  no_fire_image_alarm_fp=0, no_fire_image_alarm_tn=0,
                  lost_baseline_positive_images=0)
    for record in records:
        predictions = record['predictions']
        chosen = [predictions[i] for i in filter_predictions(predictions, policy, record['shape'])]
        fire = [p for p in chosen if p['class_id'] == 1]
        matched = match_fire(record['gt_fire'], fire, 0.)
        out.update({k: matched[k] for k in ('tp', 'fp', 'fn')})
        if record['gt_fire']:
            out['image_alarm_tp' if fire else 'image_alarm_fn'] += 1
            baseline_predictions = reference[record['file_name']]['predictions'] if reference else predictions
            baseline_alarm = any(p['class_id'] == 1 and p['score'] >= .25 for p in baseline_predictions)
            out['lost_baseline_positive_images'] += baseline_alarm and not fire
        else:
            out['no_fire_image_alarm_fp' if fire else 'no_fire_image_alarm_tn'] += 1
    out['precision'] = out['tp']/(out['tp']+out['fp']) if out['tp']+out['fp'] else 0.
    out['recall'] = out['tp']/(out['tp']+out['fn']) if out['tp']+out['fn'] else 0.
    return dict(out)


def guard(current, baseline):
    return (current['tp'] >= baseline['tp'] and current['fp'] <= baseline['fp']
            and current['image_alarm_tp'] >= baseline['image_alarm_tp']
            and current['no_fire_image_alarm_fp'] <= baseline['no_fire_image_alarm_fp']
            and current['lost_baseline_positive_images'] == 0)


def candidates():
    # Simple bounded policies, not label-specific pixel/filename exceptions.
    for weak_conf in (.25, .23, .22, .20, .18, .15):
        for duplicate_iou in (.7, .65, .6, .55, .5, .45, .4):
            supports = [('none', .25)] if weak_conf == .25 else [
                ('image_strong_fire', .25), ('near_strong_fire', .15),
                ('near_strong_fire', .25), ('near_strong_fire', .4)]
            for support, radius in supports:
                yield dict(strong_conf=.25, weak_conf=weak_conf, weak_support=support,
                           support_radius=radius, duplicate_iou=duplicate_iou, smoke_conf=.25)
    for strong_conf in (.30, .35, .40):
        for weak_conf in (.25, .22, .20):
            for smoke_conf in (.3, .5, .7):
                for coverage in (.2, .5):
                    for duplicate_iou in (.65, .55, .45):
                        yield dict(strong_conf=strong_conf, weak_conf=weak_conf,
                            weak_support='fire_or_smoke', support_radius=.4,
                            support_smoke_conf=smoke_conf, support_smoke_coverage=coverage,
                            duplicate_iou=duplicate_iou, smoke_conf=.25)


def tune(source, output, baseline_source=None):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = read_json(source / 'summary.json')
    if (summary['split'] != 'val' or summary['imgsz'] not in (640,960)
            or summary['rect'] is not False or summary.get('benchmark_only',False)):
        raise ValueError('Use full VAL predictions at imgsz=640/960 and rect=False')
    if summary['prediction_floor'] != .05 or summary['nms_iou'] != .7:
        raise ValueError('Raw prediction floor/NMS differ')
    records = [json.loads(line) for line in (source / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(records) != summary['images'] or len({r['file_name'] for r in records}) != len(records):
        raise ValueError('Incomplete/duplicate prediction records')
    baseline_summary = read_json(Path(baseline_source)/'summary.json') if baseline_source else summary
    reference_records = ([json.loads(line) for line in (Path(baseline_source)/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
                         if baseline_source else records)
    reference = {r['file_name']:r for r in reference_records}
    if set(reference) != {r['file_name'] for r in records}:
        raise ValueError('Baseline image IDs differ')
    if baseline_summary['model_sha256'] != summary['model_sha256']:
        raise ValueError('Baseline model differs')
    for record in records:
        if reference[record['file_name']]['gt_fire'] != record['gt_fire']:
            raise ValueError('Baseline ground truth differs')
    calibration, audit = [], []
    for label in (False, True):
        group = sorted((r for r in records if bool(r['gt_fire']) == label), key=lambda r: r['file_name'])
        random.Random(20261005).shuffle(group)
        boundary = int(len(group)*.7)
        calibration.extend(group[:boundary])
        audit.extend(group[boundary:])
    base_cal, base_audit, base_all = (metrics([reference[r['file_name']] for r in subset], BASELINE)
                                    for subset in (calibration, audit, records))
    original = next(r for r in baseline_summary['threshold_sweep'] if r['confidence'] == .25)
    assert all(base_all[k] == original[k] for k in ('tp', 'fp', 'fn'))
    trials = []
    best = None
    for index, policy in enumerate(candidates(), 1):
        score = metrics(calibration, policy, reference)
        trials.append(dict(policy=policy, calibration=score, guard_pass=guard(score, base_cal)))
        if guard(score, base_cal) and score['fp'] < base_cal['fp']:
            key = (score['no_fire_image_alarm_fp'], score['fp'], -score['tp'],
                   -policy['weak_conf'], -policy['duplicate_iou'])
            if best is None or key < best[0]:
                best = (key, policy, score)
        if index % 10 == 0:
            print(f'Calibration policies checked: {index}', flush=True)
    write_json(output / 'calibration_trials.json', trials)
    split = dict(seed=20261005, calibration=[r['file_name'] for r in calibration],
                 audit=[r['file_name'] for r in audit],
                 note='Stratified IMAGE split within previously inspected val. '
                      'No incident/camera grouping available; NOT independent final test.')
    write_json(output / 'policy_split.json', split)
    if best is None:
        result = dict(state='no_safe_improvement', baseline=base_all)
    else:
        policy = best[1]
        # No second parameter choice based on audit performance.
        audit_result, all_result = metrics(audit, policy, reference), metrics(records, policy, reference)
        approved = guard(audit_result, base_audit) and guard(all_result, base_all)
        result = dict(state='validated_candidate' if approved else 'audit_guard_failed',
            imgsz=summary['imgsz'],baseline_imgsz=baseline_summary['imgsz'],
            policy=policy, baseline_calibration=base_cal, candidate_calibration=best[2],
            baseline_audit=base_audit, candidate_audit=audit_result,
            baseline=base_all, candidate=all_result, calibration_images=len(calibration),
            audit_images=len(audit), candidate_count=len(trials),
            selection='Minimize calibration box FP subject to no baseline box/image recall loss '
                      'and no increase in fire-absent-image false alarms. Then one audit pass.',
            limitations='All material is prior val; this is not independent test or real CCTV validation.')
        if approved:
            write_json(output / 'fire_policy_v1.json', dict(version='fire-policy-v1',
                model_sha256=summary['model_sha256'], source_predictions_sha256=sha256(source/'predictions.jsonl'),
                raw_predict=dict(conf=.05, iou=.7, imgsz=summary['imgsz'], rect=False, max_det=300),
                filter=policy, validation=result))
    write_json(output / 'tuning_summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-source',type=Path)
    args = parser.parse_args()
    tune(args.source, args.output,args.baseline_source)
