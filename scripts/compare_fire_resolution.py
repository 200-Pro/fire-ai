"""Read-only validation-only CPU 640/960 comparison using the existing model.

Only copies official validation images into a new local directory. Verifies ZIP,
manifest, model and normalized labels; never extracts train/test or trains.
"""
import argparse
import csv
import hashlib
import json
import random
import time
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

from analyze_fire_misses import read_fire_truth, THRESHOLDS, match_fire
from training_workflow import copy_verified, read_json, sha256, write_json


def materialize(run, archive, work, saved_predictions, archive_hash_mode='val-crc'):
    from prepare_dfire import normalize_label_text
    manifest = run / 'dataset_manifest_v001.csv'
    metadata = read_json(run / 'run.json')
    assert sha256(manifest) == metadata['dataset_manifest_sha256']
    if archive_hash_mode == 'full':
        print('Checking original ZIP hash...', flush=True)
        assert sha256(archive) == metadata['source_sha256']['dataset']
    else:
        print('Checking validation-only ZIP members by CRC, label hash and saved geometry; '
              'full ZIP hash is NOT rechecked.', flush=True)
    weights = run / 'final/best.pt'
    expected = '57c133a489976dc2f7341381b12cf160622e0bbaff33aeffdee4926e2093611d'
    assert sha256(weights) == expected
    work.mkdir(parents=True, exist_ok=False)
    images, labels = work/'val/images', work/'val/labels'
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r['split'] == 'val']
    reference = {r['file_name']: r for r in saved_predictions}
    assert set(reference) == {r['file_name'] for r in rows}
    recovered, entries = [], []
    with zipfile.ZipFile(archive) as z:
        for index, row in enumerate(rows, 1):
            name = row['file_name']
            assert Path(name).name == name
            raw_label = z.read('train/labels/'+Path(name).stem+'.txt')
            cleaned, changes = normalize_label_text(raw_label.decode('utf-8-sig'), row['label_path'])
            label_bytes = cleaned.encode('utf-8') if changes else raw_label
            assert hashlib.sha256(label_bytes).hexdigest() == row['label_sha256'], name
            label_path = labels/(Path(name).stem+'.txt')
            label_path.write_bytes(label_bytes)
            image_path = images/name
            image_bytes = z.read('train/images/'+name)
            image_path.write_bytes(image_bytes)
            if image_bytes[-2:] != b'\xff\xd9':
                with Image.open(image_path) as image:
                    restored = ImageOps.exif_transpose(image).convert('RGB')
                restored.save(image_path, 'JPEG', quality=100, subsampling=0)
                recovered.append(name)
            with Image.open(image_path) as image:
                shape = (image.height, image.width)
            assert list(shape) == reference[name]['shape'], name
            gt = read_fire_truth(label_path, shape)
            expected_gt = reference[name]['gt_fire']
            assert len(gt) == len(expected_gt)
            assert all(abs(a-b) < 1e-5 for box, old in zip(gt, expected_gt) for a,b in zip(box,old))
            entries.append(dict(file_name=name, shape=list(shape), gt_fire=gt,
                                image_sha256=sha256(image_path), label_sha256=sha256(label_path)))
            if index % 300 == 0:
                print(f'Validation images prepared: {index}/{len(rows)}', flush=True)
    copy_verified(weights, work/'weights/best.pt')
    write_json(work/'ready.json', dict(images=len(entries), entries=entries,
        model_sha256=expected, training_zip_sha256=metadata['source_sha256']['dataset'],
        archive_verification=archive_hash_mode,
        manifest_sha256=sha256(manifest), jpeg_repairs=recovered,
        note='Local validation only. Original ZIP/model untouched.'))


def predict_resolution(work, output, size, benchmark=False):
    import torch
    from ultralytics import YOLO
    from collections import Counter
    torch.set_num_threads(4)
    ready = read_json(work/'ready.json')
    assert sha256(work/'weights/best.pt') == ready['model_sha256']
    for entry in ready['entries']:
        assert sha256(work/'val/images'/entry['file_name']) == entry['image_sha256']
        assert sha256(work/'val/labels'/(Path(entry['file_name']).stem+'.txt')) == entry['label_sha256']
    entries = {e['file_name']: e for e in ready['entries']}
    model = YOLO(str(work/'weights/best.pt'))
    assert model.names == {0:'smoke', 1:'fire'}
    if benchmark:
        chosen = random.Random(42).sample(sorted(entries), min(16,len(entries)))
        source = [str(work/'val/images'/name) for name in chosen]
    else:
        source = str(work/'val/images')
    target = output/str(size)
    target.mkdir(parents=True, exist_ok=False)
    counts = {c: Counter(tp=0,fp=0,fn=0) for c in THRESHOLDS}
    start, seen = time.monotonic(), set()
    with (target/'predictions.jsonl').open('w',encoding='utf-8') as handle:
        for result in model.predict(source, stream=True, conf=.05, iou=.7, imgsz=size,
                rect=False, max_det=300, batch=1, device='cpu', save=False, verbose=False):
            name = Path(result.path).name
            assert name not in seen and name in entries
            seen.add(name)
            predictions = [dict(class_id=int(b.cls.item()), score=float(b.conf.item()),
                                 xyxy=b.xyxy[0].tolist()) for b in result.boxes]
            record = dict(file_name=name,shape=list(result.orig_shape),
                          gt_fire=entries[name]['gt_fire'],predictions=predictions)
            handle.write(json.dumps(record)+'\n')
            for conf in THRESHOLDS:
                matches = match_fire(record['gt_fire'],predictions,conf)
                counts[conf].update({k:matches[k] for k in ('tp','fp','fn')})
            if len(seen)%100==0:
                elapsed=time.monotonic()-start
                print(f'imgsz={size}: {len(seen)}/{len(entries)} images; elapsed {elapsed:.1f}s',flush=True)
                handle.flush()
    elapsed=time.monotonic()-start
    if not benchmark:
        assert seen==set(entries)
    write_json(target/'summary.json',dict(split='val',model_sha256=ready['model_sha256'],
        dataset_manifest_sha256=ready['manifest_sha256'], images=len(seen),
        prediction_floor=.05,nms_iou=.7,imgsz=size,rect=False,max_det=300,
        batch=1,device='cpu',torch_threads=torch.get_num_threads(), elapsed_seconds=elapsed,
        seconds_per_image=elapsed/len(seen),benchmark_only=benchmark,
        threshold_sweep=[dict(confidence=c,**dict(v)) for c,v in counts.items()],
        note='Compare same local CPU runtime. Prior Colab GPU numbers can differ slightly. '
             'Benchmark samples are timing checks only, not performance evidence.'))
    print(f'Completed imgsz={size}, images={len(seen)}, seconds={elapsed:.1f}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--benchmark-only',action='store_true')
    parser.add_argument('--archive-hash-mode',choices=['val-crc','full'],default='val-crc')
    parser.add_argument('--sizes',type=int,nargs='+',default=[640,960])
    args=parser.parse_args()
    if not (args.work/'ready.json').is_file():
        reference=[json.loads(line) for line in args.reference.read_text(encoding='utf-8').splitlines()]
        materialize(args.run,args.archive,args.work,reference,args.archive_hash_mode)
    for size in args.sizes:
        predict_resolution(args.work,args.output,size,args.benchmark_only)
