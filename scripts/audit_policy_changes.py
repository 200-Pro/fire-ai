"""Export reproducible policy comparison and both improvement/regression examples."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from analyze_fire_misses import draw_review, match_fire
from fire_policy import filter_predictions
from tune_fire_policy import BASELINE, metrics
from training_workflow import copy_verified, read_json, sha256, write_json


def audit(source, policy_path, images, output, resolution_results):
    source, policy_path, images, output = map(Path,(source,policy_path,images,output))
    output.mkdir(parents=True,exist_ok=False)
    config=read_json(policy_path)
    records=[json.loads(line) for line in (source/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    baseline=metrics(records,BASELINE)
    candidate=metrics(records,config['filter'])
    gains, losses, changes = [], [], []
    lookup={r['file_name']:r for r in records}
    for record in records:
        predictions=record['predictions']
        chosen=[predictions[i] for i in filter_predictions(predictions,config['filter'],record['shape'])]
        old=set(match_fire(record['gt_fire'],predictions,.25)['missed_indices'])
        new=set(match_fire(record['gt_fire'],chosen,0.)['missed_indices'])
        for index in sorted(old-new):
            gains.append(dict(file_name=record['file_name'],gt_index=index))
        for index in sorted(new-old):
            losses.append(dict(file_name=record['file_name'],gt_index=index))
        if old != new:
            changes.append(dict(file_name=record['file_name'],before_missed=sorted(old),after_missed=sorted(new)))
    assert candidate['fn'] == baseline['fn']-len(gains)+len(losses)
    write_json(output/'box_changes.json',dict(recovered_boxes=gains,newly_missed_boxes=losses,image_changes=changes))
    comparison=dict(model_sha256=config['model_sha256'],source_predictions_sha256=sha256(source/'predictions.jsonl'),
        policy_sha256=sha256(policy_path),images=len(records),baseline=baseline,candidate=candidate,
        recovered_boxes=len(gains),newly_missed_boxes=len(losses),
        raw_cpu640=read_json(Path(resolution_results)/'640/summary.json'),
        raw_cpu960=read_json(Path(resolution_results)/'960/summary.json'),
        note='All validation, no independent test. Aggregate miss count improves but individual boxes can regress.')
    write_json(output/'comparison.json',comparison)
    copy_verified(policy_path,output/'fire_policy_v1.json')
    (output/'examples').mkdir()
    names=['AoF04266.jpg']
    for rows in (gains,losses):
        chosen_names=[]
        for row in rows:
            if row['file_name'] not in names+chosen_names:
                chosen_names.append(row['file_name'])
            if len(chosen_names)==3:
                break
        names.extend(chosen_names)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    for name in names:
        record=lookup[name]
        predictions=record['predictions']
        before=[p for p in predictions if p['score']>=.25]
        after=[predictions[i] for i in filter_predictions(predictions,config['filter'],record['shape'])]
        paths=[]
        for label,preds in (('before',before),('after',after)):
            missed=match_fire(record['gt_fire'],preds,0.)['missed_indices']
            path=output/'examples'/f'{Path(name).stem}_{label}.jpg'
            draw_review(images/name,record['gt_fire'],preds,missed,path,display_conf=0.)
            paths.append(path)
        sheet=Image.new('RGB',(1440,500),'#202020')
        draw=ImageDraw.Draw(sheet)
        for index,path in enumerate(paths):
            image=ImageOps.contain(Image.open(path).convert('RGB'),(710,452))
            sheet.paste(image,(index*720+(720-image.width)//2,42+(452-image.height)//2))
            draw.text((index*720+8,6),f'{name} {"BEFORE" if index==0 else "AFTER"}',fill='white',font=font)
        sheet.save(output/'examples'/f'{Path(name).stem}_comparison.jpg',quality=95)
    print(json.dumps(dict(baseline=baseline,candidate=candidate,recovered=len(gains),new_missed=len(losses),examples=names),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resolution-results',type=Path,required=True)
    args=parser.parse_args()
    audit(args.source,args.policy,args.images,args.output,args.resolution_results)
