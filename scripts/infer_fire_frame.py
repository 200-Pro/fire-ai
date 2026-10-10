"""Run a single API/image frame with the versioned fire filtering policy."""
import argparse
from pathlib import Path

from fire_policy import filter_predictions
from training_workflow import make_id, read_json, sha256, write_json


def predict_frame(weights, source, output, policy_path, device='cpu'):
    from PIL import Image
    from ultralytics import YOLO
    weights, source, policy_path = Path(weights), Path(source), Path(policy_path)
    if source.suffix.lower() not in ('.jpg','.jpeg','.png','.bmp','.webp'):
        raise ValueError('This command is for image frames only, not MP4')
    config = read_json(policy_path)
    if sha256(weights) != config['model_sha256']:
        raise ValueError('Policy was not validated for this model')
    raw, policy = config['raw_predict'], config['filter']
    if raw['conf'] > policy['weak_conf']:
        raise ValueError('Raw prediction threshold discards policy candidates')
    model = YOLO(str(weights))
    if model.names != {0:'smoke',1:'fire'}:
        raise ValueError('Expected trained fire/smoke model')
    # Decoding through Pillow avoids Korean Windows file path problems.
    with Image.open(source) as image:
        image = image.convert('RGB')
    result = model.predict(image, device=device, save=False, verbose=False, **raw)[0]
    predictions = [dict(class_id=int(b.cls.item()), score=float(b.conf.item()),
                        xyxy=b.xyxy[0].tolist()) for b in result.boxes]
    indices = filter_predictions(predictions,policy,result.orig_shape)
    target = Path(output).resolve()/make_id('fire_frame')
    target.mkdir(parents=True,exist_ok=False)
    chosen = [predictions[i] for i in indices]
    write_json(target/'detections.json',dict(source=str(source),weights_sha256=sha256(weights),
        policy_sha256=sha256(policy_path),policy=config,shape=list(result.orig_shape),
        fire_candidate=any(p['class_id']==1 for p in chosen),
        smoke_candidate=any(p['class_id']==0 for p in chosen),detections=chosen,
        note='Frame-level candidates only. Not a confirmed fire incident.'))
    result.update(boxes=result.boxes.data[indices])
    Image.fromarray(result.plot()[:,:,::-1]).save(target/'annotated.jpg',quality=90)
    print('Filtered frame saved:',target,flush=True)
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policy',type=Path,default=Path(__file__).resolve().parents[1]/'configs/fire_policy_v1.json')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    predict_frame(args.weights,args.source,args.output,args.policy,args.device)
