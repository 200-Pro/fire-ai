"""Opt-in CPU integration check using real YOLO11, without ANY training.

Creates an untrained two-class YOLO11n fixture, validates synthetic images,
checks copied files/JSON, then exercises the media adapter. Scores are not
project performance measurements. Outputs live only in a temporary directory.
"""
import json
import tempfile
from pathlib import Path

from workflow_fixture import prepared_dataset, evaluation_run
import training_workflow as wf


def main():
    from ultralytics import YOLO, settings
    from ultralytics.nn.tasks import DetectionModel
    from infer_media import predict
    settings.update({'mlflow':False,'wandb':False})
    with tempfile.TemporaryDirectory(prefix='fire_validation_check_') as tmp:
        root=Path(tmp)
        data=prepared_dataset(root)
        model=YOLO('yolo11n.yaml')
        model.model=DetectionModel('yolo11n.yaml',nc=2,verbose=False)
        model.model.names={0:'smoke',1:'fire'}
        fixture=root/'untrained_yolo11n_fixture.pt'
        model.save(str(fixture))
        run=evaluation_run(root,data,fixture)
        result=wf.evaluate(run,data,root/'local',device='cpu')
        assert result['split']=='val'
        assert [c['name'] for c in result['per_class']]==['smoke','fire']
        assert (run/'validation_summary.json').is_file()
        assert list((run/'evaluations').glob('*/confusion_matrix.png'))
        assert wf.collect_results(root/'drive')[0]['state']=='validated'
        prediction=predict(run/'final/best.pt',data/'val/images/c.jpg',root/'inference')
        rows=[json.loads(line) for line in (prediction/'detections.jsonl').read_text().splitlines()]
        assert len(rows)==1 and rows[0]['frame_index']==0
        print('PASS: real YOLO11 CPU validation -> callback -> copied plots -> metrics JSON -> team summary -> image inference')
        print('Training was NOT run. Synthetic fixture metrics are NOT model quality results.')


if __name__=='__main__':
    main()
