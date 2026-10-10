"""Tiny synthetic data for pipeline checks, never model-quality evaluation."""
import contextlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import training_workflow as wf
from prepare_dfire import prepare


def prepared_dataset(root):
    from PIL import Image
    root = Path(root)
    archive, splits, data = root/'images.zip', root/'splits.zip', root/'data'
    names = {'train': ['a.jpg','b.jpg'], 'val': ['c.jpg','d.jpg'], 'test': ['e.jpg','f.jpg']}
    with zipfile.ZipFile(archive, 'w') as z:
        for split, files in names.items():
            for index, name in enumerate(files):
                folder = 'test' if split == 'test' else 'train'
                buf = io.BytesIO()
                Image.new('RGB', (64,64), (80+index*40,90,100)).save(buf, format='JPEG')
                z.writestr(f'{folder}/images/{name}',buf.getvalue())
                z.writestr(f'{folder}/labels/{Path(name).stem}.txt', f'{index} 0.5 0.5 0.4 0.4\n')
    with zipfile.ZipFile(splits,'w') as z:
        prefix='Data splitting/5-fold cross validation/'
        z.writestr(prefix+'dfire_train1.txt', '\n'.join(names['train']))
        z.writestr(prefix+'dfire_valid1.txt', '\n'.join(names['val']))
        z.writestr('Data splitting/dfire_test.txt', '\n'.join(names['test']))
    with contextlib.redirect_stdout(io.StringIO()):
        prepare(SimpleNamespace(dataset_zip=archive,split_zip=splits,output=data,fold=1,verify_images=True))
    return data


def evaluation_run(root, data, weights=None):
    root, data = Path(root), Path(data)
    run=root/'drive/yolo11n_v1/fixture'
    (run/'final').mkdir(parents=True)
    if weights:
        shutil.copyfile(weights,run/'final/best.pt')
    else:
        (run/'final/best.pt').write_bytes(b'fake; unit test only')
    summary=wf.read_json(data/'dataset_summary.json')
    config=wf.read_json(wf.PROJECT/'configs/train_yolo11n_v1.json')
    for name in wf.META_FILES:
        shutil.copyfile(data/name,run/name)
    wf.write_json(run/'run.json',dict(member='fixture',run_id='fixture',config=config,
        pretrained_sha256='fixture',source_sha256=summary['source_sha256'],
        dataset_manifest_sha256=wf.sha256(data/'dataset_manifest_v001.csv'),
        split_sha256=wf.sha256(data/'split_v001.csv')))
    wf.write_json(run/'status.json',{'state':'trained'})
    return run
