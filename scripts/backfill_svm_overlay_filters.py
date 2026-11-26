#!/usr/bin/env python3
"""Render outcome-filtered saved SVM trajectories; never rerun policies or rewards."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'OGPO'))
import imageio.v2 as imageio
import numpy as np
from ogpo.svm_logging import render_reward_overlay

BACKGROUND_CODE='''
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
cfg=json.loads(Path(sys.argv[1]).read_text())['spec'];state_id=int(sys.argv[2]);out=Path(sys.argv[3])
repo=Path(cfg['policy_repo']);sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'LIBERO'))
from experiments.reward_audit.envs import prepare_libero_paths,observation
prepare_libero_paths('/tmp/svm-overlay-background-'+str(cfg['seed']),repo)
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
suite=benchmark.get_benchmark_dict()[cfg['suite']]()
env=OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(cfg['task_id']),camera_heights=256,camera_widths=256)
try:
 env.seed(cfg['seed']+10000);env.reset();raw=env.set_init_state(np.asarray(suite.get_task_init_states(cfg['task_id']))[state_id])
 for _ in range(10):raw,_,_,_=env.step(np.array([0,0,0,0,0,0,-1],np.float32))
 Image.fromarray(observation(raw)['agentview_image']).save(out)
finally:env.close()
'''


def atomic_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def background(directory,manifest):
    if manifest.get('background') and (directory/manifest['background']).is_file():
        return imageio.imread(directory/manifest['background']), 'captured initial RGB'
    run=directory.parents[2]
    cfg=json.loads((run/'launch.json').read_text())['spec']
    state=manifest['background_init_state_id']
    cached=run/'eval_artifacts'/f'background_state_{state}.png'
    if not cached.exists():
        env=os.environ.copy();gpu=str(cfg.get('policy_gpu',0))
        env.update(CUDA_VISIBLE_DEVICES=gpu,MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID=gpu,EGL_DEVICE_ID=gpu)
        subprocess.run([cfg['sim_python'],'-c',BACKGROUND_CODE,str(run/'launch.json'),str(state),str(cached)],
                       env=env,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    rgb=imageio.imread(cached)
    # The first recorded frame is the actual initial image, before any policy action.
    if manifest.get('videos'):
        reader=imageio.get_reader(str(directory/manifest['videos'][0]))
        try:frame=reader.get_data(0)
        finally:reader.close()
        error=float(np.abs(frame.astype(float)-rgb.astype(float)).mean())
        if error>5:
            raise ValueError(f'Regenerated background does not match recording: MAE={error:.2f} ({directory})')
    return rgb,'regenerated from official reset state, original simulator seed and10 settle actions; checked against initial video frames where available'


def fill(path):
    manifest=json.loads(path.read_text());directory=path.parent
    if manifest.get('overlays') and all((directory/name).is_file() for name in manifest['overlays'].values()):
        return False
    rgb,provenance=background(directory,manifest)
    trajectories=[]
    with np.load(directory/manifest['trajectory_data'],allow_pickle=False) as data:
        for i,result in enumerate(manifest['episode_results']):
            trajectories.append(dict(success=result['success'],pixels=data[f'episode_{i:02d}_pixels'],
                                     process_rewards=data[f'episode_{i:02d}_process_rewards']))
    cfg=json.loads((directory.parents[2]/'launch.json').read_text())['spec']
    svm=cfg.get('svm',{})
    bound=float(manifest.get('reward_color_bound',svm.get('scale',.05)*svm.get('clip',20.)))
    variants={}
    for outcome,name in [('all','reward_overlay.png'),('success','reward_overlay_success.png'),('failure','reward_overlay_failure.png')]:
        render_reward_overlay(rgb,trajectories,directory/name,physical_step=manifest['physical_step'],
            mode=manifest['mode'],seed=manifest['seed'],bound=bound,ready=manifest['discriminator_ready'],outcome=outcome)
        variants[outcome]=name
    imageio.imwrite(directory/'background.png',rgb)
    manifest.update(overlays=variants,background='background.png',background_provenance=provenance,reward_color_bound=bound)
    atomic_json(path,manifest)
    print('Outcome overlays:',directory,flush=True)
    return True


def refresh_page_metadata(campaign):
    status=json.loads((campaign/'status.json').read_text())
    if any(run['status'] in ('running','queued') for run in status['runs']):return
    # The training monitor also passes these manifest fields through on its next poll.
    # Patch the page copy as well so completed/idle campaigns update immediately.
    for name in ['svm-run.json','svm-replicates.json']:
        path=ROOT/'docs/experiment_atlas'/name
        page=json.loads(path.read_text())
        if page.get('campaign_id')!=str(campaign):continue
        changed=False
        for item in page.get('eval_artifacts',[]):
            source=campaign/f"seed{item['seed']}"/'eval_artifacts'/f"step_{item['physical_step']:06d}"/item['mode']/'manifest.json'
            if not source.exists():continue
            saved=json.loads(source.read_text())
            if saved.get('overlays') and item.get('overlays')!=saved['overlays']:
                item.update(overlays=saved['overlays']);changed=True
        if changed:
            # Do not race the training monitor's fixed .tmp filename.
            temp=path.with_name(path.name+'.overlay-tmp');temp.write_text(json.dumps(page,indent=2)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=ROOT.parent/'experiment_results/ogpo_libero/svm_verified_20261123_seeds01')
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();campaign=args.campaign.resolve()
    while True:
        for path in sorted(campaign.glob('seed*/eval_artifacts/step_*/*/manifest.json')):
            fill(path)
        refresh_page_metadata(campaign)
        status=json.loads((campaign/'status.json').read_text())
        if not args.watch:return
        if all(run['status'] not in ('running','queued') for run in status['runs']):
            # Include a final manifest that may have landed during the previous scan.
            for path in sorted(campaign.glob('seed*/eval_artifacts/step_*/*/manifest.json')):
                fill(path)
            refresh_page_metadata(campaign)
            return
        time.sleep(15)


if __name__=='__main__':main()
