#!/usr/bin/env python3
"""Train BC on 30 task-2 demos, select its checkpoint, then run 100k-step SVM RL."""
import argparse
import csv
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PAGE_DATA = ROOT / 'docs/experiment_atlas/svm-run.json'
SPEC_DIR = ROOT / 'experiments/libero/specs/svm'
SPEC_PATH = SPEC_DIR / 'libero90_task02_svm_100k_seed100.json'
BC_SPEC_PATH = SPEC_DIR / 'libero90_task02_bc30demos_5k_seed100.json'
EVIDENCE = ROOT / 'docs/reports/ogpo_success_summary_20261031/evidence.json'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def baseline():
    """Reuse captured 100k sparse runs, keeping their different BC provenance visible."""
    result = []
    with (EVIDENCE.parent / 'evaluation_points.csv').open() as stream:
        for row in csv.DictReader(stream):
            if re.fullmatch(r'libero_90_task02_sparse_100k_seed10[0-4]_20261030', row['run']):
                result.append(dict(run=row['run'], seed=int(row['seed']), mode=row['mode'],
                    step=float(row['physical_steps_per_env']), success=float(row['success_rate'])))
    return result


def evaluations(out):
    result = []
    flags = list(Path(out).glob('logs/**/flags.json'))
    if len(flags) == 1:
        for mode, name in [('ODE','eval.csv'), ('SDE','eval_sde.csv')]:
            path = flags[0].with_name(name)
            if path.exists():
                with path.open() as stream:
                    for row in csv.DictReader(stream):
                        try:
                            result.append(dict(run=Path(out).name, seed=100, mode=mode,
                                step=float(row['step']), success=float(row['success'])))
                        except (KeyError, ValueError, TypeError):
                            continue
    return result


def progress(out, phase, budget):
    console = Path(out) / 'console.log'
    if not console.exists():
        return 0
    with console.open('rb') as stream:
        stream.seek(max(0, console.stat().st_size - 300000))
        matches = re.findall(re.escape(phase) + r':.*?(\d+)/' + str(budget), stream.read().decode(errors='replace'))
    return int(matches[-1]) if matches else 0


def logging_data(directory, seed):
    """Only measured completed episodes and manifests; no reconstructed old rollouts."""
    output = Path(directory)
    episodes = []
    for path in output.glob('logs/**/training_episodes.csv'):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                try:
                    episodes.append(dict(seed=seed,physical_step=int(row['physical_step']),
                        completed_episodes=int(row['completed_episodes']),window_episodes=int(row['window_episodes']),
                        window_successes=int(row['window_successes']),success=float(row['rolling_success_rate'])))
                except (ValueError,TypeError,KeyError):
                    continue
    artifacts = []
    for path in sorted(output.glob('eval_artifacts/step_*/*/manifest.json')):
        item=json.loads(path.read_text())
        def artifact_url(name):
            target=path.parent/name
            try:
                return '/experiment-artifacts/'+target.relative_to(ROOT.parent/'experiment_results').as_posix()
            except ValueError:
                return None
        item.update(overlay_url=artifact_url(item['overlay']),data_url=artifact_url(item['trajectory_data']),
                    video_urls=[artifact_url(name) for name in item['videos']])
        artifacts.append(item)
    return dict(training_episodes=episodes,eval_artifacts=artifacts)


def publish(cfg, bc, state, message='', returncode=None):
    write_json(PAGE_DATA, dict(campaign_id=cfg['output_dir'],status=state, message=message,
        updated_at=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
        physical_steps=progress(cfg['output_dir'],'Online RL',100000), budget=100000, seed=100,
        bc_updates=progress(bc['output_dir'],'BC Training',5000), bc_budget=5000,
        bc_evaluations=evaluations(bc['output_dir']), bc_selection=cfg.get('bc_source'),
        baseline=baseline(), evaluations=evaluations(cfg['output_dir']), returncode=returncode,
        **logging_data(cfg['output_dir'],cfg['seed']),
        spec='/experiments/libero/specs/svm/'+SPEC_PATH.name,
        bc_spec='/experiments/libero/specs/svm/'+BC_SPEC_PATH.name,
        baseline_source='/docs/reports/ogpo_success_summary_20261031/evaluation_points.csv'))


def prepare(args):
    evidence = json.loads(EVIDENCE.read_text())
    original = next(r['spec'] for r in evidence['runs'] if r['name']=='libero_90_task02_sparse_100k_seed100_20261030')
    cfg = json.loads(json.dumps(original))
    for key in ['required_sanity','colocation_sanity','bc_source','bc_selection_spec','reward_gpu','reward_checkpoint']:
        cfg.pop(key, None)
    out=Path(args.output_dir).resolve()
    bc_out=Path(args.bc_output_dir).resolve() if args.bc_output_dir else out.with_name(out.name+'_bc30demos_5k')
    cfg.update(policy_repo=str(ROOT), sim_python=str(Path(args.sim_python).absolute()),
        output_dir=str(out), demo_file=str(Path(args.demo_file).resolve()), num_demonstrations=30,
        encoder_checkpoint=str(Path(args.encoder_checkpoint).resolve()), feature_cache=str(bc_out/'features.npy'),
        seed=100, reward='svm', eval_logging=dict(enabled=True,record_seed=args.record_seed,record_episodes=4), save_replay_buffer=False, comparison_id='task02_svm_bc30demos_selected_100k_seed100',
        svm=dict(scale=.05, clip=20., learning_rate=1e-4, batch_size=64, gradient_steps=32, update_every=1000),
        note='Fresh BC on 30 demos for 5000 updates, evaluate 25 episodes/mode every500. Select first ODE in [25%,50%], otherwise lowest positive ODE. SVM uses0/1 outcomes; historical sparse used-1/0 and a different50-demo BC start. No demos used for discriminator labels.')
    overrides=cfg['overrides']
    overrides.update({'checkpoint.restore_bc_path':None, 'checkpoint.reseed_restored_rng':True,
        'training.bc_pi_steps':0,'training.bc_only':False,'training.online_steps':100000,
        'experiment.run_group':'task02_svm_bc30demos_100k_seed100',
        'agent.paligemma_tokenizer_path':str(Path(args.tokenizer).resolve()),
        'intervals.eval_interval':10000,'eval.eval_episodes':25,'eval.video_episodes':0})
    bc=json.loads(json.dumps(cfg))
    bc.pop('svm');bc.update(output_dir=str(bc_out),reward='sparse_minus_one',
        comparison_id='task02_bc30demos_5k_seed100',note='30 demos, full5000 BC updates; ODE selects [25%,50%], lowest positive fallback; all-zero blocks RL.')
    bc['overrides'].update({'training.bc_only':True,'training.bc_pi_steps':5000,'training.online_steps':0,
        'intervals.eval_interval_bc':500,'intervals.save_interval_bc':500,'eval.eval_episodes':25,
        'bc.clip_bc':False,'agent.bc_select_policy':'interval_or_smallest_positive',
        'agent.bc_select_success_min':.25,'agent.bc_select_success_max':.5,
        'agent.bc_select_max_inclusive':True,'agent.bc_select_stop':False,
        'experiment.run_group':'task02_bc30demos_5k_seed100'})
    cfg['bc_selection_spec']=str(BC_SPEC_PATH)
    missing=[]
    for key in ['sim_python','demo_file','encoder_checkpoint']:
        if not Path(cfg[key]).exists():missing.append(f'{key}: {cfg[key]}')
    for key,value in [('tokenizer',args.tokenizer),('ogpo_python',args.ogpo_python)]:
        if not Path(value).exists():missing.append(f'{key}: {value}')
    return cfg,bc,missing


def run_phase(args, phase_cfg, spec_path, cfg, bc, state):
    out=Path(phase_cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    command=[args.ogpo_python,'-u',str(ROOT/'OGPO/scripts/libero.py'),'run','--spec',str(spec_path),'--gpu',str(args.gpu)]
    with (out/'console.log').open('a') as log:
        process=subprocess.Popen(command,cwd=ROOT/'OGPO',stdout=log,stderr=subprocess.STDOUT)
        write_json(out/'launch.json',dict(pid=process.pid,command=command,started=time.time(),spec=phase_cfg))
        try:
            while process.poll() is None:
                publish(cfg,bc,state,'BC:30 demos,5000 updates,25 episodes every500; then selected-checkpoint SVM RL for100k steps.')
                time.sleep(15)
        except KeyboardInterrupt:
            process.terminate();process.wait()
            publish(cfg,bc,'stopped','Interrupted; exact replay resume is unsupported.',process.returncode)
            return 130
    if process.returncode:
        publish(cfg,bc,'failed',f'{state} exited {process.returncode}; inspect {out}/console.log',process.returncode)
    return process.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    assets=Path('/home/rohan/Robotics/Real-World-RL/experiments/pi05_latency/checkpoints')
    parser.add_argument('--demo-file',default=str(ROOT.parent/'experiment_results/datasets/libero_90/KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_demo.hdf5'))
    parser.add_argument('--encoder-checkpoint',default=str(assets/'pi05_libero/params'))
    parser.add_argument('--tokenizer',default=str(assets/'paligemma_tokenizer.model'))
    parser.add_argument('--sim-python',default=str(ROOT.parent/'mpail-research/.venv-libero/bin/python'))
    parser.add_argument('--ogpo-python',default='/home/rohan/Robotics/Real-World-RL/OGPO_public/.venv/bin/python')
    parser.add_argument('--output-dir',default=str(ROOT.parent/'experiment_results/ogpo_libero/libero_90_task02_svm_bc30demos_100k_seed100'))
    parser.add_argument('--bc-output-dir',default=None)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--record-seed',type=int,default=0,help='Only this literal training seed records the first4 scored episodes per sampler')
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args();cfg,bc,missing=prepare(args)
    if args.launch and list(Path(cfg['output_dir']).glob('logs/**/flags.json')):
        raise SystemExit('RL output already contains a run; choose a fresh output directory.')
    write_json(SPEC_PATH,cfg);write_json(BC_SPEC_PATH,bc)
    if missing:
        publish(cfg,bc,'missing_inputs','\n'.join(missing));print('Run not launched. Missing inputs:\n'+'\n'.join(missing));return 2
    if not args.launch:
        publish(cfg,bc,'ready','Inputs verified. Fresh30-demo BC and100k-step SVM RL are prepared.')
        print('Preflight passed. Add --launch to train BC and then SVM RL.');return 0
    existing=list(Path(bc['output_dir']).glob('logs/**/flags.json'))
    if not existing:
        code=run_phase(args,bc,BC_SPEC_PATH,cfg,bc,'bc_training')
        if code:return code
    # Reuse BC only if the entire requested schedule finished, including both final evaluations.
    rows=evaluations(bc['output_dir'])
    for mode in ['ODE','SDE']:
        if not any(r['mode']==mode and r['step']==5000 for r in rows):
            publish(cfg,bc,'failed','BC did not finish5000 updates with final ODE/SDE evaluations.');return 1
    resolver=runpy.run_path(str(ROOT/'OGPO/scripts/libero.py'))['resolve_bc_selection']
    try:
        cfg=resolver(cfg)
    except RuntimeError as error:
        publish(cfg,bc,'failed',str(error));return 1
    write_json(SPEC_PATH,cfg)
    code=run_phase(args,cfg,SPEC_PATH,cfg,bc,'rl_training')
    if code:return code
    publish(cfg,bc,'completed','BC selection and100k physical SVM training steps completed.',0)
    return 0


if __name__=='__main__':
    sys.exit(main())
