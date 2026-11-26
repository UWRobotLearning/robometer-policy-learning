#!/usr/bin/env python3
"""Launch two SVM seeds, optionally after a previous pair completes."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_success_visitation import ROOT, SPEC_PATH, SPEC_DIR, BC_SPEC_PATH, baseline, evaluations, progress, logging_data, write_json
from run_svm_replicates import points, summarize

PAGE = ROOT/'docs/experiment_atlas'


def read_console(out):
    path=Path(out)/'console.log'
    if not path.exists():return ''
    with path.open('rb') as stream:
        # Activation is near the beginning, progress is near the end.
        start=stream.read(200000)
        stream.seek(max(0,path.stat().st_size-100000))
        return (start+b'\n'+stream.read()).decode(errors='replace')


def diagnostics(out):
    result=dict(svm_active='[SVM] Activated:' in read_console(out), discriminator_updates=0)
    logs=list(Path(out).glob('logs/**/svm.csv'))
    if logs:
        with logs[0].open() as stream:rows=list(csv.DictReader(stream))
        if rows:
            row=rows[-1]
            for key in ['updates','positive_transitions','negative_transitions','skipped_missing_class']:
                if row.get(key):result['discriminator_updates' if key=='updates' else key]=int(float(row[key]))
    return result


def publish(jobs, source, campaign, previous=()):
    jobs=[*previous,*jobs]
    seeds=','.join(str(j['seed']) for j in jobs)
    now=datetime.now(timezone.utc).isoformat()
    telemetry=[logging_data(j['output_dir'],j['seed']) for j in jobs]
    measured=[p for j in jobs for p in points(j['output_dir'],j['seed'])]
    runs=[{k:v for k,v in j.items() if k not in ('process','log','config')} for j in jobs]
    sparse=baseline()
    cohort=dict(campaign_id=str(campaign),updated_at=now,runs=runs,evaluations=measured,
        training_episodes=[r for t in telemetry for r in t['training_episodes']],
        eval_artifacts=[r for t in telemetry for r in t['eval_artifacts']],
        summaries={'SVM':summarize(measured),'Historical sparse':summarize(sparse)},
        interval=f'95% Student-t across the {len(jobs)} available RL seeds; no interval for n=1.',
        cohort_note=f'Corrected SVM seeds {seeds} share the selected30-demo BC checkpoint. The old inactive SVM-labeled runs are excluded. Historical sparse uses a different50-demo BC start.')
    bc=json.loads(BC_SPEC_PATH.read_text())
    base=dict(campaign_id=str(campaign),status='completed' if all(j['status']=='completed' for j in jobs) else 'rl_training',
        message=f'Corrected SVM seeds {seeds};100k physical steps each. First evaluate the restored BC baseline, then initialize SVM and train.',
        updated_at=now,physical_steps=jobs[0]['physical_steps'],budget=100000,seed=0,
        bc_updates=5000,bc_budget=5000,bc_evaluations=evaluations(bc['output_dir']),
        bc_selection=source['bc_source'],baseline=sparse,
        spec='/experiments/libero/specs/svm/'+Path(jobs[0]['spec']).relative_to(SPEC_DIR).as_posix(),
        bc_spec='/experiments/libero/specs/svm/'+BC_SPEC_PATH.name,
        baseline_source='/docs/reports/ogpo_success_summary_20261031/evaluation_points.csv',
        **{k:cohort[k] for k in ['runs','evaluations','training_episodes','eval_artifacts','summaries']})
    write_json(PAGE/'svm-run.json',base)
    write_json(PAGE/'svm-replicates.json',cohort)
    write_json(campaign/'status.json',cohort)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',default='svm_verified_20261123_seeds01')
    parser.add_argument('--python',default='/home/rohan/Robotics/Real-World-RL/OGPO_public/.venv/bin/python')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--seeds',type=int,nargs=2,default=[0,1])
    parser.add_argument('--after-campaign',type=Path,help='Wait for both runs to complete successfully, then include them in the atlas.')
    args=parser.parse_args()
    if len(set(args.seeds))!=2 or min(args.seeds)<0:raise SystemExit('Choose two distinct nonnegative seeds')
    if Path(args.campaign).name!=args.campaign:raise SystemExit('Campaign must be a single directory name')
    source=json.loads(SPEC_PATH.read_text())
    checkpoint=Path(source['overrides']['checkpoint.restore_bc_path'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=source['bc_source']['sha256']:
        raise SystemExit('Selected BC checkpoint hash mismatch')
    for path in [source['demo_file'],source['encoder_checkpoint'],source['feature_cache'],source['sim_python'],args.python,
                 source['overrides']['agent.paligemma_tokenizer_path']]:
        if not Path(path).exists():raise SystemExit('Missing input: '+path)
    campaign=ROOT.parent/'experiment_results/ogpo_libero'/args.campaign
    if any((campaign/name).exists() for name in ['status.json','queue.json']):raise SystemExit('Campaign already exists; refusing to overwrite')
    campaign.mkdir(parents=True,exist_ok=True)
    specs=SPEC_DIR/args.campaign
    jobs=[]
    for gpu,seed in enumerate(args.seeds):
        cfg=json.loads(json.dumps(source));cfg.pop('bc_selection_spec',None)
        out=campaign/f'seed{seed}'
        cfg.update(seed=seed,policy_gpu=gpu,output_dir=str(out),reward='svm',
            comparison_id=args.campaign,eval_logging=dict(enabled=True,record_seed=0,record_episodes=4),
            note='Corrected SVM activation; fresh100k RL from selected30-demo BC2000. Paper Table5 discriminator cadence. No old inactive runs included.')
        cfg['svm']=dict(scale=.05,clip=20.,learning_rate=1e-4,batch_size=64,gradient_steps=32,update_every=1000)
        cfg['overrides'].update({'experiment.seed':seed,'experiment.run_group':args.campaign,
            'checkpoint.reseed_restored_rng':True,'training.bc_pi_steps':0,'training.bc_only':False,
            'training.online_steps':100000,'eval.eval_episodes':25,'eval.n_eval_envs':1,
            'intervals.eval_interval':10000,'eval.video_episodes':0})
        path=specs/f'libero90_task02_svm_100k_seed{seed}.json';write_json(path,cfg)
        jobs.append(dict(seed=seed,gpu=gpu,status='queued',physical_steps=0,phase='queued',
                         svm_active=False,discriminator_updates=0,output_dir=str(out),spec=str(path),config=cfg))
    if not args.launch:
        print('Prepared corrected SVM seeds',args.seeds,'. BC hash verified. Specs:',specs);return 0
    previous=[]
    if args.after_campaign:
        dependency=args.after_campaign.resolve()
        if dependency==campaign.resolve():raise SystemExit('A campaign cannot depend on itself')
        queue=dict(status='waiting',pid=os.getpid(),seeds=args.seeds,
                   after_campaign=str(dependency),created_at=datetime.now(timezone.utc).isoformat())
        write_json(campaign/'queue.json',queue)
        print('Waiting for successful completion:',dependency,flush=True)
        while True:
            prior=json.loads((dependency/'status.json').read_text())
            states=[run['status'] for run in prior['runs']]
            if not states:raise SystemExit('Dependency has no runs')
            if any(state in ('failed','stopped') for state in states):
                queue.update(status='blocked',reason='Previous campaign failed or stopped')
                write_json(campaign/'queue.json',queue)
                raise SystemExit(queue['reason'])
            if all(state=='completed' for state in states):
                previous=prior['runs']
                if set(args.seeds)&{run['seed'] for run in previous}:raise SystemExit('Seeds overlap previous campaign')
                break
            time.sleep(15)
        queue.update(status='launching',dependency_completed_at=datetime.now(timezone.utc).isoformat())
        write_json(campaign/'queue.json',queue)
    history=PAGE/'history'/('before_'+args.campaign);history.mkdir(parents=True,exist_ok=True)
    for filename in ['svm-run.json','svm-replicates.json']:
        if (PAGE/filename).exists():shutil.copy2(PAGE/filename,history/filename)
    try:
        for job in jobs:
            out=Path(job['output_dir']);out.mkdir()
            command=[args.python,'-u',str(ROOT/'OGPO/scripts/libero.py'),'run','--spec',job['spec'],'--gpu',str(job['gpu'])]
            job['log']=(out/'console.log').open('a')
            job['process']=subprocess.Popen(command,cwd=ROOT/'OGPO',stdout=job['log'],stderr=subprocess.STDOUT)
            job.update(status='running',pid=job['process'].pid,started=datetime.now(timezone.utc).isoformat())
            write_json(out/'launch.json',dict(pid=job['pid'],command=command,started=time.time(),spec=job['config']))
            print(f"Seed{job['seed']} started on GPU{job['gpu']}, PID{job['pid']}",flush=True)
        while any(j['status']=='running' for j in jobs):
            for job in jobs:
                if job['status']!='running':continue
                job.update(diagnostics(job['output_dir']))
                job['physical_steps']=progress(job['output_dir'],'Online RL',100000)
                job['phase']='RL training' if job['svm_active'] else 'initialization / BC baseline evaluation'
                console=read_console(job['output_dir'])
                # Refuse the silent-deactivation failure before allowing online RL to proceed.
                if 'Online RL:' in console and not job['svm_active']:
                    job['process'].terminate();job['error']='Online RL began without SVM activation'
                code=job['process'].poll()
                if code is not None:
                    valid=code==0 and job['svm_active'] and job['discriminator_updates']>0 and job['physical_steps']>=100000
                    job.update(status='completed' if valid else 'failed',returncode=code,finished=datetime.now(timezone.utc).isoformat())
                    job['log'].close()
                    print(f"Seed{job['seed']} {job['status']}",flush=True)
            publish(jobs,source,campaign,previous)
            if any(j['status']=='running' for j in jobs):time.sleep(15)
    except KeyboardInterrupt:
        for job in jobs:
            if job['status']=='running':
                job['process'].terminate();job['process'].wait();job['log'].close();job['status']='stopped'
        publish(jobs,source,campaign,previous);return 130
    if args.after_campaign:
        queue.update(status='completed' if all(j['status']=='completed' for j in jobs) else 'failed')
        write_json(campaign/'queue.json',queue)
    return int(any(j['status']!='completed' for j in jobs))


if __name__=='__main__':
    sys.exit(main())
