import unittest
from experiments.ogpo_dashboard import phase_offset, points, wilson, chart, remaining_time, progress_timing, experiment_series, native_remaining_time, shared_bc, aggregate_reward_curves, comparison_groups


class OGPOPlotTests(unittest.TestCase):
    def test_multiple_rl_seeds_share_group_and_ignore_missing_evaluations(self):
        runs=[]
        for seed,rate in [(100,.2),(101,.4),(102,.6)]:
            runs.append({'spec': {'suite':'libero_90','task_id':2,'seed':seed,
                                 'comparison_id':'shared','reward':'sparse_minus_one',
                                 'bc_source':{'checkpoint':'same'}},
                         'budget':50000,'n':25,'offset':0,
                         'evaluation':{'ODE':[{'step':0,'success':rate}], 'SDE':[]}})
        runs[0]['evaluation']['ODE'].append({'step':20000,'success':.8})
        self.assertEqual(len(comparison_groups(runs)), 1)
        curve=aggregate_reward_curves(runs,'ODE')[0]
        self.assertEqual(curve['counts'], [3,1])
        self.assertAlmostEqual(curve['points'][0][1],.4)
        self.assertEqual(curve['points'][1], (20000,.8))
        self.assertEqual(curve['bounds'][0][1],0)
        self.assertAlmostEqual(curve['bounds'][0][2],.8968275423439091)

    def test_shared_bc_is_once_per_pair_and_keeps_its_update_axis(self):
        parent = {'config': {'training': {'bc_pi_steps': 5000}},
                  'bc_progress': (1000, 5000, 60), 'state': 'Behavior cloning',
                  'runtime': 'Elapsed: 1m', 'eta': 'Remaining: 4m',
                  'selection': {'bc_updates': 500, 'ode_success': .32},
                  'evaluation': {'ODE': [{'step': '500', 'success': '.32'}],
                                 'SDE': [{'step': '500', 'success': '.40'}]}}
        group = [{'bc_run': parent}, {'bc_run': parent}]
        series, status = shared_bc(group)
        self.assertEqual(len(series), 2)
        self.assertEqual(series[0][2], [(500, .32)])
        self.assertEqual(status.count('Shared BC</b>'), 1)
        self.assertIn('1,000/5,000 optimizer updates', status)
        self.assertIn('Selected checkpoint: 500', status)
        self.assertEqual(len(shared_bc(group, 'ODE')[0]), 1)
        self.assertEqual(shared_bc([{}]), ([], ''))
        svg = chart(series + [('Sparse ODE', '#087f8c', [(0, .32), (20000, .6)])], 'Shared', probability=True)
        self.assertIn('bc-marker', svg)
        self.assertIn('rl-marker', svg)

    def test_native_eta_includes_remaining_evaluations(self):
        console = 'RL step=20000 updates=1 elapsed=100s\nRL step=30000 updates=2500 elapsed=1500s'
        evaluations = {mode: [{'step': '20000', 'seconds': '200'},
                              {'step': '30000', 'seconds': '200'}] for mode in ('ODE', 'SDE')}
        result = native_remaining_time(console, {'status': 'running', 'steps': 30000},
                                       {'steps': 50000}, evaluations)
        self.assertIn('~47m', result)
        self.assertIn('evaluations', result)
        self.assertIn('no time remaining', native_remaining_time('', {'status': 'completed'}, {}, {}))
        self.assertIn('timing data', native_remaining_time('', {'status': 'running'}, {}, {}))

    def test_rl_eta_uses_elapsed_time_not_tqdm_instantaneous_rate(self):
        log='Online RL:  10%|x| 20000/200000 [1:00:00<999:00:00, 0.1it/s]'
        self.assertEqual(progress_timing(log,'Online RL'), (20000,200000,3600))
        self.assertIn('~9h 00m',remaining_time(log,'Online RL',{},{}))
        self.assertEqual(remaining_time(log,'Completed',{},{}),'')

    def test_feature_eta_handles_resume_and_does_not_claim_total(self):
        features={'completed':1500,'session_start':1000,'seconds_this_session':100,
                  'identity':{'shape':[2000,4104]}}
        result=remaining_time('', 'Extracting features', features,{})
        self.assertIn('~2m',result)
        self.assertIn('total experiment ETA pending',result)

    def test_bc_and_rl_share_numbers_but_remain_separate_series(self):
        task={'spec':{},'offset':20000,'evaluation':{
            'ODE':[{'step':'1000','success':'.36'},{'step':'20000','success':'.76'},
                   {'step':'40000','success':'.84'}], 'SDE':[]}}
        series,_=experiment_series(task)
        values={name:ps for name,_,ps in series}
        self.assertEqual(values['BC ODE'],[(1000,.36),(20000,.76)])
        self.assertEqual(values['RL ODE'],[(0,.76),(20000,.84)])
        svg=chart(series,'Combined',probability=True)
        self.assertIn('class="bc-marker"',svg)
        self.assertIn('class="rl-marker"',svg)

    def test_restored_bc_does_not_shift_rl_by_training_update_count(self):
        task={'spec':{'bc_source':{'bc_updates':1000,'ode_success':.36,'sde_success':.88}},
              'offset':0,'evaluation':{'ODE':[{'step':'0','success':'.32'},{'step':'20000','success':'.72'}], 'SDE':[]}}
        series,_=experiment_series(task)
        self.assertEqual(dict((name,ps) for name,_,ps in series)['RL ODE'],[(0,.32),(20000,.72)])

    def test_actual_bc_offset_not_maximum(self):
        text = '[EVAL ODE] step= 20000 | success=0.76\nOnline RL Phase\n[EVAL ODE] step=20000 | success=0.84'
        self.assertEqual(phase_offset(text), 20000)
        self.assertIsNone(phase_offset('[EVAL ODE] step=20000'))

    def test_offset_and_duplicate_evaluations(self):
        data = [{'step':'20000','success':'.76'}, {'step':'20000','success':'.76'},
                {'step':'40000','success':'.84'}, {'step':'50000','success':''}]
        self.assertEqual(points(data,offset=20000), [(0,.76),(20000,.84)])
        self.assertEqual(points(data,offset=20000,minimum=1), [(20000,.84)])

    def test_episode_intervals(self):
        lo,hi = wilson(19/25,25)
        self.assertAlmostEqual(lo,.5657,places=3)
        self.assertAlmostEqual(hi,.8850,places=3)
        self.assertGreater(wilson(1,25)[0],.86)
        self.assertLess(wilson(0,25)[1],.14)

    def test_no_data_is_not_zero_success(self):
        self.assertIn('No completed measurements', chart([], 'Pending'))

    def test_linear_axes_and_baseline_ci(self):
        svg=chart([('ODE','#087f8c',[(0,.76),(20000,.84)])], 'Success',
                  probability=True,baselines=[('BC ODE','#111827',.76)])
        self.assertIn('Physical RL steps per environment',svg)
        self.assertIn('linear scale',svg)
        self.assertIn('95% Wilson CI',svg)

if __name__ == '__main__':
    unittest.main()
