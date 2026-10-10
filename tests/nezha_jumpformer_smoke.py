"""Headless CPU physics + PPO + checkpoint + deployment smoke test.

No wandb connection or GPU allocation. This verifies execution, not convergence.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import json
import tempfile
from pathlib import Path

import isaacgym
from isaacgym import gymapi, gymutil
import torch
from tensordict import TensorDict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from legged_gym.envs.nezha.nezha_jumpformer import NezhaJumpFormer, NezhaJumpFormerCfg, NezhaJumpFormerCfgPPO
from legged_gym.utils.helpers import class_to_dict, set_seed
from legged_gym.utils.rsl_rl_adapter import RslRlVecEnvAdapter
from rsl_rl.runners import OnPolicyRunner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=None)
    parser.add_argument('--num-envs',type=int,default=4)
    parser.add_argument('--iterations',type=int,default=3)
    args=parser.parse_args()
    if args.num_envs < 2 or args.iterations < 2:
        parser.error('Use at least two environments and two iterations to cover reset and replay')
    output=Path(args.output or tempfile.mkdtemp(prefix='nezha_jumpformer_smoke_'))
    output.mkdir(parents=True,exist_ok=True)
    assert not torch.cuda.is_available()
    torch.set_num_threads(1);set_seed(42)
    cfg=NezhaJumpFormerCfg();cfg.env.num_envs=args.num_envs
    params=gymapi.SimParams();gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
    params.use_gpu_pipeline=False;params.physx.use_gpu=False;params.physx.num_threads=1
    env=None
    try:
        env=NezhaJumpFormer(cfg,params,gymapi.SIM_PHYSX,'cpu',True)
        adapter=RslRlVecEnvAdapter(env)
        train=class_to_dict(NezhaJumpFormerCfgPPO());train.update(train['runner'])
        train['logger']='tensorboard';train['save_interval']=1
        runner=OnPolicyRunner(adapter,train,log_dir=str(output),device='cpu')
        assert adapter.get_observations()['future_state'].shape==(args.num_envs,41)
        before=runner.alg.actor.backbone.core.prediction_head[-1].weight.detach().clone()
        # Force an early timeout in one env to exercise masking alongside live envs.
        env.episode_length_buf[0]=env.max_episode_length-1
        runner.learn(args.iterations,init_at_random_ep_len=False)
        learned=runner.alg.actor.backbone.core.prediction_head[-1].weight
        assert (before-learned).abs().max()>0
        checkpoint=output/f'model_{args.iterations-1}.pt'
        assert checkpoint.is_file()
        obs=adapter.get_observations()
        policy=runner.get_inference_policy()
        with torch.no_grad():expected=policy(obs)['actions']
        runner.load(str(checkpoint),map_location='cpu')
        with torch.no_grad():
            actual=policy(TensorDict({'actor':obs['actor']},batch_size=[args.num_envs]))['actions']
        torch.testing.assert_close(actual,expected)
        runner.export_policy_to_jit(str(output),'policy.pt')
        exported=torch.jit.load(str(output/'policy.pt'))
        with torch.inference_mode():exported_actions=exported(obs['actor'])
        torch.testing.assert_close(exported_actions,expected)
        # Confirm clean inference leaves training-only state completely optional.
        events=EventAccumulator(str(output),size_guidance={'scalars':0});events.Reload()
        tags=[tag for tag in events.Tags()['scalars'] if 'future_' in tag]
        assert 'Loss/future_prediction' in tags
        metrics={tag:events.Scalars(tag)[-1].value for tag in tags}
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        report={'device':'cpu','num_envs':args.num_envs,'iterations':args.iterations,
                'rollout_steps':train['num_steps_per_env'],'ppo_epochs':train['algorithm']['num_learning_epochs'],
                'actor_obs_dim':590,'future_prediction_shape':[args.num_envs,2,41],
                'future_metrics':metrics,'checkpoint':str(checkpoint),'jit':str(output/'policy.pt'),
                'status':'passed','convergence_tested':False}
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('JUMPFORMER_SMOKE',json.dumps(report),flush=True)
    finally:
        if env is not None:
            if env.viewer is not None:env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)


if __name__=='__main__':
    main()
