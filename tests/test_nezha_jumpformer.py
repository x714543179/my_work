"""CPU regressions for future conditioning, label alignment and PPO integration."""

import copy
import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace as NS

import isaacgym  # Must precede torch in Isaac Gym installations.
import torch
from tensordict import TensorDict

from legged_gym.envs.nezha.nezha_jump import NezhaJumpCfg, NezhaJumpCfgPPO
from legged_gym.envs.nezha.nezha_jumpformer import NezhaJumpFormerCfg, NezhaJumpFormerCfgPPO
from legged_gym.envs.nezha.nezha_jumpformer.mdp import prediction_state
from legged_gym.managers import ObservationManager, ManagerTermCfg, ObsGroup
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.rsl_rl_adapter import RslRlVecEnvAdapter
from rsl_rl.algorithms import PPO
from rsl_rl.algorithms.plugins.future_prediction_plugin import build_future_targets, FuturePredictionPlugin
from rsl_rl.models.future_attention_model import FutureAttentionModel

torch.set_num_threads(1)


def sample_obs(n=4):
    return TensorDict({"actor": torch.randn(n, 590), "critic": torch.randn(n, 249),
                       "history": torch.randn(n, 590), "future_state": torch.randn(n, 41)},
                      batch_size=[n])


def build_model(obs):
    cfg = copy.deepcopy(NezhaJumpFormerCfgPPO.actor["backbone"])
    cfg.pop("class_name")
    return FutureAttentionModel(obs, {"actor": ["actor"]}, "actor", 16, **cfg)


class FutureTargets(unittest.TestCase):
    def test_two_steps_time_alignment_and_episode_boundaries(self):
        states = torch.arange(5.).reshape(5, 1, 1).expand(-1, 2, -1).clone()
        states[:, 1] += 100
        last = torch.tensor([[5.], [105.]])
        dones = torch.zeros(5, 2, 1)
        dones[1, 0] = 1
        dones[4, 1] = 1  # timeout is equally a sequence boundary
        targets, valid = build_future_targets(states, last, dones, 2)
        self.assertEqual(valid[:, 0].tolist(), [[True,False],[False,False],[True,True],[True,True],[True,False]])
        self.assertEqual(valid[:, 1].tolist(), [[True,True],[True,True],[True,True],[True,False],[False,False]])
        self.assertEqual(targets[3, 0, :, 0].tolist(), [4., 5.])
        self.assertEqual(targets[4, 0, 0, 0].item(), 5.)
        self.assertEqual(targets[0, 1, :, 0].tolist(), [101., 102.])
        self.assertTrue((targets[~valid] == 0).all())

    def test_all_reset_and_invalid_nan_targets_are_excluded(self):
        targets, valid = build_future_targets(torch.full((3,2,4),float('nan')),
                                             torch.full((2,4),float('nan')), torch.ones(3,2), 2)
        self.assertFalse(valid.any())
        self.assertTrue(torch.isfinite(targets).all())
        with self.assertRaises(ValueError):
            build_future_targets(torch.zeros(2,1,1), torch.zeros(1,1), torch.zeros(2,1), 3)


class PredictionLoss(unittest.TestCase):
    def test_invalid_labels_have_zero_gradient_and_advantage_weight_is_applied(self):
        prediction=torch.tensor([[[3.],[4.]],[[9.],[9.]]],requires_grad=True)
        obs=TensorDict({FuturePredictionPlugin.TARGET_KEY:torch.zeros(2,2,1),
                        FuturePredictionPlugin.MASK_KEY:torch.tensor([[True,False],[False,False]])},
                       batch_size=[2])
        plugin=FuturePredictionPlugin(loss_coef=.5,negative_advantage_weight=2.)
        plugin._squared_error=torch.zeros(2);plugin._valid_count=torch.zeros(2)
        ppo=NS(actor=NS(backbone=NS(predict_future=lambda observations:prediction)))
        batch=NS(observations=obs,advantages=torch.tensor([[-1.],[1.]]))
        loss=plugin.on_per_batch_extra_loss(ppo,batch)['future_prediction']
        self.assertEqual(loss.item(),9.)
        loss.backward()
        torch.testing.assert_close(prediction.grad,torch.tensor([[[6.],[0.]],[[0.],[0.]]]))


class AttentionPolicy(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.obs = sample_obs()
        self.model = build_model(self.obs)

    def test_attention_backends_preserve_outputs_gradients_and_checkpoint_keys(self):
        reference = copy.deepcopy(self.model)
        for block in reference.core.blocks:
            block.attention_backend = 'auto'
        for block in self.model.core.blocks:
            block.attention_backend = 'math'
        reference.load_state_dict(self.model.state_dict(), strict=True)
        self.assertEqual(reference.state_dict().keys(), self.model.state_dict().keys())
        inputs = []
        outputs = []
        for model in (self.model, reference):
            obs = self.obs.clone()
            obs['actor'].requires_grad_(True)
            prediction = model(obs)
            loss = prediction['actions'].square().mean() + prediction['predicted_future'].square().mean()
            loss.backward()
            inputs.append(obs['actor'].grad)
            outputs.append(prediction)
        for key in ('actions', 'predicted_future'):
            torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(inputs[0], inputs[1], rtol=1e-4, atol=1e-7)
        for (name, parameter), (_, baseline) in zip(self.model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(parameter.grad, baseline.grad, rtol=1e-4, atol=1e-7,
                                       msg=lambda detail: name + ': ' + detail)

    def test_single_actor_group_does_not_copy_history(self):
        self.assertIs(self.model._history(self.obs), self.obs['actor'])

    def test_rollout_inference_history_remains_usable_with_grad_enabled(self):
        with torch.inference_mode():
            obs = self.obs.clone()
        self.assertTrue(torch.is_inference(obs['actor']))
        actions = self.model(obs)['actions']
        actions.square().mean().backward()
        self.assertIsNotNone(self.model.core.token_projection.weight.grad)

    def test_no_privileged_or_future_label_leakage(self):
        a = self.model(self.obs)
        changed = self.obs.clone()
        changed['critic'].fill_(1e6)
        changed['future_state'].fill_(float('nan'))
        changed['history'].fill_(-1e6)
        changed['_future_prediction_targets'] = torch.full((4,2,41),1e9)
        b = self.model(changed)
        c = self.model(TensorDict({'actor':self.obs['actor']}, batch_size=[4]))
        torch.testing.assert_close(a['actions'],b['actions'],rtol=0,atol=0)
        torch.testing.assert_close(a['actions'],c['actions'],rtol=0,atol=0)
        self.assertEqual(tuple(a['predicted_future'].shape),(4,2,41))

    def test_predictions_are_actor_inputs_and_receive_action_gradients(self):
        inputs=[]
        hook=self.model.core.action_head.register_forward_pre_hook(lambda module,args: inputs.append(args[0]))
        output=self.model(self.obs)
        hook.remove()
        torch.testing.assert_close(inputs[0][:,:59],self.obs['actor'][:,-59:])
        torch.testing.assert_close(inputs[0][:,-82:],output['predicted_future'].flatten(1))
        output['actions'].square().sum().backward()
        grad=self.model.core.prediction_head[-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(),0)
        old=output['actions'].detach()
        with torch.no_grad():self.model.core.prediction_head[-1].bias.add_(1)
        self.assertGreater((self.model(self.obs)['actions']-old).abs().max().item(),1e-5)

    def test_history_order_and_deterministic_jit_inference(self):
        expected=self.model(self.obs)['actions']
        repeat=self.model(self.obs)['actions']
        torch.testing.assert_close(expected,repeat,rtol=0,atol=0)
        changed=self.obs.clone()
        changed['actor']=changed['actor'].reshape(4,10,59).flip(1).flatten(1)
        self.assertGreater((self.model(changed)['actions']-expected).abs().max().item(),1e-5)
        exported=self.model.as_jit()
        self.assertTrue(all(not p.requires_grad for p in exported.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
        scripted=torch.jit.script(exported)
        torch.testing.assert_close(scripted(self.obs['actor']),expected)
        with torch.inference_mode():inference_history=self.obs['actor'].clone()
        torch.testing.assert_close(scripted(inference_history),expected)
        stream=io.BytesIO();torch.jit.save(scripted,stream);stream.seek(0)
        loaded=torch.jit.load(stream)
        with torch.inference_mode():loaded_actions=loaded(inference_history)
        torch.testing.assert_close(loaded_actions,expected)

    def test_supervised_prediction_can_fit_a_small_controlled_batch(self):
        target=self.obs['actor'][:,-41:].unsqueeze(1).expand(-1,2,-1).detach()
        optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-3)
        initial=(self.model.predict_future(self.obs)-target).square().mean().item()
        for _ in range(25):
            loss=(self.model.predict_future(self.obs)-target).square().mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
        final=(self.model.predict_future(self.obs)-target).square().mean().item()
        self.assertLess(final,initial*0.2)


class ManagerIntegration(unittest.TestCase):
    def test_optional_group_history_reset_and_adapter_isolation(self):
        class Groups:
            class actor(ObsGroup):
                history_length=3
                frame=ManagerTermCfg(func=lambda env:env.frame,env_arg=True)
            class critic(ObsGroup):
                clean=ManagerTermCfg(func=lambda env:env.frame+10,env_arg=True)
            class future_state(ObsGroup):
                clean=ManagerTermCfg(func=lambda env:env.frame+20,env_arg=True)
        env=NS(num_envs=2,device='cpu',frame=torch.tensor([[1.],[2.]]),cfg=NS(noise=NS(add_noise=False)))
        manager=ObservationManager(env,Groups());env.observation_manager=manager
        manager.compute();env.frame+=1;manager.compute()
        self.assertEqual(env.obs_buf.tolist(),[[0.,1.,2.],[0.,2.,3.]])
        manager.reset(torch.tensor([0]));env.frame+=1;manager.compute()
        self.assertEqual(env.obs_buf.tolist(),[[0.,0.,3.],[2.,3.,4.]])
        self.assertEqual(env.obs_extra_bufs['future_state'].tolist(),[[23.],[24.]])
        adapter=RslRlVecEnvAdapter.__new__(RslRlVecEnvAdapter)
        adapter.env=env;adapter.num_envs=2;adapter.device='cpu'
        obs=adapter._make_obs_tensordict(env.obs_buf,env.privileged_obs_buf,env.privileged_obs_buf,env.obs_buf)
        self.assertEqual(obs['actor'].shape[-1],3)
        self.assertEqual(obs['future_state'].shape[-1],1)
        env.obs_extra_bufs={}
        old=adapter._make_obs_tensordict(env.obs_buf,env.privileged_obs_buf,env.privileged_obs_buf,env.obs_buf)
        self.assertNotIn('future_state',old.keys())

    def test_clean_state_scaling_and_wheel_angle_invariance(self):
        env=NS(base_quat=torch.tensor([[0.,0.,0.,1.]]),gravity_vec=torch.tensor([[0.,0.,-1.]]),
               dof_pos=torch.arange(16.).reshape(1,16),default_dof_pos=torch.ones(1,16),
               dof_vel=torch.ones(1,16)*10,base_lin_vel=torch.ones(1,3),base_ang_vel=torch.ones(1,3)*2,
               wheel_indices=torch.tensor([3,7,11,15]),obs_scales=NS(dof_pos=1.,dof_vel=.05,lin_vel=2.,ang_vel=.25))
        state=prediction_state(env)
        self.assertEqual(tuple(state.shape),(1,41))
        torch.testing.assert_close(state[:,:3],torch.tensor([[0.,0.,-1.]]))
        self.assertTrue((state[:,3+env.wheel_indices]==0).all())
        env.dof_pos[:,env.wheel_indices]+=1000
        torch.testing.assert_close(state,prediction_state(env))
        self.assertTrue((state[:,19:35]==.5).all())

    def test_task_config_preserves_original_mdp(self):
        old=NezhaJumpCfg();new=NezhaJumpFormerCfg()
        self.assertEqual(class_to_dict(old.rewards_manager),class_to_dict(new.rewards_manager))
        self.assertEqual(class_to_dict(old.control),class_to_dict(new.control))
        self.assertEqual(class_to_dict(old.terminations),class_to_dict(new.terminations))
        self.assertEqual(class_to_dict(old.commands),class_to_dict(new.commands))
        self.assertEqual(new.env.num_observations,590)
        self.assertEqual(new.env.num_privileged_obs,249)
        self.assertFalse(hasattr(old.observations,'future_state'))
        self.assertEqual(NezhaJumpCfgPPO.actor['backbone']['class_name'],'rsl_rl.models:MLPModel')


class PPOIntegration(unittest.TestCase):
    def test_shared_prediction_matches_recompute_with_either_plugin_order(self):
        for reverse_plugins in (False, True):
            with self.subTest(reverse_plugins=reverse_plugins):
                torch.manual_seed(31)
                obs = sample_obs()
                cfg = class_to_dict(NezhaJumpFormerCfgPPO())
                cfg.update(num_steps_per_env=4, multi_gpu=None)
                cfg['algorithm'].update(num_learning_epochs=1, num_mini_batches=2)
                env = NS(num_envs=4, num_actions=16)
                with redirect_stdout(io.StringIO()):
                    shared = PPO.construct_algorithm(obs, env, cfg, 'cpu')
                    reference = PPO.construct_algorithm(obs, env, cfg, 'cpu')
                reference.load(copy.deepcopy(shared.save()), load_cfg=None, strict=True)
                with torch.inference_mode():
                    for step in range(4):
                        nxt = sample_obs()
                        dones = torch.tensor([step == 1, False, False, step == 3])
                        for alg in (shared, reference):
                            torch.manual_seed(100 + step)
                            alg.act(obs)
                            reward = torch.ones(4)
                            for plugin in alg.plugins:
                                reward = plugin.on_after_step(None, nxt, reward, dones, {})
                            alg.process_env_step(nxt, reward, dones, {})
                        obs = nxt
                    for alg in (shared, reference):
                        alg.compute_returns(obs)

                # Reference uses exactly the pre-optimization recomputation.
                plugin = next(p for p in reference.plugins if isinstance(p, FuturePredictionPlugin))
                original_hook = plugin.on_per_batch_extra_loss
                def recompute(ppo, batch):
                    extra = batch.actor_extra
                    batch.actor_extra = None
                    try:
                        return original_hook(ppo, batch)
                    finally:
                        batch.actor_extra = extra
                plugin.on_per_batch_extra_loss = recompute
                if reverse_plugins:
                    shared.plugins.reverse()
                    reference.plugins.reverse()

                # Retain batches only in this test to check the graph lifetime.
                seen_batches = []
                generator = shared.storage.mini_batch_generator
                def capture_batches(*args):
                    for batch in generator(*args):
                        seen_batches.append(batch)
                        yield batch
                shared.storage.mini_batch_generator = capture_batches
                counts = [0, 0]
                def count_shared(module, args):
                    counts[0] += 1
                def count_reference(module, args):
                    counts[1] += 1
                hooks = [shared.actor.backbone.core.token_projection.register_forward_pre_hook(count_shared),
                         reference.actor.backbone.core.token_projection.register_forward_pre_hook(count_reference)]
                try:
                    torch.manual_seed(200)
                    actual = shared.update()
                    torch.manual_seed(200)
                    expected = reference.update()
                finally:
                    for hook in hooks:
                        hook.remove()
                self.assertEqual(counts, [4, 6])  # 2 minibatches: PPO + symmetry vs + recompute
                self.assertTrue(all(batch.actor_extra is None for batch in seen_batches))
                self.assertEqual(actual.keys(), expected.keys())
                for key in actual:
                    torch.testing.assert_close(torch.tensor(actual[key]), torch.tensor(expected[key]),
                                               rtol=1e-5, atol=1e-7, msg=key)
                for name in ('actor', 'critic'):
                    for (key, parameter), (_, baseline) in zip(getattr(shared, name).named_parameters(),
                                                              getattr(reference, name).named_parameters()):
                        torch.testing.assert_close(parameter.grad, baseline.grad, rtol=1e-4, atol=1e-7, msg=key)
                        # Sharing changes FP32 gradient summation order; Adam can
                        # amplify roundoff on theoretically zero attention-key biases.
                        torch.testing.assert_close(parameter, baseline, rtol=1e-5, atol=1e-6,
                                                   msg=lambda detail: key + ': ' + detail)
                with torch.no_grad():
                    torch.testing.assert_close(shared.actor(obs)['actions'], reference.actor(obs)['actions'],
                                               rtol=1e-5, atol=1e-6)

    def test_original_mlp_task_update_remains_supported(self):
        obs = sample_obs()
        cfg = class_to_dict(NezhaJumpCfgPPO())
        cfg.update(num_steps_per_env=2, multi_gpu=None)
        cfg['algorithm'].update(num_learning_epochs=1, num_mini_batches=1)
        with redirect_stdout(io.StringIO()):
            alg = PPO.construct_algorithm(obs, NS(num_envs=4, num_actions=16), cfg, 'cpu')
        with torch.inference_mode():
            for step in range(2):
                alg.act(obs)
                obs = sample_obs()
                alg.process_env_step(obs, torch.ones(4), torch.zeros(4, dtype=torch.bool), {})
            alg.compute_returns(obs)
        metrics = alg.update()
        self.assertIn('symmetry', metrics)
        self.assertNotIn('future_prediction', metrics)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    def test_two_updates_symmetry_checkpoint_and_metadata(self):
        torch.manual_seed(11)
        obs=sample_obs()
        cfg=class_to_dict(NezhaJumpFormerCfgPPO())
        cfg.update(num_steps_per_env=4,multi_gpu=None)
        cfg['algorithm'].update(num_learning_epochs=1,num_mini_batches=2)
        env=NS(num_envs=4,num_actions=16)
        with redirect_stdout(io.StringIO()):alg=PPO.construct_algorithm(obs,env,cfg,'cpu')
        before=alg.actor.backbone.core.prediction_head[-1].weight.detach().clone()
        for iteration in range(2):
            with torch.inference_mode():
                for step in range(4):
                    alg.act(obs)
                    nxt=sample_obs();dones=torch.tensor([step==1,False,False,step==3])
                    reward=torch.ones(4)
                    for plugin in alg.plugins:reward=plugin.on_after_step(None,nxt,reward,dones,{})
                    alg.process_env_step(nxt,reward,dones,{})
                    obs=nxt
                alg.compute_returns(obs)
            metrics=alg.update()
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
            self.assertIn('symmetry',metrics)
            self.assertIn('future_prediction',metrics)
            self.assertNotIn(FuturePredictionPlugin.TARGET_KEY,alg.storage.observations.keys())
            self.assertEqual(alg.storage.step,0)
        self.assertGreater((before-alg.actor.backbone.core.prediction_head[-1].weight).abs().sum().item(),0)
        saved=alg.save()
        with redirect_stdout(io.StringIO()):restored=PPO.construct_algorithm(obs,env,cfg,'cpu')
        restored.load(saved, load_cfg=None, strict=True)
        torch.testing.assert_close(alg.actor(obs)['actions'],restored.actor(obs)['actions'])
        saved['future_prediction_metadata']['future_steps']=3
        with self.assertRaises(ValueError):restored.plugins[0].on_load(restored,saved)


if __name__=='__main__':
    unittest.main()
