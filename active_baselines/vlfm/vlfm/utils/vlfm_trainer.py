# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
)
from habitat_baselines.rl.ddppo.algo import DDPPO  # noqa: F401.
from habitat_baselines.rl.ppo.single_agent_access_mgr import (  # noqa: F401.
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    extract_scalars_from_info as extract_scalars_from_info_habitat,
)
from omegaconf import OmegaConf

#TODO: Added for gui
#from habitat import Env
import cv2
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations import maps

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"


def extract_scalars_from_info(info: Dict[str, Any]) -> Dict[str, float]:
    info_filtered = {k: v for k, v in info.items() if not isinstance(v, list)}
    return extract_scalars_from_info_habitat(info_filtered)

#TODO: Added for embed'
def get_start_pos_rot(vec_envs):

    pose_info = {}

    for i in range(vec_envs.num_envs):

        info = vec_envs.call_at(index = i,
                                function_name="current_episode",
                                function_args={"all_info":True})
        
        pose_info[i] = {"init_pos_abs" : info.start_position, 
                        "init_rot_abs" : info.start_rotation}
    
    return pose_info

#Get current habitat position for each env
def get_curr_hab_pos(vec_envs):

    curr_hab_pos = {}
    for i in range(vec_envs.num_envs):

        agent_state = vec_envs.call_at(index = i,
                                       function_name = "get_curr_state")
        curr_hab_pos[i] = agent_state.position
    print(f" Current position: {agent_state.position}\n Current Rotation: {agent_state.rotation}")
    return curr_hab_pos


@baseline_registry.register_trainer(name="vlfm")
class VLFMTrainer(PPOTrainer):
    envs: VectorEnv

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Some configurations require not to load the checkpoint, like when using
        # a hierarchial policy
        if self.config.habitat_baselines.eval.should_load_ckpt:
            # map_location="cpu" is almost always better than mapping to a CUDA device.
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            step_id = ckpt_dict["extra_state"]["step"]
            print(step_id)
        else:
            ckpt_dict = {"config": None}

        config = self._get_resume_state_config_or_new_config(ckpt_dict["config"])

        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split


        if len(self.config.habitat_baselines.eval.video_option) > 0:
            agent_config = get_agent_config(config.habitat.simulator)
            agent_sensors = agent_config.sim_sensors
            extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
            with read_write(agent_sensors):
                agent_sensors.update(extra_sensors)
            with read_write(config):
                if config.habitat.gym.obs_keys is not None:
                    for render_view in extra_sensors.values():
                        if render_view.uuid not in config.habitat.gym.obs_keys:
                            config.habitat.gym.obs_keys.append(render_view.uuid)
                config.habitat.simulator.debug_render = True

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        curr_dir = os.getcwd()

        #Check if results already exist
        scene_name = config.habitat.dataset.content_scenes[0]
        model_name = config.habitat_baselines.rl.policy.embed_model_name
        embed_root_dir = config.grid_embed.embed_log_dir

        print(f"\n\nScene Name: {scene_name}")
        print(f"Model Name: {model_name}")
        print(f"Embed Root Dir: {embed_root_dir}")

        init_dir = os.path.join(os.getcwd(), embed_root_dir, "init_dicts")
        init_dict_name = f"init_dict_model_{model_name}_scene_{scene_name}.npz"
        init_dict_path = os.path.join(init_dir, init_dict_name)

        print(f"Init Dict Path: {init_dict_path}\n\n")

        assert not os.path.exists(init_dict_path), f"Init Dict already exists at path: {init_dict_path}!"

        #Change directory to initialize env using data in habitat-lab
        hab_dir = os.path.join(curr_dir, 'habitat-lab')
        print(f"Chaning directory to : {hab_dir}")
        os.chdir(hab_dir)

        self._init_envs(config, is_eval=True)

        print(f"Changing back directory to: {curr_dir}")
        os.chdir(curr_dir)

        #TODO: Checkpoint - habitat_baselines -> rl -> ppo -> single_agent_access_mgr.py
        #TODO: Imp -> Initializes the policy (the configs are loaded using the from_config method inside HabitatMixin)
        self._agent = self._create_agent(None) 
        action_shape, discrete_actions = get_action_space_info(self._agent.policy_action_space)

        if self._agent.actor_critic.should_load_agent_state:
            self._agent.load_state_dict(ckpt_dict)

        print("resetting envs")
        #TODO: Imp -> Initializes the observations
        observations = self.envs.reset()

        #TODO: Added for embed (Tracks agent height for determining floors)
        # curr_hab_pos = get_curr_hab_pos(self.envs)
        # observations[0]['agent_height'] = curr_hab_pos[0][1]
        observations[0]['agent_height'] = 0.2
        # print(f" Agent Height: {curr_hab_pos[0][1]}")
        # print(f'obs keys: {observations[0].keys()}')

        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        #TODO: Added for embed
        #Obtains initial position and rotation coordinates
        init_pose_info = get_start_pos_rot(self.envs)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        #TODO: Replaced for making compatible with hab 2.5
        # test_recurrent_hidden_states = torch.zeros(
        #     (
        #         self.config.habitat_baselines.num_environments,
        #         *self._agent.hidden_state_shape,
        #     ),
        #     device=self.device,
        # )
        test_recurrent_hidden_states = torch.zeros(
            (
                self.config.habitat_baselines.num_environments,
                *(0, 512),
            ),
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[Any, Any] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        rgb_frames: List[List[np.ndarray]] = [[] for _ in range(self.config.habitat_baselines.num_environments)]
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(self.config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = self.config.habitat_baselines.test_episode_count
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes, dataset only has {{total_num_eps}}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert number_of_eval_episodes > 0, "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        self._agent.eval()

        from vlfm.utils.habitat_visualizer import HabitatVis

        num_successes = 0
        num_total = 0
        hab_vis = HabitatVis()      #TODO: Imp -> Visualization

        #TODO: Added for gui
        if config.grid_embed.gui_nav:
            print(f"\nNavigation using GUI. \nIf you want to navigate using saved path, please set gui_nav to False.")
            step_actions = {}
            curr_ep_info = self.envs.current_episodes()
            scene_name = os.path.basename(curr_ep_info[0].scene_id).split(".")[0]
            episode_id = curr_ep_info[0].episode_id

            file_path = os.path.join(config.grid_embed.embed_log_dir, f'step_actions/step_actions_scene_{scene_name}_ep_{episode_id}.npz')
            is_file_present = os.path.exists(file_path)

            assert not is_file_present, f"There is already a saved navigation file at {config.grid_embed.embed_log_dir}. Either change saving directory or choose another scene or episode."

            step_actions['init_pos_abs'] = init_pose_info[0]['init_pos_abs']
            step_actions['init_rot_abs'] = init_pose_info[0]['init_rot_abs']
            step_actions['scene'] = scene_name
            step_actions['episode'] = episode_id

        elif config.grid_embed.saved_nav:
            print(f"\nNavigation using Saved Path")
            
            curr_ep_info = self.envs.current_episodes()
            scene_name = os.path.basename(curr_ep_info[0].scene_id).split(".")[0]
            episode_id = curr_ep_info[0].episode_id
            saved_nav_scene_path = os.path.join(config.grid_embed.embed_log_dir, f'step_actions/step_actions_scene_{scene_name}_ep_{episode_id}.npz')

            if not os.path.exists(saved_nav_scene_path):
                saved_nav_scene_path = os.path.join(config.grid_embed.embed_log_dir, f'step_actions/scene_{scene_name}_step_actions.npz')

            assert os.path.exists(saved_nav_scene_path), f"Saved path for scene {scene_name} : {saved_nav_scene_path} does not exist! Please provide already saved scene."
            saved_nav_scene = dict(np.load(saved_nav_scene_path, allow_pickle=True))

            #Check if the initial position of episode is same as saved episode for scene
            try:
                assert (saved_nav_scene['init_pos_abs'] == init_pose_info[0]['init_pos_abs']).all() and (saved_nav_scene['init_rot_abs'] == init_pose_info[0]['init_rot_abs']).all(), \
                    f"Initial Position and Rotation are not same as in saved file. Please make sure you are using scene: {saved_nav_scene['scene']} and episode: {saved_nav_scene['episode']}" 
                
                del saved_nav_scene['init_pos_abs']
                del saved_nav_scene['init_rot_abs']
                del saved_nav_scene['scene']
                del saved_nav_scene['episode']
            except:
                print(f"\n\n!!--- Check for Initial Position and Rotation FAILED! Resuming Run...")
                pass

            step_actions = {int(k):torch.Tensor(v).to(dtype=torch.int64) for (k, v) in saved_nav_scene.items()}

        gui_info = {}       #Passing relevant info to act method
        while len(stats_episodes) < (number_of_eval_episodes * evals_per_ep) and self.envs.num_envs > 0:
            
            #Contains info about scene and episode number (nothing else)
            current_episodes_info = self.envs.current_episodes()

            scene_name = os.path.basename(current_episodes_info[0].scene_id).split(".")[0]
            gui_info["scene_name"] = scene_name

            with inference_mode():

                #TODO: Added for gui
                if not config.grid_embed.gui_nav:

                    #TODO: Imp -> Policy Action (See Habitat, ITM, ObjectNav, Base Policies)
                    action_data = self._agent.actor_critic.act(     #The output action_data should contain the policy_info
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=False,
                        gui_info = gui_info
                    )


                #TODO: Added for gui
                if config.grid_embed.gui_nav:
                    print(f"\nStep: {self._agent.actor_critic._num_steps}")
                    self._agent.actor_critic._num_steps += 1
                    if self._agent.actor_critic._num_steps == 1:
                        plot_frame = batch["rgb"][0].cpu().numpy().astype(np.uint8)[:, :, ::-1]                
                    else:
                        rgb = batch["rgb"][0].cpu().numpy().astype(np.uint8)[:, :, ::-1] 
                        top_down = maps.colorize_draw_agent_and_fit_to_height(infos[0]["top_down_map"], rgb.shape[0])
                        plot_frame = np.hstack((rgb, top_down))

                    cv2.imshow(f"Scene {os.path.basename(current_episodes_info[0].scene_id).split('.')[0]}", \
                            plot_frame)
                        
                    keystroke = cv2.waitKey(0)
                    while keystroke not in [ord(FORWARD_KEY), ord(LEFT_KEY), ord(RIGHT_KEY), ord(FINISH)]:
                        print(f"INVALID KEY, Please try again.\n Valid Keys: [Forward: {FORWARD_KEY}, Left: {LEFT_KEY}, Right: {RIGHT_KEY}, Stop: {FINISH}]")
                        keystroke = cv2.waitKey(0)
                
                    if keystroke == ord(FORWARD_KEY):
                        gui_action = HabitatSimActions.move_forward
                        print("\nGUI action: FORWARD")
                    elif keystroke == ord(LEFT_KEY):
                        gui_action = HabitatSimActions.turn_left
                        print("\nGUI action: LEFT")
                    elif keystroke == ord(RIGHT_KEY):
                        gui_action = HabitatSimActions.turn_right
                        print("\nGUI action: RIGHT")
                    elif keystroke == ord(FINISH):
                        gui_action = HabitatSimActions.stop
                        print("\nGUI action: FINISH")

                        print("Stopping Run...")
                        n_steps_done = len(stats_episodes)
                        remaining_steps = (number_of_eval_episodes * evals_per_ep) - n_steps_done
                        for i in range(remaining_steps):
                            stats_episodes[i] = {"reward": 0}
                    
                    gui_action_t = torch.Tensor([gui_action]).unsqueeze(0).to(dtype=torch.int64)

                    #TODO: Added for gui
                    # action_data.actions = gui_action_t
                    step_actions[int(self._agent.actor_critic._num_steps - 1)] = gui_action_t

                elif config.grid_embed.saved_nav:
                    curr_step = int(self._agent.actor_critic._num_steps - 1)
                    action_data.actions = step_actions[curr_step]
                    print(f"Step: {curr_step}, Saved Action: {step_actions[curr_step]}")

                    if step_actions[curr_step][0][0] == HabitatSimActions.stop:
                        print("Stopping Run...")
                        n_steps_done = len(stats_episodes)
                        remaining_steps = (number_of_eval_episodes * evals_per_ep) - n_steps_done
                        for i in range(remaining_steps):
                            stats_episodes[i] = {"reward": 0}


                if not config.grid_embed.gui_nav:
                    if "VLFM_RECORD_ACTIONS_DIR" in os.environ:
                        action_id = action_data.actions.cpu()[0].item()
                        filepath = os.path.join(
                            os.environ["VLFM_RECORD_ACTIONS_DIR"],
                            "actions.txt",
                        )
                        # If the file doesn't exist, create it
                        if not os.path.exists(filepath):
                            open(filepath, "w").close()
                        with open(filepath, "a") as f:
                            f.write(f"{action_id}\n")

                    if action_data.should_inserts is None:
                        test_recurrent_hidden_states = action_data.rnn_hidden_states
                        prev_actions.copy_(action_data.actions)  # type: ignore
                    else:
                        for i, should_insert in enumerate(action_data.should_inserts):
                            if should_insert.item():
                                test_recurrent_hidden_states[i] = action_data.rnn_hidden_states[i]
                                prev_actions[i].copy_(action_data.actions[i])  # type: ignore
            
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if not config.grid_embed.gui_nav:
                if is_continuous_action_space(self._env_spec.action_space):
                    # Clipping actions to the specified limits
                    step_data = [
                        np.clip(
                            a.numpy(),
                            self._env_spec.action_space.low,
                            self._env_spec.action_space.high,
                        )
                        for a in action_data.env_actions.cpu()
                    ]
                else:
                    step_data = [a.item() for a in action_data.env_actions.cpu()]
            else:
                step_data = [a.item() for a in gui_action_t]

            #TODO: Imp -> Takes step in env, updates infos using the new policy_infos
            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]

            if not config.grid_embed.gui_nav:
                policy_infos = self._agent.actor_critic.get_extra(action_data, infos, dones)
                for i in range(len(policy_infos)):
                    infos[i].update(policy_infos[i])    #Updates all info keys present in policy_info keys

            #TODO: Added for embed (Tracks agent height for determining floors)
            # curr_hab_pos = get_curr_hab_pos(self.envs)
            # observations[0]['agent_height'] = curr_hab_pos[0][1]
            observations[0]['agent_height'] = 0.2

            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)
                elif int(next_episodes_info[i].episode_id) == 123123123:
                    envs_to_pause.append(i)


                #TODO: Imp -> Collects data for visualization
                if not config.grid_embed.gui_nav:
                    if len(self.config.habitat_baselines.eval.video_option) > 0:
                        hab_vis.collect_data(batch, infos, action_data.policy_info)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if episode_stats["success"] == 1:
                        num_successes += 1
                    num_total += 1
                    print(f"Success rate: {num_successes / num_total * 100:.2f}% ({num_successes} out of {num_total})")

                    from vlfm.utils.episode_stats_logger import (
                        log_episode_stats,
                    )

                    #TODO: Imp -> Logs Infos as JSON after removing all np arrays from values
                    # try:
                    if not config.grid_embed.gui_nav:
                        # infos[i]["init_dict"]["init_pose"] = init_pose_info[i]
                        # infos[i]["embed_model_name"] = self._agent.actor_critic._obstacle_map.embed_model_name
                        failure_cause = log_episode_stats(
                            current_episodes_info[i].episode_id,
                            current_episodes_info[i].scene_id,
                            infos[i],
                            config.grid_embed.embed_log_dir
                        )
                    # except Exception as e:
                    #     print('Logging Unsuccessful due to:', e)
                    #     failure_cause = "Unknown"

                    #TODO: Added for gui
                    if config.grid_embed.gui_nav:
                        if len(step_actions) > 0:
                            scene_path = current_episodes_info[i].scene_id
                            scene_id = os.path.basename(scene_path).split(".")[0]
                            episode_id = current_episodes_info[i].episode_id

                            gui_log_root_dir = os.path.join(config.grid_embed.embed_log_dir, 'step_actions')
                            file_path = os.path.join(gui_log_root_dir, f"step_actions_scene_{scene_id}_ep_{episode_id}.npz")

                            os.makedirs(gui_log_root_dir, exist_ok=True)
                            
                            if not (os.path.exists(file_path) and os.path.getsize(file_path) > 0):
                                print(f"Logging traced step-actions for Scene: {scene_id} to {file_path}")
                                np.savez(file_path, **{str(k):v for (k, v) in step_actions.items()})
                            else:
                                print(f"Logging step-actions FAILED, as logged file already exists.")
                        else:
                            print("\nNot saving traced GUI steps and actions. step_actions dictionary is empty.")


                    #TODO: Imp -> Saves all the visualized frames into a video
                    elif len(self.config.habitat_baselines.eval.video_option) > 0:
                        rgb_frames[i] = hab_vis.flush_frames(failure_cause)

                        #TODO: Changed for vlfm
                        from PIL import Image
                        images = np.array(rgb_frames[i])                        
                        scene_name = os.path.basename(current_episodes_info[i].scene_id).split(".")[0]
                        embed_model_name = self._agent.actor_critic._obstacle_map.embed_model_name

                        gif_root_dir = os.path.join(config.grid_embed.embed_log_dir, 'gifs')
                        if not os.path.exists(gif_root_dir): os.makedirs(gif_root_dir, exist_ok=True)
                        gif_file_dir = os.path.join(gif_root_dir, f'scene_{scene_name}_ep_{current_episodes_info[i].episode_id}.gif')
                        final_file_dir = os.path.join(gif_root_dir, f'model_{embed_model_name}_scene_{scene_name}_ep_{current_episodes_info[i].episode_id}.png')
                        
                        # images = list(map(Image.fromarray, images))
                        # images[0].save(gif_file_dir, save_all=True, append_images=images[1:100], duration=50, loop=0)
                        # print(f'Created gif at : {gif_file_dir}')
                        
                        final_frame = Image.fromarray(images[-1])
                        final_frame.save(final_file_dir, format="PNG")
                        print(f"Created Last Frame Image at: {final_file_dir}")
                        


                        # generate_video(
                        #     video_option=self.config.habitat_baselines.eval.video_option,
                        #     video_dir=self.config.habitat_baselines.video_dir,
                        #     images=rgb_frames[i],
                        #     episode_id=current_episodes_info[i].episode_id,
                        #     checkpoint_idx=checkpoint_index,
                        #     metrics=extract_scalars_from_info(infos[i]),
                        #     fps=self.config.habitat_baselines.video_fps,
                        #     tb_writer=writer,
                        #     keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                        # )

                        rgb_frames[i] = []

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            self.config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()

        if "ZSOS_DONE_PATH" in os.environ:
            # Create an empty file at ZSOS_DONE_PATH to signal that the
            # evaluation is done
            done_path = os.environ["ZSOS_DONE_PATH"]
            with open(done_path, "w") as f:
                f.write("")

        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = np.mean([v[stat_key] for v in stats_episodes.values()])

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar("eval_reward/average_reward", aggregated_stats["reward"], step_id)

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()
