# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
import shutil
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat_baselines import PPOTrainer
from habitat_baselines.rl.ppo.policy import PolicyActionData
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
from vlfm.utils.embed_utils import read_valid_scenes
from vlfm.utils.img_utils import mask_rgb_by_depth_generic
from vlfm.policy.habitat_policies import array_hash

from time import time

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
    print(f" Current Hab position: {agent_state.position}\n Current Hab Rotation: {agent_state.rotation}")
    return curr_hab_pos


@baseline_registry.register_trainer(name="vlfm_grid")
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

        os.environ["ZSOS_LOG_DIR"] = os.path.join(os.getcwd(), config.grid_embed.log_dir)
        os.makedirs(os.environ["ZSOS_LOG_DIR"], exist_ok=True)
        print(f"Setting Logging Directory to be: {os.environ['ZSOS_LOG_DIR']}")

        curr_dir = os.getcwd()

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

        #Checkpoint - habitat_baselines -> rl -> ppo -> single_agent_access_mgr.py
        #Initializes the policy (the configs are loaded using the from_config method inside HabitatMixin)
        self._agent = self._create_agent(None) 
        action_shape, discrete_actions = get_action_space_info(self._agent.policy_action_space)

        if self._agent.actor_critic.should_load_agent_state:
            self._agent.load_state_dict(ckpt_dict)

        #Initializes the observations
        #Infos is called in order to use top_down for gui_nav
        observations = self.envs.reset()
        infos = self.envs.call_at(index = 0,
                                  function_name = "get_info",
                                  function_args = {"observations": observations})  
        infos = [infos] 

        #TODO: Need to add floor detection
        # curr_hab_pos = get_curr_hab_pos(self.envs)
        # observations[0]['agent_height'] = curr_hab_pos[0][1]
        observations[0]['agent_height'] = 0.2

        print(observations)        

        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        #Masks RGB with Depth Thresholds
        if config.grid_embed.rgb_masking:
            rgb_batch = batch["rgb"]
            depth_batch = batch["depth"]
            max_depth_val = self.config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth
            depth_batch_unnorm = depth_batch * max_depth_val

            # import matplotlib.pyplot as plt
            # plt.imsave("/mnt/vlfm_query_embed/junk/rgb_og.png", rgb_batch[0].cpu().numpy())
            # plt.imsave("/mnt/vlfm_query_embed/junk/depth_og.png", depth_batch_unnorm[0, :, :, 0].cpu().numpy())
            
            rgb_masked = mask_rgb_by_depth_generic(rgb_batch, depth_batch_unnorm, max_depth_val, 0.6)
            batch["rgb"] = rgb_masked

            # print(rgb_batch[0].shape)
            # print(rgb_masked[0].shape)
            # print(depth_batch_unnorm[0].shape)
            # print(depth_batch.max(), depth_batch.min())
            # print(depth_batch_unnorm.max(), depth_batch_unnorm.min())
            # print(rgb_masked.max(), rgb_masked.min())

            # plt.imsave("/mnt/vlfm_query_embed/junk/rgb_masked.png", rgb_masked[0].cpu().numpy())

        #Obtains initial position and rotation coordinates
        init_pose_info = get_start_pos_rot(self.envs)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

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
        print(f"\n\nNumber of Eval Episodes: {number_of_eval_episodes}\nNumber of Evals per Episode: {evals_per_ep}\nNumber of Envs: {self.envs.num_envs}\nNumber of Episodes: {self.envs.number_of_episodes}\n\n")

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        self._agent.eval()

        from vlfm.utils.habitat_visualizer import HabitatVis

        num_successes = 0
        num_total = 0
        hab_vis = HabitatVis()    

        step_time_path = f"{config.grid_embed.log_dir}/timings/step_times.txt"
        episode_time_path = f"{config.grid_embed.log_dir}/timings/episode_times.txt"
        os.makedirs(os.path.dirname(step_time_path), exist_ok=True)
        start_episode_time = time()

        gui_info = {}
        gui_info["top_down"] =  infos[0]["top_down_map"]
        saved_img_dir = os.path.join("/mnt/vlfm_query_embed", self.config.habitat_baselines.rl.policy.scrape_data_dir)
        saved_objects = os.listdir(saved_img_dir)
        print(f"Number of scraped images: {len(saved_objects)}")
        # skip_episode = False

        #TODO ADDED: If success eps are saved, and need to be run again
        # successes_saved_dir = os.path.dirname(os.path.dirname(self.config.habitat.dataset.data_path))
        # eai_data_type = self.config.habitat.dataset.data_path.split("/")[-1].split(".json")[0]
        # successes_saved_path = os.path.join(successes_saved_dir, "successes", eai_data_type, "successes.txt")
        # successes_saved_path = os.path.join(self.config.grid_embed.log_dir, "successes.txt")
        # if os.path.exists(successes_saved_path):
        #     with open(successes_saved_path, "r") as f:
        #         success_eps_txt = f.readlines()

        #     success_eps = {}
        #     for k in success_eps_txt:
        #         k_scene, k_ep_num = k.split(",")[0], int(k.strip().split(",")[1]) 

        #         if k_scene in success_eps.keys():
        #             success_eps[k_scene].append(k_ep_num)
        #         else:
        #             success_eps[k_scene] = [k_ep_num]
        # else:
        #     success_eps = None
        
        while len(stats_episodes) < (number_of_eval_episodes * evals_per_ep) and self.envs.num_envs > 0:

            current_episodes_info = self.envs.current_episodes()
            curr_scene_name = os.path.basename(current_episodes_info[0].scene_id).split(".")[0]
            curr_episode_id = current_episodes_info[0].episode_id

            #If Episode was done, then reset the variables
            if not not_done_masks[0][0]:

                start_episode_time = time()

                print(f"\n\nCurrent Scene Name: {curr_scene_name}")
                print(f"Current Episode ID: {curr_episode_id}")
                
                if config.grid_embed.gui_nav:
                    init_pose_info = get_start_pos_rot(self.envs)
                    step_actions, scenes_done, skip_episode, gui_info = reset_gui_nav(embed_log_dir = config.grid_embed.embed_log_dir,
                                                                                        scene_name = curr_scene_name,
                                                                                        episode_id = curr_episode_id,
                                                                                        init_pose_info = init_pose_info,
                                                                                        only_valid_eps = config.grid_embed.only_valid_eps,
                                                                                        valid_eps_base_path = config.grid_embed.valid_eps_base_path,
                                                                                        valid_eps_split = config.grid_embed.valid_eps_split)

                    infos = self.envs.call_at(index = 0,
                                                function_name = "get_info",
                                                function_args = {"observations": observations})  
                    gui_info["top_down"] = infos["top_down_map"]

                elif config.grid_embed.saved_nav:
                    init_pose_info = get_start_pos_rot(self.envs)
                    step_actions, scenes_done, skip_episode = reset_saved_nav(embed_log_dir=config.grid_embed.embed_log_dir,
                                                scene_name = curr_scene_name,
                                                episode_id = curr_episode_id,
                                                init_pose_info = init_pose_info,
                                                with_saved_eps = True)
                    
                    self._agent.actor_critic.step_actions = step_actions
                
                else:
                    #If Logged file already exists, then skip the current episode.
                    log_file_name = f"{curr_episode_id}_{curr_scene_name}.json"        
                    log_file_path = os.path.join(os.environ["ZSOS_LOG_DIR"], log_file_name)
                    gui_info["scene_name"] = curr_scene_name

                    print(f"\nLog File path: ", log_file_path)
                    skip_episode = os.path.exists(log_file_path)

                    if skip_episode:
                        print(f"\n\n----Logged File already exists at: {log_file_path}.\nSkipping Episode {curr_episode_id} for Scene {curr_scene_name}...")
            
            start_step_time = time()

                        # HSSD-specific validation for scraped target images.
            invalid_cats = [
                'couch, Charlie Medium sofa bed tight covered in Subtle fabric Pillarbox Red',
                'couch, Alder Foam 3 Piece Sectional With Right Arm Facing Armless Chaise',
                'couch, Belgravia Leather Corner Sofa 1.5 seater x 1.5 seater with ottoman',
                'plant, Silk Plant - Areca Palm Tree - 36" in Coiled Whicker Pot - Nu-Dell',
                'couch, Watkins Power Reclining Loveseat With Power Headrest, Built-In Battery & USB, Linen',
                'bottle, Kugel Easy Thermal Jug Vacuum Flask 0.94 Litres - Satin Orange',
                'toiletry, MOLTON BROWN CASSIA ENERGY HAIR &amp; BODY WASH 200ML',
                'table_lamp, Alnilam Marble Table Lamp-46Cmh- White With Gold',
                'couch, STOCKHOLM 2017 Two seat sofa, Sandbacha Dark Grey',
                'coffee_maker, Delonghi Silky White Lattissima One Espresso Maker'
            ]

            ###TODO DEBUGGING: Make sure object category is present
            #For HSSD
            if config.habitat.dataset.data_path.__contains__("hssd") and (not config.grid_embed.saved_nav) and (not config.grid_embed.gui_nav):
                obj_id = batch["objectgoal"][0].item()
                obj_category = self._agent.actor_critic.HSSD_ID_TO_CAT[obj_id]

                print(f"Object ID: {obj_id}, Object Category: {obj_category}")

                if obj_category not in saved_objects: 
                    print(f"HSSD Object not in Saved Data Directory: {os.path.join(saved_img_dir, obj_category)}! Skipping...")
                    skip_episode = True

                elif len(os.listdir(os.path.join(saved_img_dir, obj_category))) < 15:
                    print("Saved images for HSSD object category is less than 15! Skipping...")
                    skip_episode = True

                if obj_category in invalid_cats:
                    print(f"Object category {obj_category} is too long for OwLv2 detection")
                    skip_episode = True

            #For OVON
            eff_step_num = 0 if not not_done_masks[0][0] else self._agent.actor_critic._num_steps
            if (eff_step_num == 0) and \
                    (not skip_episode) and \
                    (config.habitat.dataset.data_path.__contains__("ovon")) and \
                    (not config.grid_embed.saved_nav) and \
                    (not config.grid_embed.gui_nav):
                
                goal_embed = batch["clip_objectgoal"][0].cpu().numpy()
                obj_category = self._agent.actor_critic.OVON_ID_TO_NAME[array_hash(goal_embed)]

                print(f"OVON Object Category: {obj_category}")

                if obj_category not in saved_objects: 
                    print("Object not in Saved Data Directory! Skipping...")
                    skip_episode = True

            # log_file_name = f"{curr_episode_id}_{curr_scene_name}.json"        
            # log_file_path = os.path.join(os.environ["ZSOS_LOG_DIR"], log_file_name)
            # if os.path.exists(log_file_path): 
            #     print(f"\n\n----Logged File already exists at: {log_file_path}.\nSkipping Episode {curr_episode_id} for Scene {curr_scene_name}...")
            #     skip_episode = True


            print(f"Skip Episode: {skip_episode}\n")

            

            with inference_mode():
                
                # if not not_done_masks[0][0]:
                #     print(f"\n\n-----------Step Num: 0-----------")
                # else:
                print(f"\n\n-----------Step Num: {0 if not not_done_masks[0][0] else self._agent.actor_critic._num_steps}-----------")
                
                #################
                ##DEBUG
                if config.grid_embed.debug:
                    step_num = self._agent.actor_critic._num_steps
                    junk_dir = os.path.join(os.getcwd(), "last_run/steps_imgs")
                    if step_num == 0 and os.path.exists(junk_dir): 
                        shutil.rmtree(junk_dir)
                    os.makedirs(junk_dir, exist_ok=True)

                    try:

                        frame = hab_vis._create_frame(
                            hab_vis.depth[-1],
                            hab_vis.rgb[-1],
                            hab_vis.maps[-1],
                            hab_vis.vis_maps[-1],
                            hab_vis.texts[-1],
                        )

                        from PIL import Image                  
                        final_file_dir = os.path.join(junk_dir, f'{step_num}.png')
                        final_frame = Image.fromarray(frame)
                        final_frame.save(final_file_dir, format="PNG")
                        print(f"Created Frame {step_num} Image at: {final_file_dir}")
                        
                    except:
                        pass

                ######################

                if skip_episode:
                    stop_action = torch.tensor([[0]], dtype=torch.long)
                    
                    action_data = PolicyActionData(
                                    actions=stop_action,
                                    rnn_hidden_states=test_recurrent_hidden_states,
                                    # policy_info=[{"step_action": {}}],
                                )
                    
                    if config.grid_embed.gui_nav: cv2.destroyAllWindows()
                
                else:

                    try:
                
                        #Policy Action (See Habitat, ObjectNav, Base Policies)
                        action_data = self._agent.actor_critic.act(     #The output action_data should contain the policy_info
                            batch,
                            test_recurrent_hidden_states,
                            prev_actions,
                            not_done_masks,
                            deterministic=False,
                            gui_info = gui_info
                        )

                    except:
                        skip_episode = True

                        stop_action = torch.tensor([[0]], dtype=torch.long)
                        
                        action_data = PolicyActionData(
                                        actions=stop_action,
                                        rnn_hidden_states=test_recurrent_hidden_states,
                                        # policy_info=[{"step_action": {}}],
                                    )
                        
                        if config.grid_embed.gui_nav: cv2.destroyAllWindows()

                #Save the current step and action
                if config.grid_embed.gui_nav and not skip_episode:
                    info_step_action = action_data.policy_info[0]["step_action"]
                    step_actions[info_step_action["step"]] = info_step_action["action"]

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
            # in the subprocesses.s
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

            #Takes step in env, updates infos using the new policy_infos
            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            gui_info["top_down"] = infos[0]["top_down_map"]

            # if not config.grid_embed.gui_nav:
            policy_infos = self._agent.actor_critic.get_extra(action_data, infos, dones)
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])    #Updates all info keys present in policy_info keys


            #TODO ADDED: Save the agent position to txt file
            if (self.config.grid_embed.save_trajectory) and \
                (self._agent.actor_critic._num_steps > 0) and \
                (self._agent.actor_critic._num_steps < self.config.habitat.environment.max_episode_steps):
                
                save_traj_path = os.path.join(self.config.grid_embed.log_dir, "trajectory", f"{curr_episode_id}_{curr_scene_name}.txt")
                os.makedirs(os.path.dirname(save_traj_path), exist_ok = True)

                curr_hab_pos = get_curr_hab_pos(self.envs)[0]
                num_steps = self._agent.actor_critic._num_steps
                
                with open(save_traj_path, "a" if num_steps > 1 else "w") as f:
                # with open(save_traj_path, "a") as f:
                    # s_append = f"{self._agent.actor_critic._num_steps}, {curr_hab_pos[0]}, {curr_hab_pos[1]}, {curr_hab_pos[2]}\n"
                    f.write(f"{self._agent.actor_critic._num_steps}, {curr_hab_pos[0]}, {curr_hab_pos[1]}, {curr_hab_pos[2]}\n")
                    print(f"Saved Trajectory at step {num_steps} to {save_traj_path}")



            #TODO: Need to add floor detection
            # curr_hab_pos = get_curr_hab_pos(self.envs)
            # observations[0]['agent_height'] = curr_hab_pos[0][1]
            observations[0]['agent_height'] = 0.2

            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            finish_step_time = time()
            step_time = finish_step_time - start_step_time
            with open(step_time_path, "a") as f:
                f.write(f"{step_time}\n")

            #Masks RGB with Depth Thresholds
            if config.grid_embed.rgb_masking:
                print(f"\nMasking RGB...")
                rgb_batch = batch["rgb"]
                depth_batch = batch["depth"]
                max_depth_val = self.config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth
                depth_batch_unnorm = depth_batch * max_depth_val
                
                rgb_masked = mask_rgb_by_depth_generic(rgb_batch, depth_batch_unnorm, max_depth_val, 0.6)
                batch["rgb"] = rgb_masked
                print(f"Masked RGB...\n")

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
                if (not config.grid_embed.gui_nav) and (not skip_episode) and len(self.config.habitat_baselines.eval.video_option) > 0:
                    hab_vis.collect_data(batch, infos, action_data.policy_info)

                if not not_done_masks[i].item():

                    finish_episode_time = time()
                    episode_time = finish_episode_time - start_episode_time
                    with open(episode_time_path, "a") as f:
                        f.write(f"{episode_time}\n")
                    
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

                    else:
                        #If Failure, save the last run images to a separate dir for ref
                        if config.grid_embed.debug:
                            failure_dir = f"last_run/failures"

                            source_dir = f"last_run/steps_imgs"
                            curr_scene_name = os.path.basename(current_episodes_info[i].scene_id).split(".")[0]
                            dest_dir = f"{failure_dir}/{current_episodes_info[i].episode_id}_{curr_scene_name}"
                            
                            if os.path.exists(os.path.join(os.getcwd(), dest_dir)): shutil.rmtree(dest_dir)
                            shutil.copytree(source_dir, dest_dir)
                            print(f"\n Failed Episode {current_episodes_info[i].episode_id} steps are saved at: {failure_dir}")
                        


                    num_total += 1
                    print(f"Success rate: {num_successes / num_total * 100:.2f}% ({num_successes} out of {num_total})")

                    from vlfm.utils.episode_stats_logger import (
                        log_episode_stats,
                    )

                    #Logs Infos as JSON after removing all np arrays from values
                    if not config.grid_embed.gui_nav and not skip_episode:
                        
                        try:
                            infos[i]["sim_thresh"] = self._agent.actor_critic.sim_goal_thresh
                            infos[i]["top_sims"] = [float(score) for score in self._agent.actor_critic.top_sims]
                            infos[i]["top_xy"] = [float(xy) for xy in self._agent.actor_critic.top_xy]
                            infos[i]["nav_to_best_grid"] = self._agent.actor_critic.nav_to_best_grid
                            xy_before_nav = self._agent.actor_critic.xy_before_nav
                            infos[i]["xy_before_nav"] = [float(coord) for coord in self._agent.actor_critic.xy_before_nav] if xy_before_nav is not None else xy_before_nav
                        except:
                            pass

                        infos[i]["num_steps"] = self._agent.actor_critic._num_steps
                        infos[i]["final_pos"] = [float(coord) for coord in get_curr_hab_pos(self.envs)[i]]  #TODO: NEED TO REMOVE

                        infos[i]["embed_model_name"] = self._agent.actor_critic._obstacle_map.embed_model_name
                        
                        if "init_dict" in infos[i].keys():
                            infos[i]["init_dict"]["init_pose"] = init_pose_info[i]
                        else:
                            infos[i]["init_dict"] = {}
                            infos[i]["init_dict"]["init_pose"] = init_pose_info[i]

                        failure_cause = log_episode_stats(
                            current_episodes_info[i].episode_id,
                            current_episodes_info[i].scene_id,
                            infos[i],
                            config.grid_embed.embed_log_dir,
                            should_log_info=config.grid_embed.should_log_info,
                            should_log_embeds=config.grid_embed.should_log_embeds
                        )


                        if config.grid_embed.saved_nav: scenes_done.append(curr_scene_name)


                    #Added for gui
                    if config.grid_embed.gui_nav and not skip_episode:
                        if len(step_actions) > 0:

                            scene_name = step_actions["scene"]
                            episode_id = step_actions["episode"]

                            gui_log_root_dir = os.path.join(config.grid_embed.embed_log_dir, 'step_actions')
                            file_path = os.path.join(gui_log_root_dir, f"step_actions_scene_{scene_name}_ep_{episode_id}.npz")

                            os.makedirs(gui_log_root_dir, exist_ok=True)
                            
                            if not (os.path.exists(file_path) and os.path.getsize(file_path) > 0):
                                print(f"Logging traced step-actions for Scene: {scene_name} to {file_path}")
                                np.savez(file_path, **{str(k):v for (k, v) in step_actions.items()})
                            else:
                                print(f"Logging step-actions FAILED, as logged file already exists.")

                            
                            if self.config.grid_embed.only_valid_eps: scenes_done.append((scene_name, episode_id))
                            else: scenes_done.append(scene_name)

                        else:
                            print("\nNot saving traced GUI steps and actions. step_actions dictionary is empty.")


                    #Saves all the visualized frames into a video
                    elif (len(self.config.habitat_baselines.eval.video_option) > 0) and (not skip_episode):


                        # --- Saving as frames ----
                        from PIL import Image

                        #Save dir
                        embed_model_name = self._agent.actor_critic._obstacle_map.embed_model_name
                        curr_scene_name = os.path.basename(current_episodes_info[i].scene_id).split(".")[0]

                        imgs_root_dir = os.path.join(config.grid_embed.embed_log_dir, 'imgs', f"model_{embed_model_name}_scene_{curr_scene_name}_ep_{current_episodes_info[i].episode_id}")
                        os.makedirs(imgs_root_dir, exist_ok=True)


                        num_frames = len(hab_vis.depth) - 1
                        for i in tqdm.tqdm(range(num_frames)):

                            i_rgb = np.array(hab_vis.rgb[i])
                            i_depth = np.array(hab_vis.depth[i])
                            i_maps = np.array(hab_vis.maps[i])
                            i_obs_map = np.array(hab_vis.vis_maps[i][0])
                            i_val_map = np.array(hab_vis.vis_maps[i][1])
                            
                            print(i_rgb.shape)
                            print(i_depth.shape)
                            print(i_maps.shape)
                            print(i_obs_map.shape)
                            print(i_val_map.shape)

                            i_rgb = Image.fromarray(i_rgb)
                            i_depth = Image.fromarray(i_depth)
                            i_maps = Image.fromarray(i_maps)
                            i_obs_map = Image.fromarray(i_obs_map)
                            i_val_map = Image.fromarray(i_val_map)

                            pad_i = str(i).zfill(3)
                            rgb_save_path = os.path.join(imgs_root_dir, "rgb", f"vis_{pad_i}.png")
                            depth_save_path = os.path.join(imgs_root_dir, "depth", f"vis_{pad_i}.png")
                            maps_save_path = os.path.join(imgs_root_dir, "maps", f"vis_{pad_i}.png")
                            obs_map_save_path = os.path.join(imgs_root_dir, "obs_map", f"vis_{pad_i}.png")
                            val_map_save_path = os.path.join(imgs_root_dir, "val_map", f"vis_{pad_i}.png")

                            os.makedirs(os.path.dirname(rgb_save_path), exist_ok=True)
                            os.makedirs(os.path.dirname(depth_save_path), exist_ok=True)
                            os.makedirs(os.path.dirname(maps_save_path), exist_ok=True)
                            os.makedirs(os.path.dirname(obs_map_save_path), exist_ok=True)
                            os.makedirs(os.path.dirname(val_map_save_path), exist_ok=True)

                            i_rgb.save(rgb_save_path, format="PNG")
                            i_depth.save(depth_save_path, format="PNG")
                            i_maps.save(maps_save_path, format="PNG")
                            i_obs_map.save(obs_map_save_path, format="PNG")
                            i_val_map.save(val_map_save_path, format="PNG")

                        # -------------------

                        # try:
                        rgb_frames[i] = hab_vis.flush_frames(failure_cause)

                        from PIL import Image
                        images = np.array(rgb_frames[i])                        
                        embed_model_name = self._agent.actor_critic._obstacle_map.embed_model_name
                        curr_scene_name = os.path.basename(current_episodes_info[i].scene_id).split(".")[0]

                        gif_root_dir = os.path.join(config.grid_embed.embed_log_dir, 'gifs')
                        if not os.path.exists(gif_root_dir): os.makedirs(gif_root_dir, exist_ok=True)
                        gif_file_dir = os.path.join(gif_root_dir, f'scene_{curr_scene_name}_ep_{current_episodes_info[i].episode_id}.gif')
                        
                        #Save only Final Frame
                        final_file_dir = os.path.join(gif_root_dir, f'model_{embed_model_name}_scene_{curr_scene_name}_ep_{current_episodes_info[i].episode_id}.png')
                        final_frame = Image.fromarray(images[-1])
                        final_frame.save(final_file_dir, format="PNG")

                        print(f"Created Last Frame Image at: {final_file_dir}")

                        #Save all the Frames
                        # print(f"Saving the frames...")
                        # save_frames_dir = os.path.join(gif_root_dir, f'scene_{curr_scene_name}_ep_{current_episodes_info[i].episode_id}')
                        # os.makedirs(save_frames_dir, exist_ok=True)

                        # for i in tqdm.tqdm(range(len(images))):

                        #     pad_i = str(i).zfill(3)
                        #     frame_save_path = os.path.join(save_frames_dir, f'vis_{pad_i}.png')

                        #     curr_frame = Image.fromarray(images[i])
                        #     curr_frame.save(frame_save_path, format="PNG")

                        

                        if config.grid_embed.create_gif:
                            gif_images = list(map(Image.fromarray, images))
                            gif_images[0].save(gif_file_dir, save_all=True, append_images=gif_images[1:], duration=200, loop=0)
                            print(f'Created gif at : {gif_file_dir}')
                        
                        # except:
                        #     print(f"GIF or Last Frame Image could not be created.")
                        

                        # rgb_frames[i] = []

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

        # assert (
        #     len(ep_eval_count) >= number_of_eval_episodes
        # ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        # if not config.grid_embed.gui_nav and not config.grid_embed.saved_nav:

        #     aggregated_stats = {}
        #     for stat_key in next(iter(stats_episodes.values())).keys():
        #         aggregated_stats[stat_key] = np.mean([v[stat_key] for v in stats_episodes.values()])

        #     for k, v in aggregated_stats.items():
        #         logger.info(f"Average episode {k}: {v:.4f}")

        #     step_id = checkpoint_index
        #     if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
        #         step_id = ckpt_dict["extra_state"]["step"]

        #     writer.add_scalar("eval_reward/average_reward", aggregated_stats["reward"], step_id)

        #     metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        #     for k, v in metrics.items():
        #         writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()



def reset_gui_nav(embed_log_dir, scene_name, episode_id, init_pose_info, 
                  only_valid_eps=False, valid_eps_base_path="habitat-lab/data/datasets/objectnav/hm3d/v2", valid_eps_split="val"):
    r""""
    Returns:
        - step_actions: Newly defined dictionary with a few entries about the initialization. This is used to keep track of the steps and actions.
        - scenes_done: List of scene names for which GUI Navigation have been completed
        - should_skip: If the scene is already navigated (GUI), then skip episode
        - gui_info: Updates the gui_info with the current scene name and episode id
    """

    print(f"\n\nNavigation using GUI.")
    cv2.destroyAllWindows()     #Destroy all active GUI cv2 windows

    #Define step_actions
    # Add initial information: position, rotation, scene name, episode id
    step_actions = {}
    step_actions['init_pos_abs'] = init_pose_info[0]['init_pos_abs']
    step_actions['init_rot_abs'] = init_pose_info[0]['init_rot_abs']
    step_actions['scene'] = scene_name
    step_actions['episode'] = episode_id

    #Define scenes_done: Gets all scenes names (and episode ids) for which step-actions has been saved 
    # (These scenes have already been GUI Navigated)
    step_actions_dir = os.path.join(embed_log_dir, f"step_actions")
    if os.path.exists(step_actions_dir):
        scenes_done = [file_name for file_name in os.listdir(os.path.join(embed_log_dir, f"step_actions"))]
        
        if only_valid_eps:
            #scenes_done = [(file_name.split("_")[3], int(file_name.split("_")[-1].split(".")[0])) for file_name in scenes_done]
            scenes_done = [( file_name.split("scene_")[-1].split("_ep")[0], file_name.split("ep_")[-1].split(".npz")[0] ) for file_name in scenes_done]
        else:
            # scenes_done = [file_name.split("_")[3] for file_name in scenes_done]
            scenes_done = [file_name.split("scene_")[-1].split("_ep")[0] for file_name in scenes_done]
    else:
        scenes_done = []

    print(f"GUI Done for {len(scenes_done)} Scenes: {scenes_done}\n")

    ########

    if only_valid_eps:
        
        #If (Scene, Episode) already saved, then skip
        should_skip = ((scene_name, int(episode_id)) in scenes_done)
        if should_skip:
            print(f"\n({scene_name} Scene, {episode_id} Episode) has already been mapped! Skipping Episode...")

        #If (Scene, Episode) is not valid, then skip
        valid_info = read_valid_scenes(base_path=valid_eps_base_path, split=valid_eps_split)
        valid_scene_eps = [(valid_info[i]["file_name"].split(".")[0], valid_info[i]["id"]) for i in range(len(valid_info))]    #(Scene Name, Episode ID)
        if (scene_name, int(episode_id)) not in valid_scene_eps:
            should_skip = True

    else:

        #If Scene already saved, then skip
        should_skip = (scene_name in scenes_done)
        if should_skip:
            print(f"\n{scene_name} Scene has already been mapped! Skipping Episode...")

    #Update gui_info with new scene name and episode id
    gui_info = {}
    gui_info["scene_id"] = scene_name
    gui_info["episode_id"] = episode_id

    return step_actions, scenes_done, should_skip, gui_info


def reset_saved_nav(embed_log_dir, scene_name, episode_id, init_pose_info, with_saved_eps = False):
    r""""
    Returns:
        - step_actions: Dictionary of steps (keys) and actions (values) to navigate the (Scene, Episode)
        - should_skip: If file is not found, then skip the episode
    """
    
    print(f"\n\nNavigation using Saved Path")

    #Define scenes_done: Gets all scenes names for which step-actions has been saved 
    # (These scenes have already been GUI Navigated)
    embeds_dir = os.path.join(embed_log_dir, f"embed_dicts")
    if os.path.exists(embeds_dir):

        if with_saved_eps:
            scenes_done = []
            for scene_done_id in os.listdir(embeds_dir):
                scene_done_path = os.path.join(embeds_dir, scene_done_id)                       #Saved directory path for given scene 
                eps_done_ids = [file.split("_")[-1].split(".")[0] for file in os.listdir(scene_done_path)]  #Completed episodes of the scene
                print(f"Scene : {scene_done_id}, Episodes: {eps_done_ids}")

                for eps_done_id in eps_done_ids:
                    scenes_done.append((scene_done_id, eps_done_id))

        else:
            scenes_done = [file_name for file_name in os.listdir(embeds_dir)]
    else:
        scenes_done = []

    print(f"Embed Maps generated for {len(scenes_done)} Scenes: {scenes_done}\n")

    #If Step-Actions for (Scene, Episode) does not exist, then return
    saved_nav_scene_path = os.path.join(embed_log_dir, f'step_actions/step_actions_scene_{scene_name}_ep_{episode_id}.npz')
    # saved_nav_scene_path = os.path.join(embed_log_dir, f'step_actions/scene_{scene_name}_step_actions.npz')
    if not os.path.exists(saved_nav_scene_path): 
        print(f"\nSaved Step-Action Navigation does NOT exist for (Scene, Episode): ({scene_name}, {episode_id}). Skipping episode...")
        return None, scenes_done, True

    if with_saved_eps and ((scene_name, episode_id) in scenes_done):
        print(f"Found Saved Embed File! Skipping...")
        return None, scenes_done, True
    
    if not with_saved_eps and (scene_name in scenes_done):
        print(f"Found Saved Embed File! Skipping...")
        return None, scenes_done, True

    print(f"Loading Saved Step-Actions for (Scene, Episode): ({scene_name}, {episode_id})...")
    #Loads the Saved Step-Actions for the current (Scene, Episode)
    saved_nav_scene = dict(np.load(saved_nav_scene_path, allow_pickle=True))

    #Check if the initial position of episode is same as saved episode for scene
    # try:
    assert (saved_nav_scene['init_pos_abs'] == init_pose_info[0]['init_pos_abs']).all() and (saved_nav_scene['init_rot_abs'] == init_pose_info[0]['init_rot_abs']).all(), \
        f"Initial Position and Rotation are not same as in saved file! Please make sure you are using scene: {saved_nav_scene['scene']} and episode: {saved_nav_scene['episode']}" 
    
    del saved_nav_scene['init_pos_abs']
    del saved_nav_scene['init_rot_abs']
    del saved_nav_scene['scene']
    del saved_nav_scene['episode']
    # except:
    #     print(f"\n\n!!--- Check for Initial Position and Rotation FAILED! Resuming Run...")
    #     pass

    step_actions = {int(k):torch.Tensor(v).to(dtype=torch.int64) for (k, v) in saved_nav_scene.items()}

    return step_actions, scenes_done, False


def reset_saved_nav_with_eps(embed_log_dir, scene_name, episode_id, init_pose_info):
    r""""
    Returns:
        - step_actions: Dictionary of steps (keys) and actions (values) to navigate the (Scene, Episode)
        - should_skip: If file is not found, then skip the episode
    """
    
    print(f"\n\nNavigation using Saved Path")

    #Define scenes_done: Gets all scenes names for which step-actions has been saved 
    # (These scenes have already been GUI Navigated)
    embeds_dir = os.path.join(embed_log_dir, f"embed_dicts")
    if os.path.exists(embeds_dir):

        scenes_done = []
        for scene_done_id in os.listdir(embeds_dir):
            scene_done_path = os.path.join(embeds_dir, scene_done_id)                       #Saved directory path for given scene 
            eps_done_ids = [file.split("_")[-1].split(".")[0] for file in scene_done_path]  #Completed episodes of the scene

            for eps_done_id in eps_done_ids:
                scenes_done.append((scene_done_id, eps_done_id))

    else:
        scenes_done = []

    print(f"Embed Maps generated for {len(scenes_done)} Scenes: {scenes_done}\n")

    #If Step-Actions for (Scene, Episode) does not exist, then return
    saved_nav_scene_path = os.path.join(embed_log_dir, f'step_actions/step_actions_scene_{scene_name}_ep_{episode_id}.npz')
    # saved_nav_scene_path = os.path.join(embed_log_dir, f'step_actions/scene_{scene_name}_step_actions.npz')
    if not os.path.exists(saved_nav_scene_path): 
        print(f"\nSaved Step-Action Navigation does NOT exist for (Scene, Episode): ({scene_name}, {episode_id}). Skipping episode...")
        return None, scenes_done, True

    if (scene_name in scenes_done):
        print(f"Found Saved Embed File! Skipping...")
        return None, scenes_done, True

    print(f"Loading Saved Step-Actions for (Scene, Episode): ({scene_name}, {episode_id})...")
    #Loads the Saved Step-Actions for the current (Scene, Episode)
    saved_nav_scene = dict(np.load(saved_nav_scene_path, allow_pickle=True))

    #Check if the initial position of episode is same as saved episode for scene
    # try:
    assert (saved_nav_scene['init_pos_abs'] == init_pose_info[0]['init_pos_abs']).all() and (saved_nav_scene['init_rot_abs'] == init_pose_info[0]['init_rot_abs']).all(), \
        f"Initial Position and Rotation are not same as in saved file! Please make sure you are using scene: {saved_nav_scene['scene']} and episode: {saved_nav_scene['episode']}" 
    
    del saved_nav_scene['init_pos_abs']
    del saved_nav_scene['init_rot_abs']
    del saved_nav_scene['scene']
    del saved_nav_scene['episode']
    # except:
    #     print(f"\n\n!!--- Check for Initial Position and Rotation FAILED! Resuming Run...")
    #     pass

    step_actions = {int(k):torch.Tensor(v).to(dtype=torch.int64) for (k, v) in saved_nav_scene.items()}

    return step_actions, scenes_done, False