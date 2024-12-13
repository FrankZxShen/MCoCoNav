from collections import deque, defaultdict
from typing import Dict
from itertools import count
import os
import logging
import time
import json
import sys
import gym
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import quaternion
import pickle
import io
import re

from skimage import measure
import skimage.morphology
from PIL import Image

import math
import cv2
import habitat
import habitat_sim
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)

from agents.vlm_agents import LLM_Agent
from agents.vlm_agents_gt import LLM_Agent_GT
# from agents.llm_agents import LLM_Agent
from constants import (
    color_palette, coco_categories, coco_categories_hm3d2mp3d,
    hm3d_category, category_to_id, object_category
)
from envs.habitat.multi_agent_env_vlm import Multi_Agent_Env
# from envs.habitat.multi_agent_env import Multi_Agent_Env

# from src.geom import get_cam_intr, get_scene_bnds
from src.vlm import CogVLM2
from src.SystemPrompt import (
    form_prompt_for_PerceptionVLM, 
    form_prompt_for_FN,
    form_prompt_for_DecisionVLM_Frontier,
    Perception_weight_decision,
    Perception_weight_decision4,
)

import utils.pose as pu

import utils.visualization as vu

from arguments import get_args

# from detect_yolov9 import Detect
from detect.ultralytics import YOLOv10

@habitat.registry.register_action_space_configuration
class PreciseTurn(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[HabitatSimActions.TURN_LEFT_S] = habitat_sim.ActionSpec(
            "turn_left",
            habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE_S),
        )
        config[HabitatSimActions.TURN_RIGHT_S] = habitat_sim.ActionSpec(
            "turn_right",
            habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE_S),
        )

        return config


def Objects_Extract(args, full_map_pred, use_sam):

    semantic_map = full_map_pred[4:]

    dst = np.zeros(semantic_map[0, :, :].shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))

    Object_list = {}
    for i in range(len(semantic_map)):
        if semantic_map[i, :, :].sum() != 0:
            Single_object_list = []
            se_object_map = semantic_map[i, :, :].cpu().numpy()
            se_object_map[se_object_map>0.1] = 1
            se_object_map = cv2.morphologyEx(se_object_map, cv2.MORPH_CLOSE, kernel)
            contours, hierarchy = cv2.findContours(cv2.inRange(se_object_map,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                if len(cnt) > 30:
                    epsilon = 0.05 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    Single_object_list.append(approx)
                    cv2.polylines(dst, [approx], True, 1)
            if len(Single_object_list) > 0:
                if use_sam:
                    Object_list[object_category[i]] = Single_object_list
                else:
                    if 'objectnav_mp3d' in args.task_config:
                        Object_list[object_category[i]] = Single_object_list
                    elif 'objectnav_hm3d' in args.task_config:
                        Object_list[hm3d_category[i]] = Single_object_list
    return Object_list

def all_agents_exit_false(agents):
    for agent in agents:
        if agent.EXIT:
            return False
    return True

def all_agents_exit_true(agents):
    for agent in agents:
        if not agent.EXIT:
            return False
    return True

def ExtractExplorableAreas(full_map_pred, explo_area_map, VLM_PR, VLM_PR_last, color_map, count):
    PR = VLM_PR[0]

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    full_w = full_map_pred.shape[1]

    # local_ob_map = cv2.dilate(full_map_pred[0].cpu().numpy(), kernel)
    show_ex = cv2.inRange(full_map_pred[1].cpu().numpy(), 0.1, 1)

    kernel = np.ones((5, 5), dtype=np.uint8)
    free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(free_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    local_ob_map = cv2.dilate(full_map_pred[0].cpu().numpy(), kernel)
    explo_area_map_cur = np.zeros_like(local_ob_map)

    if len(contours) > 0:
        for contour in contours:
            if cv2.contourArea(contour) > 4: # Exclude very small areas from the example
                cv2.drawContours(explo_area_map_cur, [contour], -1, PR, -1) # Fill the interior with PR

    # Clear the border section
    explo_area_map_cur[0:2, 0:full_w] = 0
    explo_area_map_cur[full_w-2:full_w, 0:full_w] = 0
    explo_area_map_cur[0:full_w, 0:2] = 0
    explo_area_map_cur[0:full_w, full_w-2:full_w] = 0

    if VLM_PR_last:
        # mask = np.logical_and(explo_area_map_cur != PR, explo_area_map == VLM_PR_last[0])
        coords = np.where(explo_area_map != 0)
        # PR_coords = list(zip(coords[0], coords[1]))
        explo_area_map_cur[coords] = explo_area_map[coords]

    
    # Mark explorable areas as current colour
    intensity = int(PR * 100 * 2.55)
    intensity = max(0, min(intensity, 100))
    color_map[np.where(explo_area_map_cur == PR)] = [intensity, intensity, intensity]  #  RGB 值

    lipped_map = cv2.flip(color_map, 0)
    color_map__ = Image.fromarray(lipped_map)
    color_map__ = color_map__.convert("RGB")

    
    return explo_area_map_cur, color_map

def Frontiers(full_map_pred):
    # ------------------------------------------------------------------
    ##### Get the frontier map and filter
    # ------------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    full_w = full_map_pred.shape[1]
    local_ex_map = np.zeros((full_w, full_w))
    local_ob_map = np.zeros((full_w, full_w))

    local_ob_map = cv2.dilate(full_map_pred[0].cpu().numpy(), kernel)

    show_ex = cv2.inRange(full_map_pred[1].cpu().numpy(),0.1,1)
    
    kernel = np.ones((5, 5), dtype=np.uint8)
    free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)

    contours,_=cv2.findContours(free_map, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if len(contours)>0:
        contour = max(contours, key = cv2.contourArea)
        cv2.drawContours(local_ex_map,contour,-1,1,1)

    # clear the boundary
    local_ex_map[0:2, 0:full_w]=0.0
    local_ex_map[full_w-2:full_w, 0:full_w-1]=0.0
    local_ex_map[0:full_w, 0:2]=0.0
    local_ex_map[0:full_w, full_w-2:full_w]=0.0

    target_edge = local_ex_map-local_ob_map
    # print("local_ob_map ", self.local_ob_map[200])
    # print("full_map ", self.full_map[0].cpu().numpy()[200])

    target_edge[target_edge>0.8]=1.0
    target_edge[target_edge!=1.0]=0.0

    wall_edge = local_ex_map - target_edge

    # contours, hierarchy = cv2.findContours(cv2.inRange(wall_edge,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours)>0:
    #     dst = np.zeros(wall_edge.shape)
    #     cv2.drawContours(dst, contours, -1, 1, 1)

    # edges = cv2.Canny(cv2.inRange(wall_edge,0.1,1), 30, 90)
    Wall_lines = cv2.HoughLinesP(cv2.inRange(wall_edge,0.1,1), 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=10)

    # original_image_color = cv2.cvtColor(cv2.inRange(wall_edge,0.1,1), cv2.COLOR_GRAY2BGR)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(original_image_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    
    img_label, num = measure.label(target_edge, connectivity=2, return_num=True) # Output all connected fields in the binary image
    props = measure.regionprops(img_label) # Output properties of connected fields, including area, etc.

    Goal_edge = np.zeros((img_label.shape[0], img_label.shape[1]))
    Goal_point = []
    Goal_area_list = []
    dict_cost = {}
    for i in range(1, len(props)):
        if props[i].area > 4:
            dict_cost[i] = props[i].area

    if dict_cost:
        dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=True)

        for i, (key, value) in enumerate(dict_cost):
            Goal_edge[img_label == key + 1] = 1
            Goal_point.append([int(props[key].centroid[0]), int(props[key].centroid[1])])
            Goal_area_list.append(value)
            if i == 3:
                break
        # frontiers = cv2.HoughLinesP(cv2.inRange(Goal_edge,0.1,1), 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)

        # original_image_color = cv2.cvtColor(cv2.inRange(Goal_edge,0.1,1), cv2.COLOR_GRAY2BGR)
        # if frontiers is not None:
        #     for frontier in frontiers:
        #         x1, y1, x2, y2 = frontier[0]
        #         cv2.line(original_image_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return Wall_lines, Goal_area_list, Goal_edge, Goal_point

# 画出所有的Frontier
def Visualize(args, episode_n, l_step, pose_pred, full_map_pred, goal_name, visited_vis, map_edge, Frontiers_dict, goal_points):
    dump_dir = "{}/dump/{}/".format(args.dump_location,
                                    args.exp_name)
    ep_dir = '{}/episodes/eps_{}/'.format(
        dump_dir, episode_n)
    if not os.path.exists(ep_dir):
        os.makedirs(ep_dir)

    full_w = full_map_pred.shape[1]

    map_pred = full_map_pred[0, :, :].cpu().numpy()
    exp_pred = full_map_pred[1, :, :].cpu().numpy()

    sem_map = full_map_pred[4:, :,:].argmax(0).cpu().numpy()

    sem_map += 5

    # no_cat_mask = sem_map == 20
    if 'objectnav_hm3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2
    elif 'objectnav_mp3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2 + 5
    map_mask = np.rint(map_pred) == 1
    exp_mask = np.rint(exp_pred) == 1
    edge_mask = map_edge == 1

    sem_map[no_cat_mask] = 0
    m1 = np.logical_and(no_cat_mask, exp_mask)
    sem_map[m1] = 2

    m2 = np.logical_and(no_cat_mask, map_mask)
    sem_map[m2] = 1

    for i in range(args.num_agents):
        sem_map[visited_vis[i] == 1] = 3+i
    sem_map[edge_mask] = 3


    def find_big_connect(image):
        img_label, num = measure.label(image, return_num=True) # Output all connected fields in the binary image
        props = measure.regionprops(img_label) # Output properties of connected fields, including area, etc.
        # print("img_label.shape: ", img_label.shape) # 480*480
        resMatrix = np.zeros(img_label.shape)
        tmp_area = 0
        for i in range(0, len(props)):
            if props[i].area > tmp_area:
                tmp = (img_label == i + 1).astype(np.uint8)
                resMatrix = tmp
                tmp_area = props[i].area 
        
        return resMatrix

    goal = np.zeros((full_w, full_w)) 
    if 'objectnav_mp3d' in args.task_config:
        cn = goal_name + 4
    elif 'objectnav_hm3d' in args.task_config:
        cn = coco_categories[goal_name] + 4
    if full_map_pred[cn, :, :].sum() != 0.:
        cat_semantic_map = full_map_pred[cn, :, :].cpu().numpy()
        cat_semantic_scores = cat_semantic_map
        cat_semantic_scores[cat_semantic_scores > 0] = 1.
        goal = find_big_connect(cat_semantic_scores)

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
    elif len(goal_points) == args.num_agents and goal_points[i][0] != 9999:
        for i in range(args.num_agents):
            goal = np.zeros((full_w, full_w)) 
            goal[goal_points[i][0], goal_points[i][1]] = 1
            selem = skimage.morphology.disk(4)
            goal_mat = 1 - skimage.morphology.binary_dilation(
                goal, selem) != True
            goal_mask = goal_mat == 1

            sem_map[goal_mask] = 3 + i
    

    color_pal = [int(x * 255.) for x in color_palette]
    sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
    sem_map_vis.putpalette(color_pal)
    sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.flipud(sem_map_vis)

    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
    sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST)

    color = []
    for i in range(args.num_agents):
        color.append((int(color_palette[11+3*i] * 255),
                    int(color_palette[10+3*i] * 255),
                    int(color_palette[9+3*i] * 255)))

    # vis_image = vu.init_multi_vis_image(category_to_id[goal_name], color)
    if 'objectnav_mp3d' in args.task_config:
        vis_image = vu.init_multi_vis_image(object_category[goal_name], color)
    elif 'objectnav_hm3d' in args.task_config:
        vis_image = vu.init_multi_vis_image(object_category[coco_categories_hm3d2mp3d[goal_name]], color)

    vis_image[50:530, 15:495] = sem_map_vis

    color_black = (0,0,0)
    pattern = r'<centroid: (.*?), (.*?), number: (.*?)>'
    alpha = [chr(ord("A") + i) for i in range(26)]
    alpha0 = 0
    
    def d240(x):
        if x < 240:
            x = x + 2*(240-x)
        elif x >= 240:
            x = x - 2*(x-240)
        return x
    
    if Frontiers_dict:
        for keys, value in Frontiers_dict.items():
            match = re.match(pattern, value)
            if match:
                centroid_x = int(match.group(1)[1:])
                centroid_y = int(match.group(2)[:-1])
                number = float(match.group(3))
                # print(f"Centroid: ({centroid_x}, {centroid_y})")
                # print(f"Number: {number}")
                
                cv2.circle(sem_map_vis, (centroid_y, d240(centroid_x)), 5, color_black, -1)
                label = f"{alpha[alpha0]}"
                alpha0 += 1
                cv2.putText(sem_map_vis, label, (centroid_y + 5, d240(centroid_x) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_black, 1)
    for i in range(args.num_agents):
        agent_arrow = vu.get_contour_points(pose_pred[i], origin=(15, 50), size=10)

        cv2.drawContours(vis_image, [agent_arrow], 0, color[i], -1)
    if args.visualize:
        # Displaying the image
        cv2.imshow("episode_n {}".format(episode_n), vis_image)
        cv2.waitKey(1)

    if args.print_images:
        fn = '{}/episodes/eps_{}/Step-{}.png'.format(
            dump_dir, episode_n,
            l_step)
        # print(fn)
        cv2.imwrite(fn, vis_image)   

def Decision_Generation_Vis(args, agents_seg_list, agent_j, episode_n, l_step, pose_pred, full_map_pred, goal_name,
                             visited_vis, map_edge, history_nodes, Frontiers_dict, goal_points, pre_goal_point):
    full_w = full_map_pred.shape[1]

    map_pred = full_map_pred[0, :, :].cpu().numpy()
    exp_pred = full_map_pred[1, :, :].cpu().numpy()

    sem_map = full_map_pred[4:, :,:].argmax(0).cpu().numpy()

    sem_map += 5

    # no_cat_mask = sem_map == 20
    if 'objectnav_hm3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2
    elif 'objectnav_mp3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2 + 5
    map_mask = np.rint(map_pred) == 1
    exp_mask = np.rint(exp_pred) == 1
    edge_mask = map_edge == 1

    sem_map[no_cat_mask] = 0
    m1 = np.logical_and(no_cat_mask, exp_mask)
    sem_map[m1] = 2

    m2 = np.logical_and(no_cat_mask, map_mask)
    sem_map[m2] = 1

    for i in range(args.num_agents):
        sem_map[visited_vis[i] == 1] = 3+i
    sem_map[edge_mask] = 3


    def find_big_connect(image):
        img_label, num = measure.label(image, return_num=True) # Output all connected fields in the binary image
        props = measure.regionprops(img_label) # Output properties of connected fields, including area, etc.
        # print("img_label.shape: ", img_label.shape) # 480*480
        resMatrix = np.zeros(img_label.shape)
        tmp_area = 0
        for i in range(0, len(props)):
            if props[i].area > tmp_area:
                tmp = (img_label == i + 1).astype(np.uint8)
                resMatrix = tmp
                tmp_area = props[i].area 
        
        return resMatrix

    goal = np.zeros((full_w, full_w)) 
    if 'objectnav_mp3d' in args.task_config:
        cn = goal_name + 4
    elif 'objectnav_hm3d' in args.task_config:
        cn = coco_categories[goal_name] + 4
    if full_map_pred[cn, :, :].sum() != 0.:
        cat_semantic_map = full_map_pred[cn, :, :].cpu().numpy()
        cat_semantic_scores = cat_semantic_map
        cat_semantic_scores[cat_semantic_scores > 0] = 1.
        goal = find_big_connect(cat_semantic_scores)

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
    elif len(goal_points) == args.num_agents and goal_points[i][0] != 9999:
        for i in range(args.num_agents):
            goal = np.zeros((full_w, full_w)) 
            goal[goal_points[i][0], goal_points[i][1]] = 1
            selem = skimage.morphology.disk(4)
            goal_mat = 1 - skimage.morphology.binary_dilation(
                goal, selem) != True
            goal_mask = goal_mat == 1

            sem_map[goal_mask] = 3 + i
    pattern = r'<centroid: (.*?), (.*?), number: (.*?)>'
    if Frontiers_dict:
        for keys, value in Frontiers_dict.items():
            match = re.match(pattern, value)
            if match:
                centroid_x = int(match.group(1)[1:])
                centroid_y = int(match.group(2)[:-1])
                number = float(match.group(3))
            fgoal = np.zeros((full_w, full_w)) 
            fgoal[centroid_x, centroid_y] = 1
            selem = skimage.morphology.disk(4)
            goal_mat = 1 - skimage.morphology.binary_dilation(
                fgoal, selem) != True
            goal_mask = goal_mat == 1
            sem_map[goal_mask] = 2

    
    color = []
    for i in range(args.num_agents):
        color.append((int(color_palette[11+3*i] * 255),
                    int(color_palette[10+3*i] * 255),
                    int(color_palette[9+3*i] * 255)))
    
    color_pal = [int(x * 255.) for x in color_palette]
    sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
    sem_map_vis.putpalette(color_pal)
    sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.flipud(sem_map_vis)

    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
    sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST)

    color_black = (0,0,0)
    color_green = (0,255,0)
    color_red = (0,0,255)
    color_blue = (255,0,0)
    pattern = r'<centroid: (.*?), (.*?), number: (.*?)>'
    alpha = [chr(ord("A") + i) for i in range(26)]
    alpha0 = 0
    
    def d240(x):
        if x < 240:
            x = x + 2*(240-x)
        elif x >= 240:
            x = x - 2*(x-240)
        return x

    


    # for i in range(args.num_agents):
    #     agent_arrow = vu.get_contour_points(pose_pred[i], origin=(0, 0), size=10)

    #     cv2.drawContours(sem_map_vis, [agent_arrow], 0, color[i], -1)
    # agent_arrow = vu.get_contour_points(pose_pred[agent_j], origin=(0, 0), size=10)

    # cv2.drawContours(sem_map_vis, [agent_arrow], 0, color[agent_j], -1)
    if Frontiers_dict:
        for keys, value in Frontiers_dict.items():
            match = re.match(pattern, value)
            if match:
                centroid_x = int(match.group(1)[1:])
                centroid_y = int(match.group(2)[:-1])
                number = float(match.group(3))
                # print(f"Centroid: ({centroid_x}, {centroid_y})")
                # print(f"Number: {number}")
                
                cv2.circle(sem_map_vis, (centroid_y, d240(centroid_x)), 5, color_black, -1)
                label = f"{alpha[alpha0]}"
                alpha0 += 1
                cv2.putText(sem_map_vis, label, (centroid_y + 5, d240(centroid_x) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_black, 1)

    sem_map_vis2 = sem_map_vis.copy()
    beta = [chr(ord("a") + i) for i in range(26)]
    alpha0 = 0
    if len(history_nodes) > 0:
        for hs in history_nodes[:26]:
            centroid_x = int(hs[0])
            centroid_y = int(hs[1])
            cv2.circle(sem_map_vis, (centroid_y, d240(centroid_x)), 5, color_green, -1)
            label = f"{beta[alpha0]}"
            alpha0 += 1
            cv2.putText(sem_map_vis, label, (centroid_y + 5, d240(centroid_x) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_green, 1)
        alpha0 = 0
        for hs in history_nodes[26:]:
            centroid_x = int(hs[0])
            centroid_y = int(hs[1])
            cv2.circle(sem_map_vis, (centroid_y, d240(centroid_x)), 5, color_green, -1)
            label = f"{alpha[alpha0]}"
            alpha0 += 1
            cv2.putText(sem_map_vis, label, (centroid_y + 5, d240(centroid_x) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_green, 1)
    # Iterate through the dictionary and draw polygons
    for key, value in agents_seg_list.items():
        # Convert each value into a format suitable for use with cv2.polylines (a numpy array).
        for array in value:
            pts = array.reshape((-1, 1, 2))
            if agent_j == 0:
                for i in pts:
                    for j in i:
                        j[1] = d240(j[1])
            
            # Draw polygons
            # cv2.polylines(sem_map_vis, [pts], isClosed=True, color=color_bule, thickness=2)
            
            # Label the key values with the text at the first coordinate of the polygon.
            text_position = (pts[0][0][0], pts[0][0][1])
            # moments = cv2.moments(pts)
            # cX = int(moments["m10"] / moments["m00"])
            # cY = int(moments["m01"] / moments["m00"])
            cv2.putText(sem_map_vis, key, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(sem_map_vis2, key, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    

    # Draw the arrows
    # cv2.circle(sem_map_vis, (int(pose_pred[agent_j][0]), int(pose_pred[agent_j][1])), 8, color_red, -1)
    # cv2.circle(sem_map_vis2, (int(pose_pred[agent_j][0]), int(pose_pred[agent_j][1])), 8, color_red, -1)
    
    agent_arrow = vu.get_contour_points(pose_pred[agent_j], origin=(0, 0), size=15)
    cv2.drawContours(sem_map_vis, [agent_arrow], 0, color_red, -1)
    cv2.drawContours(sem_map_vis2, [agent_arrow], 0, color_red, -1)
    if pre_goal_point:
        cv2.circle(sem_map_vis, (int(pre_goal_point[1]), int(d240(pre_goal_point[0]))), 8, color_blue, -1)
        cv2.circle(sem_map_vis2, (int(pre_goal_point[1]), int(d240(pre_goal_point[0]))), 8, color_blue, -1)

    
    
    
    ### TEST
    dump_dir = "{}/dump/{}/".format(args.dump_location,
                                    args.exp_name)
    vis_ep_dir = '{}/episodes/eps_{}/Agents_vis'.format(
                dump_dir, episode_n)
    if not os.path.exists(vis_ep_dir):
        os.makedirs(vis_ep_dir)
    
    fn = '{}/episodes/eps_{}/Agents_vis/VisStep-{}.png'.format(
                        dump_dir, episode_n,
                        l_step)
    fn2 = '{}/episodes/eps_{}/Agents_vis/VisStep2-{}.png'.format(
                        dump_dir, episode_n,
                        l_step)
    cv2.imwrite(fn, sem_map_vis)  
    cv2.imwrite(fn2, sem_map_vis2) 

    return sem_map_vis, sem_map_vis2



def Visualize0(args, episode_n, l_step, pose_pred, full_map_pred, goal_name, visited_vis, map_edge, goal_points):
    dump_dir = "{}/dump/{}/".format(args.dump_location,
                                    args.exp_name)
    ep_dir = '{}/episodes/eps_{}/'.format(
        dump_dir, l_step)
    if not os.path.exists(ep_dir):
        os.makedirs(ep_dir)

    full_w = full_map_pred.shape[1]

    map_pred = full_map_pred[0, :, :].cpu().numpy()
    exp_pred = full_map_pred[1, :, :].cpu().numpy()

    sem_map = full_map_pred[4:, :,:].argmax(0).cpu().numpy()

    sem_map += 5

    if 'objectnav_hm3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2
    elif 'objectnav_mp3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2 + 5
    map_mask = np.rint(map_pred) == 1
    exp_mask = np.rint(exp_pred) == 1
    edge_mask = map_edge == 1

    sem_map[no_cat_mask] = 0
    m1 = np.logical_and(no_cat_mask, exp_mask)
    sem_map[m1] = 2

    m2 = np.logical_and(no_cat_mask, map_mask)
    sem_map[m2] = 1

    for i in range(args.num_agents):
        sem_map[visited_vis[i] == 1] = 3+i
    sem_map[edge_mask] = 3


    def find_big_connect(image):
        img_label, num = measure.label(image, return_num=True) # Output all connected fields in the binary image
        props = measure.regionprops(img_label) # Output properties of connected fields, including area, etc.
        # print("img_label.shape: ", img_label.shape) # 480*480
        resMatrix = np.zeros(img_label.shape)
        tmp_area = 0
        for i in range(0, len(props)):
            if props[i].area > tmp_area:
                tmp = (img_label == i + 1).astype(np.uint8)
                resMatrix = tmp
                tmp_area = props[i].area 
        
        return resMatrix

    goal = np.zeros((full_w, full_w)) 
    if 'objectnav_mp3d' in args.task_config:
        cn = goal_name + 4
    elif 'objectnav_hm3d' in args.task_config:
        cn = coco_categories[goal_name] + 4
    if full_map_pred[cn, :, :].sum() != 0.:
        cat_semantic_map = full_map_pred[cn, :, :].cpu().numpy()
        cat_semantic_scores = cat_semantic_map
        cat_semantic_scores[cat_semantic_scores > 0] = 1.
        goal = find_big_connect(cat_semantic_scores)

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
    elif len(goal_points) == args.num_agents:
        for i in range(args.num_agents):
            goal = np.zeros((full_w, full_w)) 
            goal[goal_points[i][0], goal_points[i][1]] = 1
            selem = skimage.morphology.disk(4)
            goal_mat = 1 - skimage.morphology.binary_dilation(
                goal, selem) != True
            goal_mask = goal_mat == 1

            sem_map[goal_mask] = 3 + i


    color_pal = [int(x * 255.) for x in color_palette]
    sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
    sem_map_vis.putpalette(color_pal)
    sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.flipud(sem_map_vis)

    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
    sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST)

    color = []
    for i in range(args.num_agents):
        color.append((int(color_palette[11+3*i] * 255),
                    int(color_palette[10+3*i] * 255),
                    int(color_palette[9+3*i] * 255)))

    vis_image = vu.init_multi_vis_image(category_to_id[goal_name], color)

    vis_image[50:530, 15:495] = sem_map_vis

    for i in range(args.num_agents):
        agent_arrow = vu.get_contour_points(pose_pred[i], origin=(15, 50), size=10)

        cv2.drawContours(vis_image, [agent_arrow], 0, color[i], -1)

    if args.visualize:
        # Displaying the image
        cv2.imshow("episode_n {}".format(episode_n), vis_image)
        cv2.waitKey(1)

    if args.print_images:
        fn = '{}/episodes/eps_{}/Vis-{}.png'.format(
            dump_dir, episode_n,
            l_step)
        cv2.imwrite(fn, vis_image)

def calculate_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if args.cuda else "cpu")

###########################################################===Load Models and Datasets===########################################################
    # Initialise habitat
    HabitatSimActions.extend_action_space("TURN_LEFT_S")
    HabitatSimActions.extend_action_space("TURN_RIGHT_S")

    config_env = habitat.get_config(config_paths=["envs/habitat/configs/"
                                         + args.task_config])
    config_env.defrost()
    
    agent_sensors = []
    agent_sensors.append("RGB_SENSOR")
    agent_sensors.append("DEPTH_SENSOR")
    agent_sensors.append("SEMANTIC_SENSOR")

    config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors
    config_env.SIMULATOR.SEMANTIC_SENSOR.WIDTH = args.env_frame_width
    config_env.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = args.env_frame_height
    config_env.SIMULATOR.SEMANTIC_SENSOR.HFOV = args.hfov
    config_env.SIMULATOR.SEMANTIC_SENSOR.POSITION = \
        [0, args.camera_height, 0]

    config_env.TASK.POSSIBLE_ACTIONS = config_env.TASK.POSSIBLE_ACTIONS + [
        "TURN_LEFT_S",
        "TURN_RIGHT_S",
    ]
    config_env.TASK.ACTIONS.TURN_LEFT_S = habitat.config.Config()
    config_env.TASK.ACTIONS.TURN_LEFT_S.TYPE = "TurnLeftAction_S"
    config_env.TASK.ACTIONS.TURN_RIGHT_S = habitat.config.Config()
    config_env.TASK.ACTIONS.TURN_RIGHT_S.TYPE = "TurnRightAction_S"
    config_env.SIMULATOR.ACTION_SPACE_CONFIG = "PreciseTurn"
    config_env.freeze()

    # ------------------------------------------------------------------
    # Load VLM
    # ------------------------------------------------------------------
    # vlm = VLM(args.vlm_model_id, args.hf_token, device)
    base_url = args.base_url 
    cogvlm2 = CogVLM2(base_url) 
    # ------------------------------------------------------------------
    # Load Yolo
    # ------------------------------------------------------------------
    # yolo = Detect(imgsz=(args.env_frame_height, args.env_frame_width), device=device)
    if args.yolo == 'yolov9':
        # yolo = Detect(imgsz=(args.env_frame_height, args.env_frame_width), device=device)
        pass
    else:
        yolo = YOLOv10.from_pretrained(args.yolo_weights)
    # print(config_env)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    
    env = Multi_Agent_Env(config_env=config_env)

    num_episodes = env.number_of_episodes

    assert num_episodes > 0, "num_episodes should be greater than 0"

    num_agents = config_env.SIMULATOR.NUM_AGENTS

    agent = []
    agent_GT = []
    for i in range(num_agents):
        agent.append(LLM_Agent(args, config_env, i, device))
        if 'objectnav_hm3d' in args.task_config:
            agent_GT.append(LLM_Agent_GT(args, config_env, i, device))

    # ------------------------------------------------------------------
    ##### Setup Logging
    # ------------------------------------------------------------------
    log_dir = "{}/logs/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    logging.basicConfig(
        filename=log_dir + 'output.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    # print(args)
    # logging.info(args)
    # ------------------------------------------------------------------

    # print("num_episodes:",num_episodes)# 1000

    agg_metrics: Dict = defaultdict(float)
    # obj_SR: Dict = defaultdict(float)
    # sys_metrics: Dict = defaultdict(float)
    agg_metrics['multi_Total_SR'] = 0
    agg_metrics['SPL'] = 0
    agg_metrics['SoftSPL'] = 0
    agg_metrics['multi_SPL'] = {}
    agg_metrics['multi_SoftSPL'] = {}
    agg_metrics['multi_Navigation_SR'] = 0
    for i in range(num_agents):
        agg_metrics['multi_SPL'][f'Agent_{i}'] = 0
        agg_metrics['multi_SoftSPL'][f'Agent_{i}'] = 0

    count_episodes = 0
    count_step = 0
    goal_points = []
    
    log_start = time.time()
    last_decision = []
    total_usage = []

    history_nodes = []
    history_score = []
    history_count = []
    history_states = []

    cur_goal_points = []
    pre_goal_points = []

    # random
    log_start = time.time()
    last_decision = []
    total_usage = []

    pre_g_points = []

    target_point = []

    

    # logging.info(f"num agents: {num_agents}")

###########################################################===Main MCoCoNav===########################################################
    while count_episodes < num_episodes:
        observations = env.reset()
        for i in range(num_agents):
            agent[i].reset()
            if 'objectnav_hm3d' in args.task_config:
                agent_GT[i].reset()
        
        history_nodes.clear()
        history_score.clear()
        history_count.clear()
        history_states.clear()
        pre_g_points.clear()
        target_point.clear()

        goal_points.clear()
        for j in range(num_agents):
            goal_points.append([0, 0])

        while not env.episode_over:
            
            all_rgb = [] # hold the rgb of each robot

            Local_Policy = 0 # local policy
            start = time.time()
            count_rotating = 0
            action = []
            
            for j in range(num_agents):
                action.append(0)
                
            full_map = []
            full_map1 = []
            visited_vis = []
            pose_pred = []
            agent_objs = {} # Record target detection information for each smart body in a single time step

            agent_FrontierList = [] # Record the robot Frontier
            agent_TargetEdgeMap = []
            agent_TargetPointMap = []
            agent_MapPred = []

            for i in range(num_agents):
                agent[i].mapping(observations[i])
                if 'objectnav_hm3d' in args.task_config:
                    agent_GT[i].mapping(observations[i])
                local_map1, _ = torch.max(agent[i].local_map.unsqueeze(0), 0)
                full_map.append(agent[i].local_map)
                visited_vis.append(agent[i].visited_vis)
                start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs

                gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
                pos = (
                    (start_x * 100. / args.map_resolution - gy1)
                    * 480 / agent[i].visited_vis.shape[0],
                    (agent[i].visited_vis.shape[1] - start_y * 100. / args.map_resolution + gx1)
                    * 480 / agent[i].visited_vis.shape[1],
                    np.deg2rad(-start_o)
                )
                pose_pred.append(pos)

                    
            full_map2 = torch.cat([fm.unsqueeze(0) for fm in full_map], dim=0)
            # full_map2 = full_map[0].unsqueeze(0)
            # logging.info(f"full_map2: {full_map2.shape}") #[x,20,480,480]

            full_map_pred, _ = torch.max(full_map2, 0)
            Wall_list, full_Frontier_list, full_target_edge_map, full_target_point_map = Frontiers(full_map_pred)

            if agent[0].goal_id + 4 > 24:
                break

            if agent[0].l_step % args.num_local_steps == args.num_local_steps - 1 or agent[0].l_step == 0:
                for j in range(num_agents):
                    agent[j].Perception_PR = 0
                
                agents_seg_list = Objects_Extract(args, full_map_pred, args.use_sam)

                pre_goal_points.clear()
                if len(cur_goal_points) > 0:
                    pre_goal_points = cur_goal_points.copy()
                    cur_goal_points.clear()
                    
                if len(full_target_point_map) > 0:
                    full_Frontiers_dict = {}
                    for j in range(len(full_target_point_map)):
                        full_Frontiers_dict['frontier_' + str(j)] = f"<centroid: {full_target_point_map[j][0], full_target_point_map[j][1]}, number: {full_Frontier_list[j]}>"
                    logging.info(f'=====> Frontier: {full_Frontiers_dict}')

                    if len(history_nodes) > 0:
                        logging.info(f'=====> history_nodes: {history_nodes}')
                        logging.info(f'=====> history_score: {history_score}')


                    # ------------------------------------------------------------------
                    ##### VLM Preliminaries :>
                    # ------------------------------------------------------------------
                    for j in range(num_agents):
                        agent[j].is_Frontier = True
                        rgb = observations[j]['rgb'].astype(np.uint8)
                        
                        # full_rgb1.append(full_rgb)
                        all_rgb.append(rgb)
                        goal_name = agent[j].goal_name
                        if args.yolo == 'yolov9':
                            agent_objs[f"agent_{j}"] = yolo.run(rgb) # Record target detection information for each robot in a single time step.
                        else:
                            yolo_output = yolo(source=rgb,conf=0.2)
                            yolo_mapping = [yolo_output[0].names[int(c)] for c in yolo_output[0].boxes.cls]
                            agent_objs[f"agent_{j}"] = {k: v for k, v in zip(yolo_mapping, yolo_output[0].boxes.conf)}
                        # logging.info(agent_objs)
                        
                        # agents_seg_list = Objects_Extract(local_map1, args.use_sam)
                        single_map = [full_map[j]]

                        full_map1.append(torch.cat([fm.unsqueeze(0) for fm in single_map], dim=0))
                        full_map_pred1, _ = torch.max(full_map1[j], 0)
                        Wall_list, Frontier_list, target_edge_map, target_point_map = Frontiers(full_map_pred1)
                        agent_FrontierList.append(Frontier_list)
                        agent_TargetEdgeMap.append(target_edge_map)
                        agent_TargetPointMap.append(target_point_map)
                        agent_MapPred.append(full_map_pred1)

                        

                        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[j].planner_pose_inputs
                        r, c = start_y, start_x
                        start = [int(r * 100.0 / args.map_resolution - gx1),
                                int(c * 100.0 / args.map_resolution - gy1)]
                        start = pu.threshold_poses(start, agent[j].local_map[0, :, :].cpu().numpy().shape)
                        
                        if len(pre_goal_points) > 0:
                            # sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, j, agent[0].episode_n, agent[0].l_step, pose_pred, agent_MapPred[j], 
                            #                 agent[j].goal_id, visited_vis[j], agent_TargetEdgeMap[j], history_nodes, full_Frontiers_dict, goal_points=[], pre_goal_point=pre_goal_points[j])
                            sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, j, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                                    agent[0].goal_id, visited_vis, full_target_edge_map, history_nodes, full_Frontiers_dict, goal_points=[], pre_goal_point=pre_goal_points[j])
                        else:
                            # sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, j, agent[0].episode_n, agent[0].l_step, pose_pred, agent_MapPred[j], 
                            #                 agent[j].goal_id, visited_vis[j], agent_TargetEdgeMap[j], history_nodes, full_Frontiers_dict, goal_points=[], pre_goal_point=None)
                            sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, j, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                                    agent[0].goal_id, visited_vis, full_target_edge_map, history_nodes, full_Frontiers_dict, goal_points=[], pre_goal_point=None)
                        # full_rgb = np.hstack((rgb, sem_map))

                        
                        # ------------------------------------------------------------------
                        #### Perception VLM
                        # ------------------------------------------------------------------
                        Caption_Prompt, VLM_Perception_Prompt = form_prompt_for_PerceptionVLM(goal_name, agent_objs[f'agent_{j}'], args.yolo)
                        _, Scene_Information = cogvlm2.simple_image_chat(User_Prompt=Caption_Prompt, 
                                                                        return_string_probabilities=None, img=rgb)
                        Perception_Rel, Perception_Pred = cogvlm2.CoT2(User_Prompt1=Caption_Prompt, 
                                                                       User_Prompt2=VLM_Perception_Prompt,
                                                                       cot_pred1=Scene_Information,
                                                                       return_string_probabilities="[Yes, No]", img=rgb)
                        Perception_Rel = np.array(Perception_Rel)
                        Perception_PR = Perception_weight_decision(Perception_Rel, Perception_Pred)
                        logging.info(f"Agent_{j}--VLM_PerceptionPR: {Perception_PR}")
                        # agents_VLM_Rel[f"Agent_{i}--VLM_PerceptionRel"] = Perception_Rel
                        # agents_VLM_Pred[f"Agent_{i}--VLM_PerceptionPred"] = Perception_Pred
                        # agents_VLM_PR[f"Agent_{i}--VLM_PerceptionPR"] = Perception_PR

                        

                        is_exist_oldhistory = False
                        if len(history_nodes) > 0:
                            closest_index = -1
                            min_distance = float('inf')
                            new_x, new_y = start
                            for i, (x, y) in enumerate(history_nodes):
                                distance = math.sqrt((x - new_x) * (x - new_x) + (y - new_y) * (y - new_y))
                                if distance < 25 and distance < min_distance:
                                    min_distance = distance
                                    closest_index = i
                                    is_exist_oldhistory = True

                            if  is_exist_oldhistory == False:
                                history_nodes.append(start)
                                history_count.append(1)
                                history_state = np.zeros(360)
                            else:
                                history_count[closest_index] = history_count[closest_index] + 1

                            
                        else:
                            history_nodes.append(start)
                            history_count.append(1)
                            history_state = np.zeros(360)

                        
                        cur_goal_points.append(start)

                        if len(agent_TargetPointMap[j]) > 0:
                            
                            logging.info(f'=====> Agent_{j} state: Step: {agent[j].l_step}; Angle: {start_o}')

                            # ------------------------------------------------------------------
                            #### Judgment VLM
                            # ------------------------------------------------------------------
                            if len(history_nodes) > 0:
                                if len(pre_goal_points) > 0:
                                    FN_Prompt = form_prompt_for_FN(goal_name, agents_seg_list, Perception_PR, pre_goal_points[j], full_Frontiers_dict, start, history_nodes)
                                else:
                                    FN_Prompt = form_prompt_for_FN(goal_name, agents_seg_list, Perception_PR, pre_goal_points, full_Frontiers_dict, start, history_nodes)
                                # logging.info(FN_Prompt)
                                
                                FN_Rel, FN_Decision = cogvlm2.simple_image_chat(User_Prompt=FN_Prompt, 
                                                                                        return_string_probabilities="[Yes, No]", img=sem_map)

                                FN_PR = Perception_weight_decision(FN_Rel, FN_Decision)
                                logging.info(f"Agent_{j}--FN_PR: {FN_PR}")
                                if FN_PR == 'Neither':
                                    FN_PR = FN_Rel

                                
                                
                                angle_score = Perception_PR[0] * 2 + FN_PR[0]
                                agent[j].angle_score = angle_score
                                c_angle = int(start_o % 360)

                                if is_exist_oldhistory == False:
                                    if c_angle >= 39 and c_angle < 321:
                                        history_state[c_angle-39:c_angle+39] = angle_score
                                    elif c_angle < 39:
                                        history_state[:c_angle+39] = angle_score
                                        history_state[360-c_angle-39:] = angle_score

                                    elif c_angle >= 321:
                                        history_state[c_angle-39:] = angle_score
                                        history_state[:c_angle+39-360] = angle_score
                                    h_score = history_state.sum()
                                    history_states.append(history_state)
                                    history_score.append(h_score)
                                else:
                                    if c_angle >= 39 and c_angle < 321:
                                        history_states[closest_index][c_angle-39:c_angle+39] = angle_score
                                    elif c_angle < 39:
                                        history_states[closest_index][:c_angle] = angle_score
                                        history_states[closest_index][360-c_angle:] = angle_score
                                    elif c_angle >= 321:
                                        history_states[closest_index][c_angle:] = angle_score
                                        history_states[closest_index][:360-c_angle] = angle_score
                                    h_score = history_states[closest_index].sum() / history_count[closest_index]
                                    history_score[closest_index] = h_score

                            logging.info(f'=====> history_nodes: {history_nodes}')
                            logging.info(f'=====> history_score: {history_score}')
                            # Scores = []
                            if j == 0:
                                history_nodes_copy = history_nodes.copy()
                                history_score_copy = history_score.copy()
                                full_Frontiers_dict_copy = full_Frontiers_dict.copy()
                            else:
                                missing_key_F = []
                                if len(full_Frontiers_dict) == 4:
                                    frontier_keys = ['frontier_0', 'frontier_1', 'frontier_2', 'frontier_3']
                                elif len(full_Frontiers_dict) == 3:
                                    frontier_keys = ['frontier_0', 'frontier_1', 'frontier_2']
                                elif len(full_Frontiers_dict) == 2:
                                    frontier_keys = ['frontier_0', 'frontier_1']
                                else:
                                    frontier_keys = ['frontier_0']

                                for element in full_Frontiers_dict.keys():
                                    if element not in full_Frontiers_dict_copy.keys():
                                        missing_key_F.append(element)
                                # for element in history_nodes:
                                #     if element not in history_nodes_copy:
                                #         missing_index_H.append(element.index(element))
                            if FN_PR[0] >= 0.5 or agent[j].l_step <= 125:
                                # ------------------------------------------------------------------
                                #### Decision VLM
                                # ------------------------------------------------------------------
                                if len(pre_goal_points) > 0:
                                    Meta_Prompt = form_prompt_for_DecisionVLM_Frontier(Scene_Information, agents_seg_list, pre_goal_points[j], goal_name, start, full_Frontiers_dict_copy)
                                else:
                                    Meta_Prompt = form_prompt_for_DecisionVLM_Frontier(Scene_Information, agents_seg_list, pre_goal_points, goal_name, start, full_Frontiers_dict_copy)
                                
                                Meta_Score, Meta_Choice = cogvlm2.simple_image_chat(User_Prompt=Meta_Prompt,
                                                            return_string_probabilities="[A, B, C, D]", img=sem_map_frontier)

                                Final_PR = Perception_weight_decision4(Meta_Score, Meta_Choice)
                                
                            else:
                                Final_PR = history_score_copy

                            logging.info(f"Agent_{j}--Final_PR: {Final_PR}")

                            # Scores.append(Final_PR)
                            Choice = Final_PR.index(max(Final_PR))
                            
                            
                            if FN_PR[0] >= 0.5 or agent[j].l_step <= 125:
                                logging.info(f"VLM Choice: Agent_{j}-frontier_{Choice}")
                                Choice2 = Meta_Score.index(max(Meta_Score))

                                if len(full_Frontiers_dict) == 1:
                                    goal_points[j] = [int(x) for x in full_Frontiers_dict['frontier_0'].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                
                                elif len(full_Frontiers_dict) == 2 and num_agents == 3:
                                    if j == 0:
                                        for i, key in enumerate(frontier_keys):
                                            if Choice == i:
                                                if key in full_Frontiers_dict_copy:
                                                    goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[key].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                                    del full_Frontiers_dict_copy[key]
                                    elif j == 1:
                                        if len(missing_key_F) != 0:
                                            for keys in missing_key_F:
                                                frontier_keys.remove(keys)
                                        for i, key in enumerate(frontier_keys):
                                            goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[key].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                    else:
                                        if len(missing_key_F) != 0:
                                            for keys in missing_key_F:
                                                frontier_keys.remove(keys)
                                        for i, key in enumerate(frontier_keys):
                                            goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[key].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                
                                
                                else:
                                    if j > 0:
                                        if len(missing_key_F) != 0:
                                            for keys in missing_key_F:
                                                frontier_keys.remove(keys)
                                    else:
                                        if len(full_Frontiers_dict) == 4:
                                            frontier_keys = ['frontier_0', 'frontier_1', 'frontier_2', 'frontier_3']
                                        elif len(full_Frontiers_dict) == 3:
                                            frontier_keys = ['frontier_0', 'frontier_1', 'frontier_2']
                                        elif len(full_Frontiers_dict) == 2:
                                            frontier_keys = ['frontier_0', 'frontier_1']
                                        else:
                                            frontier_keys = ['frontier_0']

                                    invalid_answer = False
                                    for i, key in enumerate(frontier_keys):
                                        if Choice == i:
                                            if key in full_Frontiers_dict_copy:
                                                goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[key].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                                del full_Frontiers_dict_copy[key]
                                            else:
                                                invalid_answer = True
                                            break
                                    if invalid_answer:
                                        for i, key in enumerate(frontier_keys):
                                            if Choice2 == i:
                                                try:
                                                    goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[key].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                                    del full_Frontiers_dict_copy[key]
                                                    break
                                                except:
                                                    goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[frontier_keys[0]].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                                    del full_Frontiers_dict_copy[frontier_keys[0]]
                                                    break
                                        

                            else:
                                logging.info(f"VLM Choice: Agent_{j}-history_{Choice}")
                                if len(history_nodes_copy)==1:
                                    goal_points[j] = history_nodes_copy[0]
                                else:
                                    for i in range(len(history_nodes_copy)):
                                        if Choice == i:
                                            goal_points[j] = history_nodes_copy[i]
                                            del history_nodes_copy[i]
                                            del history_score_copy[i]
                                            break

                            
                            
                        else:
                            logging.info(f'===== Agent_{j} No Frontier, Random Mode =====')
                            #### Modify to history node
                            agent[j].is_Frontier = False
                            c_angle = int(start_o % 360)
                            angle_score = Perception_PR[0] * 2
                            agent[j].angle_score = angle_score

                            if is_exist_oldhistory == False:
                                if c_angle >= 39 and c_angle < 321:
                                    history_state[c_angle-39:c_angle+39] = angle_score
                                elif c_angle < 39:
                                    history_state[:c_angle+39] = angle_score
                                    history_state[360-c_angle-39:] = angle_score

                                elif c_angle >= 321:
                                    history_state[c_angle-39:] = angle_score
                                    history_state[:c_angle+39-360] = angle_score
                                h_score = history_state.sum()
                                history_states.append(history_state)
                                history_score.append(h_score)
                            else:
                                if c_angle >= 39 and c_angle < 321:
                                    history_states[closest_index][c_angle-39:c_angle+39] = angle_score
                                elif c_angle < 39:
                                    history_states[closest_index][:c_angle] = angle_score
                                    history_states[closest_index][360-c_angle:] = angle_score
                                elif c_angle >= 321:
                                    history_states[closest_index][c_angle:] = angle_score
                                    history_states[closest_index][:360-c_angle] = angle_score
                                h_score = history_states[closest_index].sum() / history_count[closest_index]
                                history_score[closest_index] = h_score

                            if j == 0:
                                history_nodes_copy = history_nodes.copy()
                                history_score_copy = history_score.copy()
                                full_Frontiers_dict_copy = full_Frontiers_dict.copy()
                            
                            if len(full_Frontiers_dict) == 1:
                                logging.info(f'=====> Agent_{j} state: Step: {agent[j].l_step}; Angle: {start_o}')
                                actions = np.random.rand(1, 2).squeeze()*(full_target_edge_map.shape[0] - 1)
                                goal_points[j] = [int(actions[0]), int(actions[1])]
                            else:
                                if  j == 0:
                                    frontier_keys = ['frontier_0', 'frontier_1', 'frontier_2', 'frontier_3']
                                logging.info(f'=====> Agent_{j} state: Step: {agent[j].l_step}; Angle: {start_o}')
                                actions = np.random.rand(1, 2).squeeze()*(full_target_edge_map.shape[0] - 1)
                                goal_points[j] = [int(actions[0]), int(actions[1])]
                            
                            
                    
                    # all_objs.append(agent_objs) 
                    # all_VLM_Pred.append(agents_VLM_Pred)
                    # all_VLM_PR.append(agents_VLM_PR)

                else:
                    
                    logging.info(f'===== No Frontier, Random Mode===== ')
                    logging.info(f'=====> Agent_{j} state: Step: {agent[j].l_step}; Angle: {start_o}')
                    
                    for j in range(num_agents):
                        agent[j].is_Frontier = False
                        rgb = observations[j]['rgb'].astype(np.uint8)
                        
                        # full_rgb1.append(full_rgb)
                        all_rgb.append(rgb)
                        goal_name = agent[j].goal_name
                        if args.yolo == 'yolov9':
                            agent_objs[f"agent_{j}"] = yolo.run(rgb) # Record target detection information for each smart body in a single time step
                        else:
                            yolo_output = yolo(source=rgb,conf=0.2)
                            yolo_mapping = [yolo_output[0].names[int(c)] for c in yolo_output[0].boxes.cls]
                            agent_objs[f"agent_{j}"] = {k: v for k, v in zip(yolo_mapping, yolo_output[0].boxes.conf)}
                        # logging.info(agent_objs)

                        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[j].planner_pose_inputs
                        r, c = start_y, start_x
                        start = [int(r * 100.0 / args.map_resolution - gx1),
                                int(c * 100.0 / args.map_resolution - gy1)]
                        start = pu.threshold_poses(start, agent[j].local_map[0, :, :].cpu().numpy().shape)
                        
                        cur_goal_points.append(start)

                        # ------------------------------------------------------------------
                        #### Perception VLM
                        # ------------------------------------------------------------------
                        Caption_Prompt, VLM_Perception_Prompt = form_prompt_for_PerceptionVLM(goal_name, agent_objs[f'agent_{j}'], args.yolo)
                        _, Scene_Information = cogvlm2.simple_image_chat(User_Prompt=Caption_Prompt, 
                                                                        return_string_probabilities=None, img=rgb)
                        Perception_Rel, Perception_Pred = cogvlm2.CoT2(User_Prompt1=Caption_Prompt, 
                                                                       User_Prompt2=VLM_Perception_Prompt,
                                                                       cot_pred1=Scene_Information,
                                                                       return_string_probabilities="[Yes, No]", img=rgb)
                        Perception_Rel = np.array(Perception_Rel)
                        Perception_PR = Perception_weight_decision(Perception_Rel, Perception_Pred)
                        logging.info(f"Agent_{j}--VLM_PerceptionPR: {Perception_PR}")

                        is_exist_oldhistory = False
                        if len(history_nodes) > 0:
                            closest_index = -1
                            min_distance = float('inf')
                            new_x, new_y = start
                            for i, (x, y) in enumerate(history_nodes):
                                distance = math.sqrt((x - new_x) * (x - new_x) + (y - new_y) * (y - new_y))
                                if distance < 25 and distance < min_distance:
                                    min_distance = distance
                                    closest_index = i
                                    is_exist_oldhistory = True

                            if  is_exist_oldhistory == False:
                                history_nodes.append(start)
                                history_count.append(1)
                                history_state = np.zeros(360)
                            else:
                                history_count[closest_index] = history_count[closest_index] + 1

                            
                        else:
                            history_nodes.append(start)
                            history_count.append(1)
                            history_state = np.zeros(360)


                        angle_score = Perception_PR[0] * 2
                        agent[j].angle_score = angle_score
                        c_angle = int(start_o % 360)

                        if is_exist_oldhistory == False:
                            if c_angle >= 39 and c_angle < 321:
                                history_state[c_angle-39:c_angle+39] = angle_score
                            elif c_angle < 39:
                                history_state[:c_angle+39] = angle_score
                                history_state[360-c_angle-39:] = angle_score

                            elif c_angle >= 321:
                                history_state[c_angle-39:] = angle_score
                                history_state[:c_angle+39-360] = angle_score
                            h_score = history_state.sum()
                            history_states.append(history_state)
                            history_score.append(h_score)
                        else:
                            if c_angle >= 39 and c_angle < 321:
                                history_states[closest_index][c_angle-39:c_angle+39] = angle_score
                            elif c_angle < 39:
                                history_states[closest_index][:c_angle] = angle_score
                                history_states[closest_index][360-c_angle:] = angle_score
                            elif c_angle >= 321:
                                history_states[closest_index][c_angle:] = angle_score
                                history_states[closest_index][:360-c_angle] = angle_score
                            h_score = history_states[closest_index].sum() / history_count[closest_index]
                            history_score[closest_index] = h_score


                        actions = np.random.rand(1, 2).squeeze()*(full_target_edge_map.shape[0] - 1)
                        goal_points[j] = [int(actions[0]), int(actions[1])]

                        
                # ------------------------------------------------------------------
                #### Logical Analysis
                # ------------------------------------------------------------------
                # The current scene is worth exploring and the intelligences are not in Frontier
                for i in range(num_agents):
                    if len(pre_g_points) == 0:
                        break
                    if calculate_distance(cur_goal_points[i], pre_g_points[i]) >= 25 and agent[i].is_Frontier == True:
                        # print(calculate_distance(cur_goal_points[i], pre_g_points[i]))
                        goal_points[i] = pre_g_points[i]

                # Local_Policy = 1
                # Determine the distance, if the distance between two intervals is too short choose a random point for navigation
                for i in range(num_agents):
                    if len(pre_goal_points) > 0 and calculate_distance(pre_goal_points[i], cur_goal_points[i]) <= 2.5:
                        actions = np.random.rand(1, 2).squeeze()*(full_target_edge_map.shape[0] - 1)
                        goal_points[i] = [int(actions[0]), int(actions[1])]
                

                # logging.info(f"pre_g_points: {pre_g_points}")        
                
                logging.info(f"goal_points: {goal_points}")
                pre_g_points = goal_points.copy()
                logging.info("===== Starting local strategy ===== ")
            
            

            for i in range(num_agents):
                if len(target_point) > 0:
                    for j in range(num_agents):
                        goal_points[j] = target_point
                action[i] = agent[i].act(goal_points[i])
                if 'objectnav_hm3d' in args.task_config:
                    _ = agent_GT[i].act(goal_points[i])
                if action[i] == 0:
                    start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs
                    r, c = start_y, start_x
                    start = [int(r * 100.0 / args.map_resolution - gx1),
                            int(c * 100.0 / args.map_resolution - gy1)]
                    start = pu.threshold_poses(start, agent[i].local_map[0, :, :].cpu().numpy().shape)
                    target_point = start.copy()
            # logging.info(f"actions: {action}")
            observations = env.step(action)
            
            # exit(0)
                    
            
                        
            # if count_rotating == 2:
            #     exit(0)
            # ------------------------------------------------------------------

            if args.visualize or args.print_images: 
                if num_agents == 2:
                    vis_ep_dir = '{}/episodes/eps_{}/Agent0_vis'.format(
                        dump_dir, agent[0].episode_n)
                    vis_ep_dir2 = '{}/episodes/eps_{}/Agent1_vis'.format(
                        dump_dir, agent[0].episode_n)
                    if not os.path.exists(vis_ep_dir):
                        os.makedirs(vis_ep_dir)
                    if not os.path.exists(vis_ep_dir2):
                        os.makedirs(vis_ep_dir2)
                    # Legend = cv2.imread("img/legend.png")
                    # height, _ = sem_map.shape[:2]
                    # legend_resized = cv2.resize(Legend, (Legend.shape[1], height))
                    # img_show = np.hstack((sem_map, legend_resized))
                    img_show = observations[0]['rgb'].astype(np.uint8)
                    img_show2 = observations[1]['rgb'].astype(np.uint8)
                    fn = '{}/episodes/eps_{}/Agent0_vis/VisStep-{}.png'.format(
                        dump_dir, agent[0].episode_n,
                        agent[0].l_step)
                    fn2 = '{}/episodes/eps_{}/Agent1_vis/VisStep-{}.png'.format(
                        dump_dir, agent[0].episode_n,
                        agent[0].l_step)
                    # print(fn)
                    cv2.imwrite(fn, img_show)
                    cv2.imwrite(fn2, img_show2)    


                Visualize(args, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                        agent[0].goal_id, visited_vis, full_target_edge_map, Frontiers_dict=None, goal_points=goal_points)
                

                # exit(0)
            

            # logging.info(f"full_map_pred.shape: {full_map_pred.shape}") # [20,480,480] HM-3D

##############################################===Metrics===##############################################

        count_episodes += 1
        # obj_SR['num_'+agent[0].goal_name] += 1
        count_step += agent[0].l_step

        # ------------------------------------------------------------------
        ##### Logging
        # ------------------------------------------------------------------
        log_end = time.time()
        time_elapsed = time.gmtime(log_end - log_start)
        log = " ".join([
            "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
            "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
            "num timesteps {},".format(count_step),
            "FPS {},".format(int(count_step / (log_end - log_start)))
        ]) + '\n'

        # Set SR to 0 if unknown tags are present
        if agent[0].goal_id + 4 > 24:
            log += '==========Unknown Label=========='
            log += '\n'
            for k, v in agg_metrics.items():
                if k == 'multi_Total_SR':
                    for i in range(num_agents):
                        if 'objectnav_hm3d' in args.task_config:
                            if agent[i].Find_Goal and agent_GT[i].Find_Goal:
                                agg_metrics[k] += 1
                                if agg_metrics[k] > count_episodes:
                                    agg_metrics[k] = count_episodes
                                agg_metrics['multi_Navigation_SR'] += 1
                                if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                    agg_metrics['multi_Navigation_SR'] = count_episodes
                                break
                            elif agent[i].Find_Goal and agent_GT[i].Find_Goal == False:
                                agg_metrics[k] += 0
                                agg_metrics['multi_Navigation_SR'] += 1
                                if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                    agg_metrics['multi_Navigation_SR'] = count_episodes
                                break
                        else:
                            if agent[i].Find_Goal:
                                agg_metrics[k] += 1
                                if agg_metrics[k] > count_episodes:
                                    agg_metrics[k] = count_episodes
                                agg_metrics['multi_Navigation_SR'] += 1
                                if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                    agg_metrics['multi_Navigation_SR'] = count_episodes
                                break

            spls = []
            for i in range(num_agents):
                start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs
                r, c = start_y, start_x
                start = [int(r * 100.0 / args.map_resolution - gx1),
                int(c * 100.0 / args.map_resolution - gy1)]
                start = pu.threshold_poses(start, agent[i].local_map[0, :, :].cpu().numpy().shape)
                if 'objectnav_hm3d' in args.task_config:
                    if agent[i].Find_Goal and agent_GT[i].Find_Goal:
                        spl = agent[i].get_spl(success=1,cur_loc=start)
                    else:
                        spl = agent[i].get_spl(success=0,cur_loc=start)
                else:
                    if agent[i].Find_Goal:
                        spl = agent[i].get_spl(success=1,cur_loc=start)
                    else:
                        spl = agent[i].get_spl(success=0,cur_loc=start)
                agg_metrics['multi_SPL'][f'Agent_{i}'] = spl
                agg_metrics['multi_SoftSPL'][f'Agent_{i}'] += spl
                spls.append(spl)
            agg_metrics['SPL'] = max(spls)
            agg_metrics['SoftSPL'] += max(spls)
            for agent_name, SPL in agg_metrics['multi_SPL'].items():
                SoftSPL = agg_metrics['multi_SoftSPL'][agent_name] / count_episodes
                log += f"{agent_name}" + "---SPL: {:.3f}, SoftSPL: {:.3f}".format(SPL, SoftSPL)
                log += '\n'

            log += "multi_Total_SR: {:.3f}, ".format(agg_metrics['multi_Total_SR'] / count_episodes)
            log += "multi_Navigation_SR/SR: {:.0f}/{:.0f}, ".format(agg_metrics['multi_Navigation_SR'], agg_metrics['multi_Total_SR'])
            log += "multi_SPL: {:.3f}, ".format(agg_metrics['SPL'])
            log += "multi_SoftSPL: {:.3f} ".format(agg_metrics['SoftSPL'] / count_episodes)
            log += " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes)
        else:
            # metrics = env.get_metrics()

            for k, v in agg_metrics.items():
                if k == 'multi_Total_SR':
                    for i in range(num_agents):
                        if 'objectnav_hm3d' in args.task_config:
                            if agent[i].Find_Goal and agent_GT[i].Find_Goal:
                                agg_metrics[k] += 1
                                if agg_metrics[k] > count_episodes:
                                    agg_metrics[k] = count_episodes
                                agg_metrics['multi_Navigation_SR'] += 1
                                if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                    agg_metrics['multi_Navigation_SR'] = count_episodes
                                break
                            elif agent[i].Find_Goal and agent_GT[i].Find_Goal == False:
                                agg_metrics[k] += 0
                                agg_metrics['multi_Navigation_SR'] += 1
                                if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                    agg_metrics['multi_Navigation_SR'] = count_episodes
                                break
                        else:
                            if agent[i].Find_Goal:
                                agg_metrics[k] += 1
                                if agg_metrics[k] > count_episodes:
                                    agg_metrics[k] = count_episodes
                                agg_metrics['multi_Navigation_SR'] += 1
                                if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                    agg_metrics['multi_Navigation_SR'] = count_episodes
                                break
            spls = []
            for i in range(num_agents):
                start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs
                r, c = start_y, start_x
                start = [int(r * 100.0 / args.map_resolution - gx1),
                int(c * 100.0 / args.map_resolution - gy1)]
                start = pu.threshold_poses(start, agent[i].local_map[0, :, :].cpu().numpy().shape)
                if 'objectnav_hm3d' in args.task_config:
                    if agent[i].Find_Goal and agent_GT[i].Find_Goal:
                        spl = agent[i].get_spl(success=1,cur_loc=start)
                    else:
                        spl = agent[i].get_spl(success=0,cur_loc=start)
                else:
                    if agent[i].Find_Goal:
                        spl = agent[i].get_spl(success=1,cur_loc=start)
                    else:
                        spl = agent[i].get_spl(success=0,cur_loc=start)
                agg_metrics['multi_SPL'][f'Agent_{i}'] = spl
                agg_metrics['multi_SoftSPL'][f'Agent_{i}'] += spl
                spls.append(spl)
            agg_metrics['SPL'] = max(spls)
            agg_metrics['SoftSPL'] += max(spls)
            for agent_name, SPL in agg_metrics['multi_SPL'].items():
                SoftSPL = agg_metrics['multi_SoftSPL'][agent_name] / count_episodes
                log += f"{agent_name}" + "---SPL: {:.3f}, SoftSPL: {:.3f}".format(SPL, SoftSPL)
                log += '\n'

            log += "multi_Total_SR: {:.3f}, ".format(agg_metrics['multi_Total_SR'] / count_episodes)
            log += "multi_Navigation_SR/SR: {:.0f}/{:.0f}, ".format(agg_metrics['multi_Navigation_SR'], agg_metrics['multi_Total_SR'])
            log += "multi_SPL: {:.3f}, ".format(agg_metrics['SPL'])
            log += "multi_SoftSPL: {:.3f} ".format(agg_metrics['SoftSPL'] / count_episodes)
            log += " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes)
        # log += "Total usage: " + str(sum(total_usage)) + ", average usage: " + str(np.mean(total_usage))
        # print(log)
        logging.info(log)
        fn = '{}/MCoCoNav_history.log'.format(log_dir)
        if count_episodes == 1:
            with open(fn,'w', encoding='utf-8') as f:
                f.write(log)
                f.write('\n')
        else:
            with open(fn,'a', encoding='utf-8') as f:
                f.write(log)
                f.write('\n')
        # ------------------------------------------------------------------


    # avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

    # return avg_metrics
    

if __name__ == "__main__":
    main()