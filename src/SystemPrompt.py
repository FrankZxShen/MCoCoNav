import re


### Perception
Caption_Prompt = '''
You are a knowledgeable and skilled expert in indoor navigation planning. You are planning the navigation of multiple robots in an indoor environment. Imagine you are standing in the center of a room. Your task is to describe the entire scene around you with as much detail as possible. Focus on the placement and relationships of the objects in the room relative to your current viewpoint. If you see objects such as a chair, bed, plant, toilet, TV monitor, sofa, or other common items, make sure to clearly detail their positions and how they are spatially related to you and each other.

For example, consider where the objects are placed:

- Are they to your left, right, in front of you, or behind you?
- Are they on the floor, mounted on the wall, or placed on top of another piece of furniture?
- How close are they to you or to each other?

Use the following format to structure your response:

------

**Scene Description:**

1. **Object List:**
   - [Other objects if any]
2. **Spatial Relationships:**
   - **[Describe other objects if any]**
3. **Additional Context:**
   - Detail any significant aspects of the lighting, colors, textures, and overall ambiance of the room.
   - Mention any items on top of furniture or decorations on the walls.

------
'''

Perception_System_Prompt = """
Based on your description, you need to determine whether the current scene is worth exploring for the robot, based on the provided information. Follow the guidelines below to make your decision clearly and explicitly.

**Information Format:**

- **Target of Navigation:** Format - <Name>
- **Scene Object (Object Detection):** Format - <Name, Confidence Score>

**Decision Criteria:**

1. **Examine the Relationship Between the Target Object and the Scene:**
   - If the object detection model predicts the presence of the target object with a high probability (e.g., >85%), determine if the scene is worth exploring.
2. **Analyze the Scene Context:**
   - Assess the context of the scene, including ceilings, walls, floors, or windows.
   - Use your knowledge of typical object locations to evaluate the likelihood of the target object being present (e.g., beds are usually found in bedrooms, TVs in living rooms).
3. **Considering Object Proximity and Context:**
   - Evaluate the proximity of the target object to the scene in the image.
   - Ensure your judgment does not violate the high probability criterion from point (1). For instance, a bathtub is unlikely in a bedroom, but the presence of a door could indicate the target object might be nearby.
4. **Disregard Generic Objects:**
   - Ignore objects commonly found in various rooms (like light switches and doors) as they do not provide strong evidence for the target object’s presence.

------
**Output Format:**
 Your output should be a simple "[Yes, No]" statement indicating whether the scene is worth exploring based on the given criteria.


"""


### Judgment
FN_System_Prompt = '''
You are a knowledgeable and skilled expert in indoor navigation planning. Based on the current top-down semantic map of an indoor environment provided, you will see various points marked as either 'Frontier Points' or 'Historical Observation Points'. Frontier points represent unexplored areas that the robot has yet to navigate, while Historical Observation Points signify areas the robot has previously explored or observed.

**Information Format:**
- **Your Navigation Target:** Format - `<Name>`
- **Scene Objects:** Format - `<Name: [Coordinates: <(x1, y1), (x2, y2)...>]>`
- **Frontier Points (The black dots and corresponding black uppercase letters on the image):** Format - `<Name: [Coordinates: <(x1, y1), (x2, y2)...>]>`
- **Historical Observation Points (The green dots and corresponding green lowercase letters on the image):** Format - `<Name: [Coordinates: <[x1, y1], [x2, y2]...>]>`
- **Your location (The red arrow):** Format - `<(x, y)>`
- **Previous Movement:** Format - `<(x, y)>`

Your goal is to guide the robot for exploration purposes based on the relationship between different objects, the structure of the explored area, the robot's position, the proximity of exploration points, and the direction of previous movement. Consider the following factors when making your decision:

1. **Level of Exploration:**
   - If there are a very small number of Historical Observation Points and a large number of gaps in the semantic map, prefer exploring Frontier Points.

2. **Explorability Worthiness:**
   - Scenes that are worth exploring ({ISWORTH}) are usually more likely to be explored by choosing Frontier Points.

3. **Proximity and Accessibility:**
   - Evaluate how Your location relates to surrounding obstacles. Frontier Points or Historical Observation Points that are close and free of obstacles tend to have higher exploration priority.

4. **Relationship Between Location and Previous Movement:**
   - If Your location is too close to Previous Movement, it may indicate a collision trap. In such cases, prefer to explore Historical Observation Points that are close to Your location.

**Decision Format:**
Your recommendation should be a simple "[Yes, No]" statement:
- **Yes:** Explore a Frontier Point.
- **No:** Revisit a Historical Observation Point.

**Example:**

---

**Your Navigation Target**: `TargetObject`

**Scene Objects**:
- `Chair: [Coordinates: (3, 4), (2, 3)]`
- `Table: [Coordinates: (5, 6), (6, 7)]`

**Frontier Points (The black dots and corresponding black uppercase letters on the image)**:
- `A: [Coordinates: (8, 9)]`
- `B: [Coordinates: (10, 11)]`

**Historical Observation Points (The green dots and corresponding green lowercase letters on the image)**:
- `a: [Coordinates: (1, 2)]`
- `b: [Coordinates: (3, 5)]`

**Your location (The red dot)**: `(4, 4)`

**Previous Movement**: `(2, 2)`


**Decision:** Yes

---

Now, begin your analysis with the provided scene image information.

- **Your Navigation Target:** `{TARGET}`
- **Scene Objects:** `{SCENE_OBJECTS}`
- **Frontier Points (The black dots and corresponding black uppercase letters on the image):** `{FRONTIERS_RESULTS}`
- **Historical Observation Points (The green dots and corresponding green lowercase letters on the image):** `{HISTORY_NODES}`
- **Your location (The red arrow):** `{CUR_LOCATION}`
- **Previous Movement:** `{PRE}`

Would you recommend:
Yes) Exploring a frontier point? If so, answer ONLY Yes.
No) Revisiting a historical observation point? If so, answer ONLY No.

---
'''

### Decision CoT1
Single_Agent_Decision_Prompt_Frontier1 = '''
You are a knowledgeable and skilled expert in indoor navigation planning. Based on the current top-down semantic map of an indoor environment provided, you will see various black points marked as 'Frontier Points'. Frontier points represent unexplored areas that the robot has yet to navigate.

**Information Format:**

- **Your Navigation Target:** <{TARGET}>
- **Scene Objects:** <{SCENE_OBJECTS}>
- **Frontier Points (The black dots and corresponding black uppercase letters on the image):** <{FRONTIERS_RESULTS}>
- **Your location (The red arrow):** <{CUR_LOCATION}>
- **Previous Movement:** <{PRE}>
- **Your location facing (the direction indicated by the red arrow) scene information:** <{SCENE_INFORMATION}>

Your task is to guide the robot for exploration purposes based on the relationship between different objects, the structure of the explored area, the robot's position, the proximity of the exploration point, and the direction of previous movement. Consider the following factors when making your decision:

1. **Proximity and Accessibility:**
   - Evaluate how Your location relates to surrounding obstacles. Frontier Points that are close and free of obstacles tend to have higher exploration priority.
2. **Relationship Between Location and Previous Movement:**
   - If Your location is too close to Previous Movement, it may indicate the robot is entering a collision trap. Prefer to explore Frontier Points that are farther from Your location.
3. **Exploration Consistency:**
   - Minimize frequent switches between Frontier Points. The robot should maintain its exploration direction unless an efficient switch is evident.
4. **Target-Oriented Exploration:**
   - If there are accessible Frontier Points in the direction of Your location (red arrow), and the scenario information includes Your Navigation Target, give the highest priority to exploring these Frontier Points without violating point (1).

**Your Recommendation:**
 Describe the information around each Frontier Point in the diagram and indicate whether you are likely to choose it for exploration based on the provided criteria.

------

**Example:**

**Your Navigation Target:** TargetObject

**Scene Objects:**

- Chair: [Coordinates: (3, 4), (2, 3)]
- Table: [Coordinates: (5, 6), (6, 7)]

**Frontier Points (The black dots and corresponding black uppercase letters on the image):**

- A: [Coordinates: (8, 9)]
- B: [Coordinates: (10, 11)]

**Your location (The red arrow):** (4, 4)

**Previous Movement:** (2, 2)

**Analysis:**

1. **Proximity and Accessibility:** Frontier Point A at (8, 9) is relatively close and free of obstacles.
2. **Relationship Between Location and Previous Movement:** The location (4, 4) is moving away from the previous movement at (2, 2), avoiding collision traps.
3. **Exploration Consistency:** Exploring Frontier Point A maintains the current exploration direction.
4. **Target-Oriented Exploration:** The scenario information does not include a specific Navigation Target, so this criterion is not applicable.

**Recommendation:** I am likely to choose Frontier Point A for exploration.

------

Now, begin your analysis with the provided scene information:

- **Your Navigation Target:** <{TARGET}>
- **Scene Objects:** <{SCENE_OBJECTS}>
- **Frontier Points (The black dots and corresponding black uppercase letters on the image):** <{FRONTIERS_RESULTS}>
- **Your location (The red arrow):** <{CUR_LOCATION}>
- **Previous Movement:** <{PRE}>
- **Your location facing (the direction indicated by the red arrow) scene information:** <{SCENE_INFORMATION}>

Describe the information around each Frontier Point and indicate whether you are likely to choose it for exploration.

------

Explanation Ends.
**Output Format:**
Your choice MUST in 'A', 'B', 'C', 'D'(if exist) **WITHOUT ANY OTHER DESCRIPTION**. You don't need to add punctuation at the end. You don't need to add space at the beginning. 
'''
### Decision CoT2



### Decision 2
Single_Agent_Decision_Prompt_History = '''
Your choice is 'Historical Observation Points'. So you will now re-explore historical nodes in this indoor environment. 

The green dots and corresponding green lowercase letters on the left image represent the Historical Observation Points awaiting your exploration: 
{HISTORY_NODES}
where the confidence level following each green lowercase letter represents the exploration likelihood that this point was recorded. Point with higher confidence have higher exploration priority.
The Arrow is Your Location: {CUR_LOCATION}
Previous Movement: {PRE}

Your goal is to find the {TARGET}. You need to consider the following factors:
(1) Consider the proximity and accessibility of the points. You need to consider the proximity and accessibility of the front and your previous movement. Points that are closer and without barriers tend to have a higher exploration priority. 
(2) Consider the objects in the scene. Use your knowledge of the location of typical objects (e.g., the bed in your bedroom, the TV in your living room) to assess the likelihood of finding the target object.
(3) Minimize frequent switches between points. Use centroid for frontier selection. You should maintain its exploration direction unless an efficient switch is evident.
You need to comprehensively consider the relevance of the scene image and the top-down semantic map, and choose the navigation point for the next timestep based on their relationship with your navigation goal. 

Your choice must be in 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' and 26 other lowercase letters representing historical observation points  **WITHOUT ANY OTHER DESCRIPTION**. You don't need to add punctuation at the end. You don't need to add space at the beginning. 
'''




# '''
# If the current scene is worth exploring, you should prioritize exploring frontier points. If the current scene is not worth exploring, you should consider revisiting historical observation points.
# Your Answer MUST in 'Yes', 'No' **WITHOUT ANY OTHER DESCRIPTION**. You don't need to add punctuation at the end. You don't need to add space at the beginning. 
# '''



def form_prompt_for_PerceptionVLM(target, objs, yolo) -> str:

    object_detection = ''
    if yolo == 'yolov9':
        if len(objs) < 1:
            object_detection = 'No Detections'
        else:
            for item in objs:
                name, confidence, coords = item
                # coord_pairs = coords[0].split(',')
                # coord1 = coord_pairs[0], coord_pairs[1]
                # coord2 = coord_pairs[2], coord_pairs[3]
                detection = f"  {name}, {confidence}\n"
                object_detection += detection
    else:
        # print(objs)
        if len(objs) < 1:
            object_detection = 'No Detections'
        else:
            for name, confidence in objs.items():
                detection = f"  {name}, {confidence}\n"
                object_detection += detection

    # semantic_segmentation = "\n".join([f"{key}: " + ", ".join([f"<" + ", ".join([f"{value[i][j][0][0], value[i][j][0][1]}" 
    #                         for j in range(len(value[i]))]) + f">"
    #                         for i in range(len(value))])
    #                         for key, value in agents_seg_list.items()]) + "\n"

    Perception_Template = """
- **Target of navigation:** {TARGET}

- Scene object (Object Detection): 
{OBJECT_DETECTION}

**Decision:**
"""
    User_Prompt = Perception_Template.format(
                        TARGET = target,
                        OBJECT_DETECTION = object_detection,
                    )
    User_Prompt = Perception_System_Prompt + User_Prompt



    return Caption_Prompt, User_Prompt

def form_prompt_for_FN(target, agents_seg_list, Perception_PR, pre_goal_point, Frontier_list, cur_location, History_nodes) -> str:

    def convert_entry(entry):
        centroid_str = entry.split('centroid: ')[1].split(', number: ')[0]
        number_str = entry.split('number: ')[1][:-1]
        return f'[Coordinates: {centroid_str}]'
    
    if pre_goal_point == []:
        pre_goal_point = 'No Movements'
    
    semantic_segmentation = "\n".join([f"{key}: " + ", ".join([f"<" + ", ".join([f"{value[i][j][0][0], value[i][j][0][1]}" 
                            for j in range(len(value[i]))]) + f">"
                            for i in range(len(value))])
                            for key, value in agents_seg_list.items()]) + "\n"
    Frontiers = []
    for i, key in enumerate(Frontier_list):
        entry = Frontier_list[key]
        converted = convert_entry(entry)
        Frontiers.append(f'{chr(65+i)}: {converted}') 
    
    Frontiers_results = ''
    for i in range(len(Frontiers)):
        Frontiers_results += Frontiers[i]
        Frontiers_results += '\n'

    if len(History_nodes) > 0:
        History = []
        for i in range(len(History_nodes)):
            History.append(f'{chr(ord("a")+i)}: [Coordinates: {History_nodes[i]}]') 
        His_results = ''
        for i in range(len(History)):
            His_results += History[i]
            His_results += '\n'
    else:
        His_results = 'No historical observation points'
    

    if Perception_PR[0] >= 0.50:
        isworth = 'The current scene is worth exploring.'
    else:
        isworth = 'The current scene is not worth exploring.'
    User_Prompt1 = FN_System_Prompt.format(
                        SCENE_OBJECTS = semantic_segmentation,
                        TARGET = target,
                        ISWORTH = isworth,
                        FRONTIERS_RESULTS = Frontiers_results,
                        HISTORY_NODES = His_results,
                        CUR_LOCATION = cur_location,
                        PRE = pre_goal_point, 
                    )


    return User_Prompt1

def form_prompt_for_DecisionVLM_Frontier(Scene_Information, agents_seg_list, pre_goal_point, target, cur_location, Frontier_list) -> str:

    def convert_entry(entry):
        centroid_str = entry.split('centroid: ')[1].split(', number: ')[0]
        number_str = entry.split('number: ')[1][:-1]
        return f'[Coordinates: {centroid_str}]'

    if pre_goal_point == []:
        pre_goal_point = 'No Movements'

    Frontiers = []
    for i, key in enumerate(Frontier_list):
        entry = Frontier_list[key]
        converted = convert_entry(entry)
        Frontiers.append(f'{chr(65+i)}: {converted}') 
    
    Frontiers_results = ''
    for i in range(len(Frontiers)):
        Frontiers_results += Frontiers[i]
        Frontiers_results += '\n'
    
    semantic_segmentation = "\n".join([f"{key}: " + ", ".join([f"<" + ", ".join([f"{value[i][j][0][0], value[i][j][0][1]}" 
                            for j in range(len(value[i]))]) + f">"
                            for i in range(len(value))])
                            for key, value in agents_seg_list.items()]) + "\n"
    
    User_Prompt = Single_Agent_Decision_Prompt_Frontier1.format(
        SCENE_INFORMATION = Scene_Information,
        SCENE_OBJECTS = semantic_segmentation,
        FRONTIERS_RESULTS = Frontiers_results,
        TARGET = target,
        CUR_LOCATION = cur_location,
        PRE = pre_goal_point, 
    )
    

    return User_Prompt

def form_prompt_for_DecisionVLM_History(pre_goal_point, target, cur_location, confidence, History_nodes) -> str:
    if all(num == 0 for num in pre_goal_point):
        pre_goal_point = 'No Movements'
    if len(History_nodes) > 0:
        History = []
        for i in range(len(History_nodes)):
            History.append(f'{chr(ord("a") + i)}: [Coordinates: {History_nodes[i]}; Confidence: {confidence[i]}]') 
        His_results = ''
        for i in range(len(History)):
            His_results += History[i]
            His_results += '\n'
    else:
        His_results = 'No historical observation points'
    
    User_Prompt = Single_Agent_Decision_Prompt_History.format(
        TARGET = target,
        CUR_LOCATION = cur_location,
        PRE = pre_goal_point, 
        HISTORY_NODES = His_results
    )
    

    return User_Prompt







def extract_scene_image_description_results(text):
    pattern = re.compile(r'Scene image description module: (Yes|No)',re.I)
    matches = pattern.findall(text)
    return matches
def extract_scene_object_detection_results(text):
    pattern = re.compile(r'Scene object detection module: (Yes|No)',re.I)
    matches = pattern.findall(text)
    return matches
def extract_scenario_exploration_analysis_results(text):
    pattern = re.compile(r'Scenario exploration analysis module: (Yes|No)',re.I)
    matches = pattern.findall(text)
    return matches
# def form_prompt_for_DecisionVLM_Meta() -> str:
#     return Meta_Agent_Decision_Prompt



def contains_yes_or_no(VLM_Pred: str) -> str:
    if "Yes" in VLM_Pred:
        return "Yes"
    elif "No" in VLM_Pred:
        return "No"
    else:
        return "Neither"

def Perception_weight_decision(VLM_Rel: list, VLM_Pred: str) -> str:
    b_decision = contains_yes_or_no(VLM_Pred)
    if b_decision == "Neither":
        return b_decision
    
    x, y = VLM_Rel
    
    if b_decision == "Yes":
        weighted_yes_prob = x 
        weighted_no_prob = y * (1 - x)
    else:  # b_decision == "No"
        weighted_yes_prob = x * (1 - y)
        weighted_no_prob = y 
    
    total_prob = weighted_yes_prob + weighted_no_prob
    if total_prob == 0:  # 避免划分零的问题
        return b_decision
    
    weighted_yes_prob /= total_prob
    weighted_no_prob /= total_prob
    
    return weighted_yes_prob, weighted_no_prob

def contains_decision(VLM_Pred: str) -> str:
    if "A" in VLM_Pred:
        return "A"
    elif "B" in VLM_Pred:
        return "B"
    elif "C" in VLM_Pred:
        return "C"
    elif "D" in VLM_Pred:
        return "D"
    else:
        return "Neither"

def Perception_weight_decision4(VLM_Rel: list, VLM_Pred: str) -> str:
    decision = contains_decision(VLM_Pred)
    if decision == "Neither":
        return decision
    
    assert len(VLM_Rel) == 4, "VLM_Rel must contain weights for four decisions (A, B, C, D)."

    weights = {
        "A": VLM_Rel[0],
        "B": VLM_Rel[1],
        "C": VLM_Rel[2],
        "D": VLM_Rel[3]
    }

    weights[decision] = weights[decision] 
    
    total_weight = sum(VLM_Rel)
    weighted_probs = {key: val/total_weight for key, val in weights.items()}
    
    if total_weight == 0:  
        return "Invalid weights provided."
    
    return weighted_probs["A"],weighted_probs["B"],weighted_probs["C"],weighted_probs["D"]


def contains_decision26(VLM_Pred: str) -> str:
    for char in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
        if char in VLM_Pred:
            return char
    return "Neither"

def Perception_weight_decision26(VLM_Rel: list, VLM_Pred: str) -> dict:
    decision = contains_decision26(VLM_Pred)
    if decision == "Neither":
        return {"Neither": 1.0}
    
    assert len(VLM_Rel) == 26, "VLM_Rel must contain weights for 26 decisions (1-26)."
    
    weights = {chr(ord("a")+i): VLM_Rel[i] for i in range(26)}  # Assign weights to each number
    
    weights[decision] = weights[decision] * 26

    total_weight = sum(VLM_Rel)
    if total_weight == 0:
        return {"Invalid weights provided.": 1.0}

    weighted_probs = [val/total_weight for _, val in weights.items()]
    
    return weighted_probs
