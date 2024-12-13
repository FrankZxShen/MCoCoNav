coco_categories = [0, 3, 2, 4, 5, 1]
coco_categories_hm3d2mp3d = [0, 6, 8, 10, 13, 5]
category_to_id = [
        "chair",
        "bed",
        "plant",
        "toilet",
        "tv_monitor",
        "sofa"
]

category_to_id_gibson = [
        "chair",
        "couch",
        "potted plant",
        "bed",
        "toilet",
        "tv"
]

category_to_id_mp3d = [
    'chair', #g
    'table', #g
    'picture', #b
    'cabinet', # in resnet
    'cushion', # in resnet
    'sofa', #g
    'bed', #g
    'chest_of_drawers', #b in resnet
    'plant', #g
    'sink', #g
    'toilet', #g
    'stool', #b
    'towel', #b in resnet
    'tv_monitor', #g
    'shower', #b
    'bathtub', #b in resnet
    'counter', #b isn't this table?
    'fireplace',
    'gym_equipment',
    'seating',
    'clothes', # in resnet
    'background'
]

mp3d_category_id = {
    'void': 1,
    'chair': 2,
    'sofa': 3,
    'plant': 4,
    'bed': 5,
    'toilet': 6,
    'tv_monitor': 7,
    'table': 8,
    'refrigerator': 9,
    'sink': 10,
    'stairs': 11,
    'fireplace': 12
}

# mp_categories_mapping = [4, 11, 15, 12, 19, 23, 6, 7, 15, 38, 40, 28, 29, 8, 17]

mp_categories_mapping = [4, 11, 15, 12, 19, 23, 26, 24, 28, 38, 21, 16, 14, 6, 16]
mp_categories_mapping21 = [4, 6, 7, 8, 9, 11, 12, 14, 15, 16, 19, 20, 21, 23, 24, 26, 27, 28, 34, 35, 39]

hm3d_category = [
        "chair",
        "couch",
        "plant",
        "bed",
        "toilet",
        "tv",
        "bathtub",
        "shower",
        "fireplace",
        "appliances",
        "towel",
        "sink",
        "chest_of_drawers",
        "table",
        "stairs"
]
mp3d_category = [
    'chair', #g
    'table', #g
    'picture', #b
    'cabinet', # in resnet
    'cushion', # in resnet
    'couch', #g
    'bed', #g
    'drawer', #b in resnet
    'plant', #g
    'sink', #g
    'toilet', #g
    'stool', #b
    'towel', #b in resnet
    'tv', #g
    'shower', #b
    'bathtub', #b in resnet
    'counter', #b isn't this table?
    'fireplace',
    'gym equipment',
    'seating',
    'clothes', # in resnet
    'background'
]

mp3d_habitat_labels = {
            'chair': 0, #g
            'table': 1, #g
            'picture':2, #b
            'cabinet':3, # in resnet
            'cushion':4, # in resnet
            'sofa':5, #g
            'bed':6, #g
            'chest_of_drawers':7, #b in resnet
            'plant':8, #g
            'sink':9, #g
            'toilet':10, #g
            'stool':11, #b
            'towel':12, #b in resnet
            'tv_monitor':13, #g
            'shower':14, #b
            'bathtub':15, #b in resnet
            'counter':16, #b isn't this table?
            'fireplace':17,
            'gym_equipment':18,
            'seating':19,
            'clothes':20, # in resnet
            'background': 21
}

gibson_coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "dining-table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14
}

object_category = [
    'chair', #g
    'table', #g
    'picture', #b
    'cabinet', # in resnet
    'cushion', # in resnet
    'sofa', #g
    'bed', #g
    'chest_of_drawers', #b in resnet
    'plant', #g
    'sink', #g
    'toilet', #g
    'stool', #b
    'towel', #b in resnet
    'tv_monitor', #g
    'shower', #b
    'bathtub', #b in resnet
    'counter', #b isn't this table?
    'fireplace',
    'gym_equipment',
    'seating',
    'clothes', # in resnet
    'background'
]

object_category_copy = [
    # --- goal categories (DO NOT CHANGE)
    "chair",
    "bed",
    "plant",
    "toilet",
    "tv",
    "couch",
    # --- add as many categories in between as you wish, which are used for LLM reasoning
    # "table",
    "desk",
    "refrigerator",
    "sink",
    "bathtub",
    "shower",
    "towel",
    "painting",
    "trashcan",
    # --- stairs must be put here (DO NOT CHANGE)
    "stairs",
    # --- void category (DO NOT CHANGE)
    "void"
]

coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}

color_palette = [
    1.0, 1.0, 1.0,
    0.6, 0.6, 0.6,
    0.95, 0.95, 0.95,
    0.96, 0.36, 0.26,
    0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
    0.9400000000000001, 0.7818, 0.66,
    0.9400000000000001, 0.8868, 0.66,
    0.8882000000000001, 0.9400000000000001, 0.66,
    0.7832000000000001, 0.9400000000000001, 0.66,
    0.6782000000000001, 0.9400000000000001, 0.66,
    0.66, 0.9400000000000001, 0.7468000000000001,
    0.66, 0.9400000000000001, 0.8518000000000001,
    0.66, 0.9232, 0.9400000000000001,
    0.66, 0.8182, 0.9400000000000001,
    0.66, 0.7132, 0.9400000000000001,
    0.7117999999999999, 0.66, 0.9400000000000001,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999,
    # NEW COLORS
    0.66, 0.9400000000000001, 0.9531999999999998,
    0.7600000000000001, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.9531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999,
    0.66, 0.9400000000000001, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.9581999999999998,
    0.8882000000000001, 0.9400000000000001, 0.9531999999999998,
    0.7832000000000001, 0.9400000000000001, 0.9581999999999998,
    0.6782000000000001, 0.9400000000000001, 0.9531999999999998,
    0.66, 0.9400000000000001, 0.7618000000000001,
    0.66, 0.9400000000000001, 0.9661999999999998,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9661999999999998,
    0.9400000000000001, 0.66, 0.7668000000000001,
    0.66, 0.9661999999999998, 0.9400000000000001,
    0.7832000000000001, 0.9661999999999998, 0.66,
    0.9400000000000001, 0.8531999999999998, 0.66,
    0.66, 0.9661999999999998, 0.9681999999999998,
    0.8882000000000001, 0.66, 0.9661999999999998,
    0.66, 0.7657999999999998, 0.9661999999999998,
]
