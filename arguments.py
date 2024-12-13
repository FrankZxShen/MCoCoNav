import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description='Multi-Agent-Semantic-Exploration')

    # General Arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--auto_gpu_config', type=int, default=0)
    parser.add_argument('--total_num_scenes', type=str, default="auto")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--sim_gpu_id", type=int, default=0,
                        help="gpu id on which scenes are loaded")
    parser.add_argument("--sem_gpu_id", type=int, default=0,
                        help="""gpu id for semantic model,""")

    # Logging, loading models, visualization
    parser.add_argument('--use_sam', action='store_true', default=False,
                        help='use rednet or sam')
    parser.add_argument('--yolo', type=str, default='yolov10',
                        help='yolo type')
    parser.add_argument('--yolo_weights', type=str, default='detect/yolov10m',
                        help='yolo_weights path')
    parser.add_argument('--log_interval', type=int, default=10,
                        help="""log interval, one log per n updates
                                (default: 10) """)
    parser.add_argument('--save_interval', type=int, default=1,
                        help="""save interval""")
    parser.add_argument('-d', '--dump_location', type=str, default="./MCoCoNav_EXP/",
                        help='path to dump models and log (default: ./MCoCoNav_EXP/)')
    parser.add_argument('--exp_name', type=str, default="exp1",
                        help='experiment name (default: exp1)')
    parser.add_argument('--save_periodic', type=int, default=500000,
                        help='Model save frequency in number of updates')
    parser.add_argument('-v', '--visualize', type=int, default=0,
                        help="""1: Render the observation and
                                   the predicted semantic map,
                                2: Render the observation with semantic
                                   predictions and the predicted semantic map
                                (default: 0)""")
    parser.add_argument('--print_images', type=int, default=1,
                        help='1: save visualization as images')

    # Environment, dataset and episode specifications
    parser.add_argument('-efw', '--env_frame_width', type=int, default=640,
                        help='Frame width (default:640)')
    parser.add_argument('-efh', '--env_frame_height', type=int, default=480,
                        help='Frame height (default:480)')
    parser.add_argument('-fw', '--frame_width', type=int, default=160,
                        help='Frame width (default:160)')
    parser.add_argument('-fh', '--frame_height', type=int, default=120,
                        help='Frame height (default:120)')
    parser.add_argument('-el', '--max_episode_length', type=int, default=500,
                        help="""Maximum episode length""")
    parser.add_argument("--task_config", type=str,
                        default="tasks/multi_objectnav_hm3d_1agent.yaml",
                        help="path to config yaml containing task information")
    parser.add_argument("--split", type=str, default="train",
                        help="dataset split (train | val | val_mini) ")
    parser.add_argument('--camera_height', type=float, default=0.88,
                        help="agent camera height in metres (default:0.88)")
    parser.add_argument('--hfov', type=float, default=79.0,
                        help="horizontal field of view in degrees (default:79)")
    parser.add_argument('--turn_angle', type=float, default=30,
                        help="Agent turn angle in degrees")
    parser.add_argument('--init_angle', type=float, default=0,
                        help="Agent init angle in degrees")
    parser.add_argument('--min_depth', type=float, default=0.5,
                        help="Minimum depth for depth sensor in meters")
    parser.add_argument('--max_depth', type=float, default=5.0,
                        help="Maximum depth for depth sensor in meters")
    parser.add_argument('--success_dist', type=float, default=1.0,
                        help="success distance threshold in meters")
    parser.add_argument('--floor_thr', type=int, default=50,
                        help="floor threshold in cm")
    parser.add_argument('--min_d', type=float, default=1.5,
                        help="min distance to goal during training in meters")
    parser.add_argument('--max_d', type=float, default=100.0,
                        help="max distance to goal during training in meters")

    # Model Hyperparameters
    parser.add_argument('--agent', type=str, default="sem_exp")
    parser.add_argument('--num_global_steps', type=int, default=20,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--num_local_steps', type=int, default=25,
                        help="""Number of steps the local policy
                                between each global step""")
    parser.add_argument('--num_sem_categories', type=int, default=16)
    parser.add_argument('--sem_pred_prob_thr', type=float, default=0.9,
                        help="Semantic prediction confidence threshold")

    # Mapping
    parser.add_argument('--global_downscaling', type=int, default=1)
    parser.add_argument('--vision_range', type=int, default=100)
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--du_scale', type=int, default=1)
    parser.add_argument('--map_size_cm', type=int, default=2400)
    parser.add_argument('--cat_pred_threshold', type=float, default=5.0)
    parser.add_argument('--map_pred_threshold', type=float, default=1.0)
    parser.add_argument('--exp_pred_threshold', type=float, default=1.0)
    parser.add_argument('--collision_threshold', type=float, default=0.10)

    # train_se_frontier
    parser.add_argument('--use_gtsem', type=int, default=0)
    parser.add_argument('--num_agents', type=int, default=1)
    parser.add_argument('--train_se_f', type=int, default=0)
    parser.add_argument('--load_se_edge', type=str, default="0",
                        help="""model path to load,
                                0 to not reload (default: 0)""")
    
    parser.add_argument('--boundary_coeff', type=int, default=4)
    parser.add_argument('--text_threshold', type=float, default=0.55)
    parser.add_argument('--prompt_type', type=str, default="scoring", 
                        choices=['deterministic', 'scoring'])
    parser.add_argument('--tag_freq', type=int, default=0)
    parser.add_argument('--random_sample', action='store_true', default=False) # overrides --start_episode
    
    # Camera, image
    parser.add_argument('--tsdf_grid_size', type=float, default=0.1)
    parser.add_argument('--margin_w_ratio', type=float, default=0.25)
    parser.add_argument('--margin_h_ratio', type=float, default=0.6)

    # Navigation setup
    parser.add_argument('--init_clearance', type=float, default=0.5)
    parser.add_argument('--max_step_room_size_ratio', type=int, default=3)
    parser.add_argument('--black_pixel_ratio', type=float, default=0.5)
    parser.add_argument('--min_random_init_steps', type=int, default=2)

    # VLM setup
    parser.add_argument('--base_url', type=str, default="http://127.0.0.1:31511")
    parser.add_argument('--hf_token', type=bool, default=None)
    parser.add_argument('--save_obs', type=bool, default=True)
    parser.add_argument('--save_freq', type=int, default=10)

    parser.add_argument('--use_active', type=bool, default=True)
    parser.add_argument('--use_lsv', type=bool, default=True)
    parser.add_argument('--use_gsv', type=bool, default=True)
    parser.add_argument('--gsv_T', type=float, default=0.5)
    parser.add_argument('--gsv_F', type=float, default=3.0)
    

    # Visual prompt
    parser.add_argument("--cluster_threshold", type=float, default=1.0, help="Cluster threshold.")
    parser.add_argument("--num_prompt_points", type=int, default=3, help="Number of prompt points.")
    parser.add_argument("--num_max_unoccupied", type=int, default=300, help="Maximum number of unoccupied points.")
    parser.add_argument("--min_points_for_clustering", type=int, default=3, help="Minimum points for clustering.")
    parser.add_argument("--point_min_dist", type=int, default=2, help="Minimum distance between points.")
    parser.add_argument("--point_max_dist", type=int, default=10, help="Maximum distance between points.")
    parser.add_argument("--cam_offset", type=float, default=0.6, help="Camera offset.")
    parser.add_argument("--min_num_prompt_points", type=int, default=1, help="Minimum number of prompt points.")
    parser.add_argument("--circle_radius", type=int, default=18, help="Circle radius for visualization.")

    # Semantic Map Planner
    parser.add_argument("--dist_T", type=int, default=10)
    parser.add_argument("--unexplored_T", type=float, default=0.2)
    parser.add_argument("--unoccupied_T", type=float, default=2.0)
    parser.add_argument("--val_T", type=float, default=0.5)
    parser.add_argument("--val_dir_T", type=float, default=0.5)
    parser.add_argument("--max_val_check_frontier", type=int, default=3)
    parser.add_argument("--smooth_sigma", type=int, default=5)
    parser.add_argument("--eps_planner", type=int, default=1)
    parser.add_argument("--min_dist_from_cur", type=float, default=0.5)
    parser.add_argument("--max_dist_from_cur", type=float, default=3.0)
    parser.add_argument("--frontier_spacing", type=float, default=1.5)
    parser.add_argument("--frontier_min_neighbors", type=int, default=3)
    parser.add_argument("--frontier_max_neighbors", type=int, default=4)
    parser.add_argument("--max_unexplored_check_frontier", type=int, default=3)
    parser.add_argument("--max_unoccupied_check_frontier", type=int, default=1)

                                
    # for sem exp
    parser.add_argument('--lr', type=float, default=2.5e-5,
                        help='learning rate (default: 2.5e-5)')
    parser.add_argument('--global_hidden_size', type=int, default=256,
                        help='global_hidden_size')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RL Optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RL Optimizer alpha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use_gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy_coef', type=float, default=0.001,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--ppo_epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num_mini_batch', type=str, default=2,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--use_recurrent_global', type=int, default=0,
                        help='use a recurrent global policy')
    parser.add_argument('--reward_coeff', type=float, default=0.1,
                        help="Object goal reward coefficient")
    parser.add_argument('--intrinsic_rew_coeff', type=float, default=0.05,
                        help="intrinsic exploration reward coefficient")

    parser.add_argument('--load', type=str, default="0",
                    help="""model path to load,
                            0 to not reload (default: 0)""")
    # parse arguments
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
