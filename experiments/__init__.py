import cv2 
import json
import numpy as np
import importlib
import os

def load_experiment(DIR):
    # Define a mapping of directories to experiment modules
    experiment_modules = {
        "scissor": "experiments.scissor.experiment.ScissorExperiment",
        "wood": "experiments.wood.experiment.WoodExperiment",
        "pencil_sharpener": "experiments.pencile_sharpener.experiment.SharpenerExperiment"
    }
    
    # Extract experiment name from the directory
    experiment_name = os.path.basename(DIR)
    
    # Load the corresponding experiment module dynamically
    if experiment_name not in experiment_modules:
        raise ValueError(f"Unknown experiment directory '{experiment_name}'")
    
    module_path = experiment_modules[experiment_name]
    module_name, class_name = module_path.rsplit('.', 1)

    print(module_name)
    module = importlib.import_module(module_name)
    experiment = getattr(module, class_name)
    
    # Load the experiment data
    demo_head_rgb = cv2.imread(f"{DIR}/demo_head_rgb.png")[..., ::-1].copy() 
    demo_head_mask = cv2.imread(f"{DIR}/demo_head_seg.png")[...,::-1].astype(bool)
    demo_wrist_rgb = cv2.imread(f"{DIR}/demo_wrist_rgb.png")[..., ::-1].copy() 
    demo_wrist_mask = cv2.imread(f"{DIR}/demo_wrist_seg.png")[...,::-1].astype(bool)

    with open(f"{DIR}/label.txt", 'r') as file:
        demo_obj = file.read()
    with open(f"{DIR}/demo_bottlenecks.json") as f:
        dbn = json.load(f)
        demo_waypoints = np.vstack([dbn[key] for key in dbn.keys()])
    
    # Create and return an instance of ExperimentData
    return experiment(
        dir=DIR,
        object=demo_obj,
        demo_waypoints=demo_waypoints,
        demo_head_rgb=demo_head_rgb,
        demo_head_mask=demo_head_mask,
        demo_wrist_rgb=demo_wrist_rgb,
        demo_wrist_mask=demo_wrist_mask
    )