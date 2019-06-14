import numpy as np

from softlearning.environments.utils import get_environment_from_params

# Add tasks & filepaths here:
TASK_GOALS = {
    'ImageScrewV2-v0': '/Users/justinvyu/Developer/summer-2019/softlearning-vice/goal_classifier/goal_180_image_True/positives.pkl',
}

def get_goal_example_from_variant(variant):
    if variant['task'] in TASK_GOALS.keys():
        import pickle
        with open(TASK_GOALS[variant['task']], 'rb') as file:
            goal_examples = pickle.load(file)
    else:
        raise NotImplementedError

    n_goal_examples = variant['data_params']['n_goal_examples']

    goal_examples_train = goal_examples[:n_goal_examples]
    goal_examples_validation = goal_examples[n_goal_examples:]

    return goal_examples_train, goal_examples_validation
