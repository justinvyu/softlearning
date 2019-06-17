import numpy as np
from PIL import Image
import os
import pickle
import sys
from softlearning.environments.gym.wrappers import NormalizeActionWrapper
# from robosuite.environments.invisible_arm_free_float_manipulation import (
#     InvisibleArmFreeFloatManipulation)
from robosuite.environments.image_invisible_arm_free_float_manipulation import (
    ImageInvisibleArmFreeFloatManipulation
)
cur_dir = os.path.dirname(os.path.realpath(__file__))

directory = cur_dir + "/pusher"

if not os.path.exists(directory):
    os.makedirs(directory)

images = True
image_shape = (64,64,3)

# Fixed Screw
# fixed_arm=True,
# fixed_claw=False,
# initial_x_range=(0., 0.),
# initial_y_range=(0., 0.),
# target_x_range=(0., 0.),
# target_y_range=(0., 0.),
# rotation_only=True,  # Find a way to generalize this across many tasks.

env = ImageInvisibleArmFreeFloatManipulation(
    image_shape=image_shape,
    viewer_params={
        "azimuth": 90,
        "elevation": -27.7,
        "distance": 0.30,
        "lookat": np.array([-2.48756381e-18, -2.48756381e-18,  7.32824139e-01])
    },
    rotation_only=True,
    fixed_arm=True,
    fixed_claw=False,
    initial_x_range=(0., 0.),
    initial_y_range=(0., 0.),
    target_x_range=(0., 0.),
    target_y_range=(0., 0.),
    initial_z_rotation_range=(np.pi, np.pi),
)

observations = []
num_positives = 0
TOTAL_EXAMPLES = 100

while num_positives <= TOTAL_EXAMPLES:
    observation = env.reset()
    print("RESETTING")
    t = 0
    while t < 25:
        low, high = env.action_spec
        action = np.random.uniform(low, high, size=(env.dof,))

        for _ in range(5):
            env.step(action)

        obs = env._get_image_observation()
        # env.render()

        rotation_dist = env._get_rotation_distances()["screw"][0]
        print("rotation angle (degrees):", rotation_dist * 180 / np.pi)

        if rotation_dist < 0.1:
            observations.append(obs)
            image = obs[:np.prod(image_shape)].reshape(image_shape)
            result = Image.fromarray(image.astype(np.uint8))
            result.save(directory + '/img%i.jpg' % num_positives)
            num_positives += 1
        if num_positives % 5 == 0:
            with open(directory + '/positives.pkl', 'wb') as file:
                pickle.dump(np.array(observations), file)
        t += 1
