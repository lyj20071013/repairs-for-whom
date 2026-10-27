# Text-environment compatibility modules

The original WebShop text environment uses `gym.Env` only as a base class. Selenium is used by the browser environment, which the text workflow does not invoke. These minimal substitutes reduce unnecessary dependencies and avoid the historical Gym/NumPy compatibility issue. The worker loads them first. They do not implement environment transitions or change observations, actions or rewards.
