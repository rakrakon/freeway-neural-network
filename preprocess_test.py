from env.freeway_env import FreewayENV
import matplotlib.pyplot as plt
from utils.preprocessing import preprocess_frame
from config.config import cfg

env = FreewayENV()
obs, _ = env.reset()

for i in range(30):
    obs, _, _, _ = env.step(env.action_space.sample())

processed = preprocess_frame(obs, cfg.train)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(obs)
plt.title('Raw Observation')
plt.subplot(1, 2, 2)
plt.imshow(processed, cmap='gray')
plt.title('Processed (what network sees)')
plt.savefig('debug_visibility.png', dpi=150)
print("Saved debug_visibility.png")