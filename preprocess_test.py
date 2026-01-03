from env.freeway_env import FreewayENV
import matplotlib.pyplot as plt
from utils.preprocessing import preprocess_frame
from config.config import cfg

env = FreewayENV()
obs, _ = env.reset()

print(f"Chicken size: {env.chicken_width}x{env.chicken_height}")
print(f"Car size: {env.car_width}x{env.car_height}")

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