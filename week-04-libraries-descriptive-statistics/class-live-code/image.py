import numpy as np
import matplotlib.pyplot as plt

img = np.random.randint(0, 255, size=(128, 128), dtype='uint8')



plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()