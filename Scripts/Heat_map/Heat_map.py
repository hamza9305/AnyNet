import numpy as np
import cv2
import matplotlib.pyplot as plt


img1 = cv2.imread('/home/haahm/Development/projects/Results/Final_results_after_filter/Divided_images/ground_truth.png',0)
img2 = cv2.imread('/home/haahm/Development/projects/Results/Final_results_after_filter/Divided_images/trained.png',0)

img1 = img1.astype(np.float)
img2 = img2.astype(np.float)

invalid_mask_a = (img1 == 0)  # For instance, 0 is disturbing in disparitiy maps if we invert them.
invalid_mask_b = (img1 == np.inf)
invalid_mask_c = (img1 == -np.inf)
invalid_mask_d = (img1 != img1)  # looks strange but "NaN != NaN" -> detects NaNs

invalid_mask = np.copy(invalid_mask_a) # make a copy if you need invalid_mask_a afterwards
invalid_mask = np.bitwise_or(invalid_mask, invalid_mask_b)
invalid_mask = np.bitwise_or(invalid_mask, invalid_mask_c)
invalid_mask = np.bitwise_or(invalid_mask, invalid_mask_d)

print(img1.dtype)
#cv2.waitKey(0)

dif = np.abs(img2 - img1)
dif[invalid_mask] = 0

print(dif)
print('dtype diff',dif.dtype)
print(np.max(dif))
print(np.min(dif))


print(dif.shape)
fig, ax = plt.subplots()
psm = ax.pcolormesh(dif)
fig.colorbar(psm, ax=ax)
ax.invert_yaxis()
plt.savefig('/home/haahm/Development/projects/Results/Final_results_after_filter/Heat_Map/heat_map.png')
plt.show()





