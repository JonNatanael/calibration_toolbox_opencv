import cv2
import os
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt

# Calibration settings
pattern_size = (4, 11)  # asymmetric circle grid
square_size = 1.0       # real-world unit

# Scales to try
scale_factors = [1.0, 0.5]
scale_factors = [1.0, 0.5, 0.2]

display = False
display = True

save_images = False
save_images = True
save_dir = 'tmp/'

def generate_grid_points(pattern_size):
	objp = []
	square_size = 1
	for i in range(pattern_size[1]):  # height
		for j in range(pattern_size[0]):  # width
			x = (2 * j + i % 2) * square_size
			y = i * square_size
			z = 0
			objp.append((x, y, z))

	# Convert to NumPy array
	objp = np.array(objp, dtype=np.float32)

	return objp

def detect_and_save(data_dir):
	objp = generate_grid_points(pattern_size)

	# Arrays to store points
	objpoints = []
	imgpoints = []
	filenames = []

	images = glob.glob('data/*.jpg')  # adjust folder
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	cv2.namedWindow("image", cv2.WINDOW_NORMAL)

	if save_images:
		os.makedirs(save_dir, exist_ok=True)

	for fname in images:
		img = cv2.imread(fname)
		orig_h, orig_w = img.shape[:2]
		pattern_found = False

		for scale in scale_factors:
			# Resize for detection
			scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
			gray_scaled = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)

			# Try to detect the asymmetric circle grid
			ret, centers_scaled = cv2.findCirclesGrid(
				gray_scaled, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
			)

			if ret:
				# Rescale points back to original resolution
				corners2 = cv2.cornerSubPix(gray_scaled, centers_scaled, (11,11), (-1,-1), criteria)

				objpoints.append(objp)
				imgpoints.append(corners2 / scale)

				print(f"Pattern detected in {fname} at scale {scale}")
				pattern_found = True
				filenames.append(fname)

				if display or save_images:
					frame_ = cv2.drawChessboardCorners(scaled_img, pattern_size, corners2, ret)

					if save_images:
						out_fn = fname.split('/')[-1]
						# print(f'{save_dir}{out_fn}')
						cv2.imwrite(f'{save_dir}{out_fn}', frame_)

					if display:

						cv2.imshow("image", frame_)
						key = cv2.waitKey(0)
						if key==27:
							return

						# frame_ = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)



						# plt.clf()
						# plt.imshow(frame_)
						# plt.title(fname)
						# plt.show()

				break  # stop trying other scales

		if not pattern_found:
			print(f"Pattern NOT detected in {fname} at any scale")

	# Save to cache
	with open('calibration_cache.pkl', 'wb') as f:
		pickle.dump({'objpoints': objpoints, 'imgpoints': imgpoints, 'image_shape': (orig_w, orig_h), 'filenames': filenames}, f)

	print("Calibration points saved to calibration_cache.pkl")

if __name__=='__main__':
	data_dir = 'data/'
	detect_and_save(data_dir)