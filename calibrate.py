import cv2, glob, os
import numpy as np
import pickle

def perform_calibration(cache_path):

	# Load cached points
	with open('calibration_cache.pkl', 'rb') as f:
		data = pickle.load(f)

	window_name = 'Undistortion Viewer (press d to toggle, q to quit)'

	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

	objpoints = data['objpoints']
	imgpoints = data['imgpoints']
	image_shape = data['image_shape']
	image_files = data['filenames']
	print(f'{image_shape=}')
	print(f'{image_files=}')

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

	print("Calibration RMS error:", ret)
	print("Calibrated Camera matrix:\n", mtx)
	print("Distortion coefficients:\n", dist)

	# Save to YAML
	fs = cv2.FileStorage('calibration.yaml', cv2.FILE_STORAGE_WRITE)
	fs.write('rms_error', ret)
	fs.write('camera_matrix', mtx)
	fs.write('dist_coefficients', dist)
	fs.write('image_width', image_shape[0])
	fs.write('image_height', image_shape[1])
	fs.release()

	print("Calibration parameters saved to calibration.yaml")

	# Load an image to undistort
	# img = cv2.imread('data/1.jpg')  # adjust filename
	# h, w = img.shape[:2]

	# undistorted_img = cv2.undistort(img, mtx, dist)

	# Load all images in data/ sorted by number
	# image_files = sorted(glob.glob('data/*.jpg'), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
	# image_files = glob.glob('data/*.jpg')
	# image_files = sorted(glob.glob('test_data/*.jpg'))
	# image_files = sorted(glob.glob('test_data_2/*.jpg'))

	if not image_files:
		print("No images found in data/")
		exit(1)

	current_index = 0
	show_undistorted = True
	display_points = True
	# display_points = False

	print("Controls:")
	print("→ next image, ← previous image, d toggle undistort, q quit")

	radius = 10

	while True:
		img_path = image_files[current_index]
		img = cv2.imread(img_path)
		h, w = img.shape[:2]

		undistorted_img = cv2.undistort(img, mtx, dist)

		while True:
			display_img = undistorted_img.copy() if show_undistorted else img.copy()

			if display_points:
				if show_undistorted:
					res, _ = cv2.projectPoints(objpoints[current_index], rvecs[current_index], tvecs[current_index], mtx, np.zeros(5))
				else:
					res, _ = cv2.projectPoints(objpoints[current_index], rvecs[current_index], tvecs[current_index], mtx, dist)

				res = res[:,0,:]

				for pt in res:
					pt = tuple(int(x) for x in pt)
					cv2.circle(display_img,(pt[0],pt[1]), radius, (0,0,255), -1)

			cv2.imshow(window_name, display_img)

			key = cv2.waitKey(0) & 0xFF

			if key == ord('d'):
				show_undistorted = not show_undistorted
				break
			elif key == ord('s'):
				if show_undistorted:
					cv2.imwrite('undistorted.jpg', display_img)
			elif key == ord('r'):
				display_points = not display_points
				# display_img = undistorted_img if show_undistorted else img
			elif key == ord('q'):
				cv2.destroyAllWindows()
				exit(0)
			elif key == 83:  # right arrow
				current_index = np.min((current_index + 1, len(image_files)-1))
				break
			elif key == 81:  # left arrow
				current_index = np.max((0, (current_index - 1)))
				break

		# cv2.destroyAllWindows()

if __name__=='__main__':
	perform_calibration('calibration_cache.pkl')