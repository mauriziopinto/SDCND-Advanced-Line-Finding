import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip
import os

# the following global variables are used to keep track of the last fit for the left and right lanes
global last_left_fit 
global last_right_fit

last_left_fit = np.zeros(0)
last_right_fit = np.zeros(0)

# calibrate camera using the chessboard reference images
def calibrate_camera(image_url):
    print("Calibrating camera...")
    nx, ny = 9, 6

    images = glob.glob(image_url)

    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    corner = (nx, ny)

    for image in images:
        img = mpimg.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, corner, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    print("Camera calibrated successfully")
    return mtx, dist

# correct an image distortion using the parameters obtained during the camera calibration phase
def undistort(distorted_img, mtx, dist):
    return cv2.undistort(distorted_img, mtx, dist, None, mtx)

# apply a perspective trasformation to an image (by means of a transformation matrix)
def warp(img):
    return cv2.warpPerspective(img, warp_matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

# convert a HLS image to a binary one
def binary_img(img):
    H = img[:,:,0]
    L = img[:,:,1]
    S = img[:,:,2]
    thresh = (90, 255)
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary

# apply thresholding tecniques to improve the accuracy of lane recognition
def threshold(img):

    # Gaussian Blur
    kernel_size = 5
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    # Convert to HLS color space and separate the S channel
    # Note: img is the warped image and not the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary


# detect lanes scanning the whole image and identifying where the lanes are by means of histograms
def detect_lane_full(binary_warped):
    
    # Input is a warped binary image

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    fit_leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    fit_rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    ret = {}
    ret["left_fit"] = left_fit
    ret["right_fit"] = right_fit
    ret["fit_leftx"] = fit_leftx
    ret["fit_rightx"] = fit_rightx
    ret["fity"] = ploty
    ret["mid_lane"] = (np.max(fit_rightx) - np.min(fit_leftx))
    ret["left"] = np.min(fit_leftx)
    ret["right"] = np.max(fit_rightx)

    return out_img, ret

# detect lines but relying on the information found in previous processing stage
def detect_lane_subsequent(binary_warped, left_fit, right_fit):

    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    margin_to_draw = 20
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    fit_leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    fit_rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.zeros_like(binary_warped).astype(np.uint8)
    out_img = np.dstack((out_img, out_img, out_img))*255


    ret = {}
    ret["left_fit"] = left_fit
    ret["right_fit"] = right_fit
    ret["fit_leftx"] = fit_leftx
    ret["fit_rightx"] = fit_rightx
    ret["fity"] = ploty
    ret["mid_lane"] = (np.max(fit_rightx) - np.min(fit_leftx))
    ret["left"] = np.min(fit_leftx)
    ret["right"] = np.max(fit_rightx)

    return out_img, ret


# calculate the radius of curvature, for both left and right lane
def calculate_radius(image, left_fit, right_fit):
    y_eval = 719
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    txt = "LEFT LANE CURVATURE RADIUS (m): " + str(round(left_curverad,2)) + "\nRIGHT LANE CURVATURE RADIUS (m): " + str(round(right_curverad,2))

    return txt, left_curverad, right_curverad


# display info on the screen
def display_info(image, text, mid_lane, l, r):
    font = cv2.FONT_HERSHEY_PLAIN
    y0, dy = 20, 20

    img_center = int(image.shape[1] / 2)
    lane_center = int(l + ((r-l)/2))
    xm_per_pix = 3.7/700 # transform from pixels to meters 
    text = text + "\nDISTANCE FROM CENTER OF LANE (m): " + str( round((img_center - lane_center) * xm_per_pix, 2) )

    # split the text on multiple lines
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        cv2.putText(image,line,(20,y), font, 1,(255,255,255),2)
        cv2.line(image, (lane_center, 720), (lane_center, 680), (255,0,0), 2)
        cv2.line(image, (img_center, 720), (img_center, 680), (0,0,255), 2)
    
    return image


# draw the lanes on an empty canvas (bird-view)
def lanes_warped(warped, left_fitx, right_fitx, ploty ):
    # Create an empty image
    binary = binary_img(warped)
    warp_zero = np.zeros_like(binary).astype(np.uint8)
    # three channels
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    # We will later change perspective and merge this image with the original frame
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    return color_warp


# merge lanes (bird-view) with the original undistorted frame
def final_image(image, color_warp):   
    # Inverse perspective matrix (inverse_warp_matrix)
    lanes = cv2.warpPerspective(color_warp, inverse_warp_matrix, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, lanes, 0.27, 0)
    return result


# transformation pipeline
def pipeline(frame):

    # last polynomials found for left and right lane
    global last_right_fit
    global last_left_fit

    # first step: correct the image distortion
    img = undistort(frame, camera_matrix, dist_coeff)

    # perspective transform (to bird-view)
    img = warp(img)

    # threshold
    img = threshold(img)

    # do a full processing if we don't know yet where the lanes are
    if (len(last_right_fit)==0 and len(last_left_fit)==0):
        img, ret = detect_lane_full(img)
        last_right_fit = ret["right_fit"]
        last_left_fit = ret["left_fit"]
    else:
        # re-use information from previous processing to optimize the lane detection phase
        img, ret = detect_lane_subsequent(img, last_left_fit, last_right_fit)
        last_right_fit = ret["right_fit"]
        last_left_fit = ret["left_fit"]
       

    left_fit = ret["left_fit"]
    right_fit = ret["right_fit"]
    fit_leftx = ret["fit_leftx"]
    fit_rightx = ret["fit_rightx"]
    fity = ret["fity"]
    mid_lane = ret["mid_lane"] 
    left = ret["left"]
    right = ret["right"]

    
    # draw polynomials curves on the warped image (bird-view)
    img = lanes_warped(img, fit_leftx, fit_rightx, fity)

    # obtain the final image (lanes + original frame)
    img = final_image(frame, img)

    # calculate the radius of curvature
    txt, left_curverad, right_curverad = calculate_radius(img, left_fit, right_fit)

    # calculate center of lane
    mid_lane = (np.max(fit_rightx) - np.min(fit_leftx))

    # calculate position of left lane
    left = np.min(fit_leftx)

    # calculate position of right lane
    right = np.max(fit_rightx)

    # display curvature radiuses and distance from the center of the lane
    img = display_info(img, txt, int(mid_lane), int(left), int(right)  )

    # pipeline is complete
    return img

# main
if __name__ == "__main__":
    
    # file where the camera calibration parameters are save
    cam_file = 'calibration.npz'

    # src and dest points for perspective transform
    src = np.array([[550, 462],[730, 462], [1280, 720],[120, 720]], dtype=np.float32)
    dst = np.array([[0, 0], [1280, 0], [1250, 720],[40, 720]], dtype=np.float32)

    
    # src to dest perspective transformation matrix
    warp_matrix = cv2.getPerspectiveTransform(src, dst)

    # dst to src perspective transformation matrix
    inverse_warp_matrix = cv2.getPerspectiveTransform(dst, src)

    # if camera has not been calibrated
    if not os.path.isfile(cam_file):
        camera_matrix, dist_coeff = calibrate_camera('./camera_cal/calibration*.jpg')
        np.savez_compressed(cam_file, camera_matrix=camera_matrix, dist_coeff=dist_coeff)
    else:
        # Camera has been already calibrated
        print('Loading camera data from', cam_file)
        data = np.load(cam_file)
        camera_matrix = data['camera_matrix']
        dist_coeff = data['dist_coeff']

    # file where the video will be saved
    output = './carnd_p4_output.mp4'

    # source video
    clip1 = VideoFileClip('./project_video.mp4')
    #clip1 = VideoFileClip('./harder_challenge_video.mp4')
    #clip1 = VideoFileClip('./challenge_video.mp4')

    # apply the pipeline transformation to all the frames in the video
    white_clip = clip1.fl_image(pipeline)

    # save the video in a file
    white_clip.write_videofile(output, audio=False)

    print("Completed.")

    