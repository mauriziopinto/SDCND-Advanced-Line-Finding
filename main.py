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

def calibrate_camera(calibration_folder):
    """
    Calculate the camera matrix and the distortion coefficients using a set of calibration images

    Args:
        calibration_folder: the folder where the calibration images (chessboards images) are
 
    Returns:
        mtx: the camera matrix
        dist: the distorsion coefficients
    """
    print("Calibrating camera...")
    nx, ny = 9, 6

    images = glob.glob(calibration_folder)

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

def undistort(distorted_img, mtx, dist):
    """
    Return an undistorted image obtained using the camera matrix and distorsion coefficients obtained during the camera calibration phase

    Args:
        distorted_img: the distorted image
        mtx: the camera matrix:
        dist: the distorsion coefficients
 
    Returns:
        the undistorted image
    """
    return cv2.undistort(distorted_img, mtx, dist, None, mtx)

def warp(img):
    """
    Apply a perspective trasformation to an image (by means of a transformation matrix)

    Args:
        img: the image
 
    Returns:
        the image after the perspective transformation with matrix warp_matrix
    """
    return cv2.warpPerspective(img, warp_matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

def adjust_gamma(image, gamma=1.0):
    """Apply gamma conversion to RGB Image.
        Args:
            image: RGB Image
            gamma: rate for gamma conversion
        Returns:
           image: converted RGB Image
    """


    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def threshold(img):
    """
    Detect lanes scanning the whole image and identifying where the lanes are by means of histograms

    Args:
        binary_warped: a bird-view image of the road
 
    Returns:
        out_img: image with lane detected
        ret: a dict containing left_fit, right_fit, fit_leftx, fit_rightx, ploty, mid_lane (center of the lane), left (left position of the lane), right (right position of the lane)
    """

    # Adjust gamma
    adjust_gamma(img, gamma=0.3)

    # Parameters
    sobel_kernel=9
    s_thresh=(180, 190)
    h_thresh=(21, 100)
    sx_thresh=(40, 100)
    dir_thresh=(0.2, 1.2)

    # select the L and S channels from the HLS image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    hls_h = hls[:, :, 0]
    hls_l = hls[:, :, 1]
    hls_s = hls[:, :, 2]
    

    # Sobel x
    sobelx = cv2.Sobel(hls_l, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Direction Threshold
    sobely = cv2.Sobel(hls_l, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobely = np.absolute(sobely)
    direction=np.arctan2(abs_sobely, abs_sobelx)
    sdbinary = np.zeros_like(direction)
    sdbinary[(direction >= dir_thresh[0]) & (direction <= dir_thresh[1])] = 1

    # Threshold s channel
    s_binary = np.zeros_like(hls_s)
    s_binary[(hls_s >= s_thresh[0]) & (hls_s <= s_thresh[1])] = 1

    # Threshold h channel
    h_binary = np.zeros_like(hls_h)
    h_binary[(hls_h >= h_thresh[0]) & (hls_h <= h_thresh[1])] = 1

    # Select the yellow patches
    yellow_img = cv2.inRange(img, (204,204,0), (255,255,153))

    # Select the white patches
    white_img = cv2.inRange(img, (200, 200, 200), (255, 255, 255))

    # Merge white and yellow patches
    yellow_and_white_img = yellow_img | white_img
    yellow_and_white_img = np.divide(yellow_and_white_img, 255)

    # Stack each channel
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(h_binary == 1) | (yellow_and_white_img == 1) | ((sxbinary == 1) & (sdbinary == 1)) ] = 1

    return combined_binary
    

def detect_lane_full(binary_warped):
    """
    Detect lanes scanning the whole image and identifying where the lanes are by means of histograms

    Args:
        binary_warped: a bird-view image of the road
 
    Returns:
        out_img: image with lane detected
        ret: a dict containing left_fit, right_fit, fit_leftx, fit_rightx, ploty, mid_lane (center of the lane), left (left position of the lane), right (right position of the lane)
    """
    
    # Input is a warped binary image

    # Remove the center of the image, where we are sure there are no lanes
    y = binary_warped.shape[0]
    x = binary_warped.shape[1]
    center = int(x / 2)
    offset = 200
    
    a3 = np.array( [[[center-offset,0],[center+offset,0],[center+offset,y],[center-offset,y]]], dtype=np.int32 )
    cv2.fillPoly( binary_warped, a3, 0 )

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
    margin = 70
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

def detect_lane_subsequent(binary_warped, left_fit, right_fit):
    """
    Detect lines but relying on the information found in previous processing stage

    Args:
        binary_warped: a bird-view image of the road
        left_fit: fit for left lane from previous processing stage
        right_fit:: fit for right lane from previous processing stage
 
    Returns:
        out_img: image with lane detected
        ret: a dict containing left_fit, right_fit, fit_leftx, fit_rightx, ploty, mid_lane (center of the lane), left (left position of the lane), right (right position of the lane)
    """

    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 70
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


def calculate_radius(image, leftx, rightx, ploty, l, r):
    """
    Calculate the radius of curvature, for both left and right lane

    Args:
        image: image
        leftx: 
        rightx: 
        ploty:
 
    Returns:
        txt: text to overlay on the frame
        left_curverad: left lane curvature radius
        right_curverad: right lane curvature radius
    """

    # define conversion in x and y from pixel space to meters
    y_eval = 719
    ym_per_pix = 30/720
    xm_per_pix = 3.7/(l-r)

    # fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    #calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    txt = "LEFT LANE CURVATURE RADIUS (m): " + str(round(left_curverad,2)) + "\nRIGHT LANE CURVATURE RADIUS (m): " + str(round(right_curverad,2))

    return txt, left_curverad, right_curverad

def display_info(image, text, mid_lane, l, r):
    """
    Display info on the screen

    Args:
        image: image
        text: text to overlay containing the left and right curvature radiuses
        mid_lane: approx. position of the center of the lane
        l: position of the left lane in the image
        r: position of the right lane in the image
 
    Returns:
        image: image with text overlaid
    """
    font = cv2.FONT_HERSHEY_PLAIN
    y0, dy = 20, 20

    img_center = int(image.shape[1] / 2)
    lane_center = int(l + ((r-l)/2))
    xm_per_pix = 3.7/(r-l) # transform from pixels to meters 
    text = text + "\nDISTANCE FROM CENTER OF LANE (m): " + str( round((img_center - lane_center) * xm_per_pix, 2) )
    
    # split the text on multiple lines
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        cv2.putText(image,line,(20,y), font, 1,(255,255,255),2)
        cv2.line(image, (lane_center, 720), (lane_center, 680), (255,0,0), 2)
        cv2.line(image, (img_center, 720), (img_center, 680), (0,0,255), 2)
    
    return image


def lanes_warped(warped, left_fitx, right_fitx, ploty ):
    """
    Draw the lanes on an empty canvas (bird-view)

    Args:
        warped: bird-view image of the road
        left_fitx:
        right_fitx:
        ploty:
 
    Returns:
        color_warp: black canvas, bird-view, with lanes overlaid (to be later merged with the original image)
    """

    # Create an empty image
    warp_zero = np.zeros_like(warped[:,:,2]).astype(np.uint8)
    # three channels
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    # We will later change perspective and merge this image with the original frame
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 180))
    return color_warp

def final_image(image, color_warp):   
    """
    Merge lanes (bird-view) with the original undistorted frame

    Args:
        image: original, undistorted, image of the road
        color_warp: black-canvas, bird-view, with lanes overlaid
   
    Returns:
        result: original, undistorted, image of the road with lanes detected (green semi-transparent filled poly)
    """

    # Inverse perspective matrix (inverse_warp_matrix)
    lanes = cv2.warpPerspective(color_warp, inverse_warp_matrix, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, lanes, 0.27, 0)
    return result



def sanity_check(ret, radius_l, radius_r):
    """
    Transformation pipeline

    Args:
        ret: a dict containing left_fit, right_fit, fit_leftx, fit_rightx, ploty, mid_lane (center of the lane), left (left position of the lane), right (right position of the lane)
   
    Returns:
        true if the sanity check was successful
    """
    radii_ok = True
    width_ok = True
    parallel_ok = True

    radius_diff = abs( 1./radius_l - 1./radius_r)
    if (radius_diff > 0.001):
        radii_ok = False

    width = abs(ret["left"] - ret["right"]) # average width of the lane
    if (width < 890 or width > 990):
        width_ok = False

    fit_diff = abs(ret["left_fit"][1] - ret["right_fit"][1])
    if (fit_diff>0.6):
        parallel_ok = False

    return radii_ok and width_ok and parallel_ok

def pipeline(frame):
    """
    Transformation pipeline

    Args:
        frame: an image (a frame of the video)
   
    Returns:
        img: the input image after all the steps in the pipeline have been executed
    """

    # last polynomials found for left and right lane
    global last_right_fit
    global last_left_fit

    # first step: correct the image distortion
    img = undistort(frame, camera_matrix, dist_coeff)

   
    # threshold
    img = threshold(img)

    
    # uncomment to be able to display the binary img
    #img[ img == 1] = 255
    #img = np.dstack((img, img, img))

    
    # perspective transform (to bird-view)
    img = warp(img)

    f = np.copy(img)

    # do a full processing if we don't know yet where the lanes are
    if (len(last_right_fit)==0 and len(last_left_fit)==0):
        img, ret = detect_lane_full(img)
        last_right_fit = ret["right_fit"]
        last_left_fit = ret["left_fit"]
    else:
        # re-use information from previous processing to optimize the lane detection phase
        img, ret = detect_lane_subsequent(img, last_left_fit, last_right_fit)
        txt, radius_l, radius_r = calculate_radius(img, ret["fit_leftx"], ret["fit_rightx"], ret["fity"], ret["left"], ret["right"])
        last_right_fit = ret["right_fit"]
        last_left_fit = ret["left_fit"]
        # sanity check and fallback on detect full
        if not sanity_check(ret, radius_l, radius_r):
            img, ret = detect_lane_full(f)
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
    txt, left_curverad, right_curverad = calculate_radius(img, fit_leftx, fit_rightx, fity, left, right)

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
    src = np.array([[490, 482],[810, 482], [1250, 720],[40, 720]], dtype=np.float32)
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
    
    # apply the pipeline transformation to all the frames in the video
    white_clip = clip1.fl_image(pipeline)

    # save the video in a file
    white_clip.write_videofile(output, audio=False)

    print("Completed. Output file: ", output)