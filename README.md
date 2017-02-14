**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Calibration image - distorted"
[image2]: ./output_images/undistorted_chessboard.jpg "Calibration image - distortion corrected"
[image3]: ./test_images/test3.jpg "Distorted - Camera car"
[image4]: ./output_images/undistorted_test3.jpg "Undistorted - Camera car"
[image5]: ./output_images/warped_image_result.jpg "Warped image example"
[image6]: ./output_images/binary_image_result.jpg "Binary image example"
[image7]: ./output_images/polynomial_result.jpg "Lane polynomials example"
[image8]: ./output_images/final_result.jpg "Final result"
[video1]: ./project_video.mp4 "Video"

## [Link to the Udacity Rubric](https://review.udacity.com/#!/rubrics/571/view)

---
###Writeup

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

The final writeup has been submitted to Udacity and is available at this [url](https://github.com/mauriziopinto/SDCND-Advanced-Line-Finding): https://github.com/mauriziopinto/SDCND-Advanced-Line-Finding

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The **calibrate_camera** method in main.py, available in this repository, scans the folder that contains the calibration images (chessboard images) and for each of them:

* converts the image to gray scale
* finds the corners (9x6) on the chessboard using the **findChessboardCorners** method in OpenCV
* calibrates the camera and saves the calibration parameters for later use (method **calibrateCamera** in OpenCV)

An example result is shown in the table below:

| Distorted | Undistorted | 
| ------------- |:-------------:| 
| ![Calibration image - distorted][image1] | ![Calibration image - distortion corrected][image2]|




###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
The distortion correction applied to camera images provides the following results:

| Distorted | Undistorted | 
| ------------- |:-------------:| 
| ![Distorted][image3] | ![Undistorted][image4]|



####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The **threshold** method in main.py implements the transformations I chose to implement:

* Gamma adjustment (in order to mitigate the effect of varying colors of the asphalt)
* Conversion to grayscale and then Sobel filter on the x axis
* Threshold for the H and S channel in the HLS version of the image
* Threshold that identifies the yellow patches in the image (mostly left lane)
* Threshold that identifies the white patches in the image (mostly right lane)
* Combination of the images resulting from the previous steps

```python
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
```

| Undistorted | Binary | 
| ------------- |:-------------:| 
| ![Undistorted][image4] | ![Binary][image6]|


####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform matrix and its inverse are created in the **main.py**:

```python
# src and dest points for perspective transform
src = np.array([[490, 482],[810, 482], [1250, 720],[40, 720]], dtype=np.float32)
dst = np.array([[0, 0], [1280, 0], [1250, 720],[40, 720]], dtype=np.float32)


# src to dest perspective transformation matrix
warp_matrix = cv2.getPerspectiveTransform(src, dst)

# dst to src perspective transformation matrix
inverse_warp_matrix = cv2.getPerspectiveTransform(dst, src)
```

The method that actually implements the transformation is called **warp** and it simply uses the OpenCV **warpPerspective** method. The source and destination points have been manually selected, in order to obtain a region of the road that could potentially apply to most of the frames in the test videos.


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 490, 482      | 0, 0          | 
| 810, 482      | 1280, 0       |
| 1250, 720     | 1250, 720     |
| 40, 720      | 40, 720       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

| Binary | Warped | 
| ------------- |:-------------:| 
| ![Binary][image6] | ![Warped][image5]|


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane-line pixels are identified in the two following methods in main.py:

* **detect_lane_full**
* **detect_lane_subsequent**

The first method implements a full processing with sliding window as described in the Udacity lessons. The second method avoids to do a full scan of the picture and, instead, relies on the areas in the previous image where the lanes have been found previously.

A simple sanity check on the following parameters:

* the difference between the curvature radius of the left and right lanes
* 

makes the second method invoke the first (because probably the lane-line pixels detection did not work as expected)

| Warped | Polynomials | 
| ------------- |:-------------:| 
| ![Original][image5] | ![Polynomials][image7]|

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radiuses of curvature for left and right lane are calculated in the **calculate_radius** method:

```python
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
```

It is based on some assumptions described in the Udacity lessons. It returns as well the text that shall be overwritten to the image.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in method **final_image** in main.py:

* first: the image containing the lanes polygon seen from a top perspective is transformed back into the correct perspective, using the inverse transformation matrix
* then the two images (original frame, without distorsion correction, and images with black background and green lanes) are merged

```python
# merge lanes (bird-view) with the original undistorted frame
def final_image(image, color_warp):   
    # Inverse perspective matrix (inverse_warp_matrix)
    lanes = cv2.warpPerspective(color_warp, inverse_warp_matrix, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, lanes, 0.27, 0)
    return result
```

The blue and red bars in the image show respectively:

* the exact center of the lane
* the (approximate) position where the vehicle is

The two small bar give a graphical representation of the distance between from center of the lane.

Here is an example of my result on a test image:

![Final result][image8]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Final video output is available [here](./carnd_p4_output.mp4) or on [Youtube](https://youtu.be/Mwz0CjrKeco)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The thresholding mechanism is still quite unable to cope well with big variations in the color of the asphalt or, more in general, with variations in the lights and shadows: using a thresholding on HLS and RGB (yellow and white patches) improved a lot the performances when compared to the first project in the Udacity course, but it is still not sufficient for any kind of environment.

The pipeline performs acceptably well on the challenge video, but it fails quite bad with the harder challenge video: very sharp bends are not detected correctly.

Something I would like to improve:

* thresholding performances and resilience to different light conditions
* overall performances (the pipeline, even on a quite powerful computer, is not fast enough to be used on a real stream coming from a dash camera for example)
