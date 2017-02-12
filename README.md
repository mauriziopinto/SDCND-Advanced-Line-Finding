**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Calibration image - distorted"
[image2]: ./output_images/undistorted_chessboard.jpg "Calibration image - distortion corrected"
[image3]: ./test_images/test1.jpg "Distorted - Camera car"
[image4]: ./output_images/undistorted_test1.jpg "Undistorted - Camera car"
[image5]: ./output_images/warped_image_result.jpg "Warped image example"
[image6]: ./output_images/binary_image_result.png "Binary image example"
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

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

**Please note that I preferred to first warp the image and then to apply the threshold**

The perspective transform matrix and its inverse are created in the **main.py**:

```python
# src and dest points for perspective transform
src = np.array([[550, 462],[730, 462], [1280, 720],[120, 720]], dtype=np.float32)
dst = np.array([[0, 0], [1280, 0], [1250, 720],[40, 720]], dtype=np.float32)


# src to dest perspective transformation matrix
warp_matrix = cv2.getPerspectiveTransform(src, dst)

# dst to src perspective transformation matrix
inverse_warp_matrix = cv2.getPerspectiveTransform(dst, src)
```

The method that actually implements the transformation is called **warp** and it simply uses the OpenCV **warpPerspective** method. The source and destination points have been manually selected, in order to obtain a region of the road that could potentially apply to most of the frames in the test videos.


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 550, 462      | 0, 0          | 
| 730, 462      | 1280, 0       |
| 1280, 720     | 1250, 720     |
| 120, 720      | 40, 720       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

| Original | Warped | 
| ------------- |:-------------:| 
| ![Original][image3] | ![Warped][image5]|


####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The **threshold** method in main.py implements the transformations I chose to implement:

* Gaussian blur with kernel size = 5
* Conversion to grayscale and then Sobel filter on the x axis
* Threshold for the color channel (HLS image)
* Combination of the images resulting from the previous steps

```python
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
```

| Warped | Binary | 
| ------------- |:-------------:| 
| ![Original][image5] | ![Binary][image6]|



####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane-line pixels are identified in the two following methods in main.py:

* **detect_lane_full**
* **detect_lane_subsequent**

The first method implements a full processing with sliding window as described in the Udacity lessons. The second method avoids to do a full scan of the picture and, instead, relies on the areas in the previous image where the lanes have been found previously.

A simple sanity check on the curvature radius of the left and right lanes, makes the second method invoke the first (because probably the lane-line pixels detection did not work as expected)

| Warped | Polynomials | 
| ------------- |:-------------:| 
| ![Original][image5] | ![Polynomials][image7]|

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radiuses of curvature for left and right lane are calculated in the **calculate_radius** method:

```python
# calculate the radius of curvature, for both left and right lane
def calculate_radius(image, left_fit, right_fit):
    y_eval = 719
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
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

Final video output is available [here](./carnd_p4_output.mp4) or on [Youtube](https://youtu.be/c-yJ2WJaFRA)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The thresholding mechanism is still quite unable to cope well with big variations in the color of the asphalt or, more in general, with variations in the lights and shadows: using a thresholding on HLS improved a lot the performances when compared to the first project in the Udacity course, but it is still not sufficient for any kind of environment.

The pipeline performs acceptably well on the challenge video, but it fails quite bad with the harder challenge video: very sharp bends are not detected correctly.

Something I would like to improve:

* sanity check (are lanes parallel? does the radius of curvature make sense)
* thresholding
* overall performances (the pipeline, even on a quite powerful computer, is not fast enough to be used on a real stream coming from a dash camera for example)
