#Importing Packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from moviepy.editor import VideoFileClip

import cv2
import numpy as np
#from goprocam import GoProCamera
#from goprocam import constants


vidcap = cv2.VideoCapture("test5.mp4")
#vidcap = cv2.VideoCapture("udp://10.5.5.9:8554")
success,gopro = vidcap.read()
success = True
i=0

lx1 = 0
lx2 = 0
ly1 = 0
ly2 = 0
rx1 = 0
rx2 = 0
ry1 = 0
ry2 = 0
present = 0


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform to the grayscaled image"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # Defining a blank mask to start with
    mask = np.zeros_like(img)

    # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    This function defines and draws `lines` with `color` and `thickness`.
    """
    # Initializing empty arrays
    right_xpoints=[]
    right_ypoints=[]
    l_xpoints=[]
    l_ypoints=[]

    sloper = []
    slopel = []
    slopes = []

    br = []
    bl = []
    bs = []
    global present, lx1 , lx2 , ly1 , ly2 , rx1 , rx2 , ry1 , ry2
    if lines is None:
        #left
        print('lol')
        cv2.line(img, (lx1, ly1), (lx2, ly2), [0, 255, 0], 6)
        #right
        cv2.line(img, (rx1, ry1), (rx2, ry2), [0, 255, 0], 6)

        #avgx1, avgy1, avgx2, avgy2 = avgLeft
        #cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [255,255,255], 8) #draw left line
        #avgx3, avgy3, avgx4, avgy4 = avgRight
        #cv2.line(img, (int(avgx3), int(avgy3)), (int(avgx4), int(avgy4)), [255,255,255], 8) #draw right line
        #cv2.line(img, (int((int(avgx3)+int(avgx1))/2), int((int(avgy3)+int(avgy1))/2)), (int((int(avgx4)+int(avgx2))/2), int((int(avgy4)+int(avgy2))/2)), [255,255,255], 8) #draw right line
        return



    for line in lines:
        for x1,y1,x2,y2 in line:
            size = math.hypot(x2 - x1, y2 - y1)
            #Finding the slope and intercept value
            slope = ((y2-y1)/(x2-x1))
            parameters = np.polyfit((x1, x2), (y1, y2), 1)

            #Filter the lines based on the slope value as right and left lanes
            if (slope > 0.3 and ((x1 and x2) > 320)):
              #right lane
                presentl = 1
                sloper.append(slope)
                br.append(parameters[1])
                #Adding right hough lines
                #cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            elif (slope < -0.3 and ((x1 and x2) < 320)):
                #left lane
                presentr = 1
                slopel.append(slope)
                bl.append(parameters[1])
                #Adding left hough lines
                #cv2.line(img, (x1, y1), (x2, y2), color, thickness)
#            elif (-0.2 < slope < 0.2 and ((y1 and y2 )< 350 )):
#                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    #Calculating mean slope and intercept values

    #right lane
    meanSloper = np.mean(sloper)
    meanBr = np.mean(br)

    if (present == 0):
         cv2.line(img, (rx1, ry1), (rx2, ry2), [0, 255, 0], 6)
         cv2.line(img, (lx1, ly1), (lx2, ly2), [0, 255, 0], 6)
    #finding two points to fit the right lane line
    x1 = 640
    if(np.isnan(meanSloper) or np.isnan(meanBr)):
      pass
    else:

        y1 = meanSloper * x1 + meanBr
        y2 = 260
        x2 = ( y2 - meanBr ) / meanSloper
        #Plotting the right lane
        if(math.isfinite(y1) and math.isfinite(x2)):
            cv2.line(img, (x1, int(y1)), (int(x2), y2), [0, 255, 0], 6)
        #print('lol')

        #right

        lx1 = x1
        lx2 = int(x2)
        ly1 = int(y1)
        ly2 = y2
    #left lane
    meanSlopel = np.mean(slopel)
    meanBl = np.mean(bl)

    #finding two points to fit the left lane line
    x1 = 0
    if(np.isnan(meanSlopel) or np.isnan(meanBl) or math.isinf(meanSlopel) or math.isinf(meanBl)):
      pass
    else:
        y1 = meanSlopel * x1 + meanBl
        y2 = 260
        x2 = ( y2 - meanBl ) / meanSlopel
        #Plotting the left lane
        if(math.isfinite(y1) and math.isfinite(x2)):
            cv2.line(img, (x1, int(y1)), (int(x2), y2), [0, 255, 0], 6)

        rx1 = x1
        rx2 = int(x2)
        ry1 = int(y1)
        ry2 = y2


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Takes the outout of canny as the input image and
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    """
    Takes the output image with hough lines as input

    The result image is computed as follows:

    initial_img * a + img * ß + ?
    """
    return cv2.addWeighted(initial_img, a, img, b, c)

# Import everything needed for creating the video clips


def process_image(image):
    # Resizing the input image to a more resonable size for processing
    image = cv2.resize(image, (640, 480), interpolation = cv2.INTER_AREA)





    # Blur to avoid edges from noise
    blurredImage = gaussian_blur(image, 7)

    # Detect edges using canny
    edgesImage = canny(blurredImage, 100, 140)

    # Mark out the vertices for region of interest
    vertices = np.array( [[
                [0, 480],
                [0, 320],
                [200, 280],
                [520, 260],
                [640, 300],
                [640, 480]
            ]], dtype=np.int32 )

    # Mask the canny output with region of interest
    regionInterestImage = region_of_interest(edgesImage, vertices)

    # Drawing the hough lines in the Masked Canny image
    lineMarkedImage = hough_lines(regionInterestImage, 1, np.pi/180, 35, 15, 100)

    # Test detected edges by uncommenting this
    # return cv2.cvtColor(regionInterestImage, cv2.COLOR_GRAY2RGB)

    # Draw output on top of original
    return weighted_img(lineMarkedImage, image)

while success:
  i=i+1

  presentr = 0
  presentl = 0
  if  gopro is not None:

      mah = process_image(gopro)
      #cv2.imshow('frame',mah)
      #cv2.imwrite("frame"+str(i)+".jpg", mah)
      key = cv2.waitKey(1)
      if(key == ord('q')):
          break
  success,gopro = vidcap.read()

  print('Read a new frame: ', success)

vidcap.release()
cv2.destroyAllWindows()
