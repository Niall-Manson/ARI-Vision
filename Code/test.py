import cv2
import numpy as np
import time

def main():
    cam = cv2.VideoCapture(0)
    background = find_background(cam)
    open_cam(cam, background)

def find_background(cam):
    frames = []
    for _ in range(50):
        _, frame = cam.read()
        frames.append(frame)

    median = np.median(frames, axis=0).astype(np.uint8)

    return median 

def open_cam(cam, background):
    #for finding fps
    start = time.time()
    timeFrameTaken = []

    while True:
        #reads frame
        _, frame = cam.read()
        cv2.imshow("frame", frame)

        foreground, displayedFrame = bg_removal(frame, background)
        cv2.imshow("foreground", foreground)
        cv2.imshow("UI", displayedFrame)

        fps, timeFrameTaken = finding_fps(timeFrameTaken, start)
        print(f"fps: {fps}")

        #exitting
        k = cv2.waitKey(1)
        if k%256 == 32:
            # SPACE pressed
            break

#only used when debugging
def image_show_large(image):
    h, w = image.shape[:2]
    sizeMultiplier = 750/max(h, w)
    temp_image = cv2.resize(image, (int(h*sizeMultiplier), int(w*sizeMultiplier)), interpolation=cv2.INTER_AREA)
    cv2.imshow("test", temp_image)
    cv2.waitKey(0)

def bg_removal(frame, background):
    frameGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    backgroundGray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    uncroppedMask = find_foreground(frameGray, backgroundGray, threshold=50)
    centreX, centreY = find_centre(uncroppedMask)

    if centreX == None:
        return uncroppedMask, frame #if this occurs, uncroppedMask will be pure black (0)
        
    else:
        upperEdge, lowerEdge, leftEdge, rightEdge = crop_foreground(uncroppedMask, centreX, centreY)
        croppedMask = uncroppedMask[upperEdge:lowerEdge, leftEdge:rightEdge]
        croppedMask = np.where(croppedMask == 100, 255, croppedMask) #crop_foreground() changes some uncroppedMask pixels from 255 to 100, don't know why. this fixes it
        croppedMask = filling_holes_fast(croppedMask)
        croppedMask = np.array(croppedMask, dtype='uint8') #idk why this is needed but throws error if removed
        croppedFrame = frame[upperEdge:lowerEdge, leftEdge:rightEdge]
        foreground = show_foreground(croppedFrame, croppedMask)
        
        displayedFrame = cv2.rectangle(frame, (leftEdge, upperEdge), (rightEdge, lowerEdge), (0, 255, 0), 3)
        
        return foreground, displayedFrame

def finding_fps(timeFrameTaken, start):
    timeFrameTaken.append(time.time() - start - sum(timeFrameTaken)) #finds time taken between current frame and previous frame

    if len(timeFrameTaken) < 20:
        avgTimePerFrame = sum(timeFrameTaken)/len(timeFrameTaken)
    else:
        avgTimePerFrame = sum(timeFrameTaken[-20:])/20
    fps = 1/avgTimePerFrame

    return fps, timeFrameTaken

def find_foreground(image1, image2, threshold):
    diff_frame = cv2.absdiff(image1, image2)
    _, diff = cv2.threshold(diff_frame, threshold, 255, 
                            cv2.THRESH_BINARY
                            )
    return diff

def find_centre(mask):
    h, w = mask.shape[:2]
    xVals = 0
    yVals = 0
    pointTally = 0

    for y in range(h):
        for x in range(w):
            if mask[y][x] == 255:
                xVals += x
                yVals += y
                pointTally += 1

    #avoiding division by 0
    if pointTally == 0:
        return None, None
    else:
        xAve = int(xVals/pointTally)
        yAve = int(yVals/pointTally)

    return xAve, yAve

#fills holes, returns updated binary image
def filling_holes_fast(binaryImage):
    h, w = binaryImage.shape[:2]    
    
    for y in range(h):
        for x in range(w):
            if binaryImage[y][x] == 0:
                mask = np.zeros((h+2, w+2), np.uint8)
                cv2.floodFill(binaryImage, mask, (x,y), 100) #temp, removed before moving onto next pixel

                firstRow = binaryImage[0]
                lastRow = binaryImage[-1]
                firstColumn = binaryImage[:, 0]
                lastColumn = binaryImage[:, -1]
                edges = np.concatenate((firstRow, lastRow, firstColumn, lastColumn), axis=None)
                
                #replaces where pixel=100 with correct val (50 when background, 255 when foreground)
                if 100 in edges:
                    binaryImage = np.where(binaryImage == 100, 50, binaryImage) #will be (0, 0, 0) in finalImage
                else:
                    binaryImage = np.where(binaryImage == 100, 255, binaryImage)
    
    #converting when 50 to 0 for consistency
    binaryImage = np.where(binaryImage == 50, 0, 255)
    return binaryImage

def show_foreground(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

def crop_foreground(mask, centreX, centreY):
    h, w = mask.shape[:2]

    #highlights main object
    fgMask = mask
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(fgMask, mask, (centreX,centreY), 100)

    #creates square at fg centre, continues expanding until it contains all of fg (pixel = 100)
    targetFgPixels = np.count_nonzero(fgMask == 100)
    upperEdge = centreY
    lowerEdge = centreY+1
    leftEdge = centreX
    rightEdge = centreX+1

    tempImage = fgMask[upperEdge:lowerEdge, leftEdge:rightEdge]
    while np.count_nonzero(tempImage == 100) < targetFgPixels:
        #vertically increasing
        currentFgPixels = np.count_nonzero(tempImage == 100)
        tempUpperEdge = max(0, upperEdge-1)
        tempLowerEdge = min(lowerEdge+1, h)

        newTempImage = fgMask[tempUpperEdge:tempLowerEdge, leftEdge:rightEdge]
        if np.count_nonzero(newTempImage == 100) > currentFgPixels:
            upperEdge = tempUpperEdge
            lowerEdge = tempLowerEdge
            tempImage = fgMask[upperEdge:lowerEdge, leftEdge:rightEdge]

        #horizontally increasing
        currentFgPixels = np.count_nonzero(tempImage == 100)
        tempLeftEdge = max(0, leftEdge-1)
        tempRightEdge = min(rightEdge+1, w)

        newTempImage = fgMask[upperEdge:lowerEdge, tempLeftEdge:tempRightEdge]
        if np.count_nonzero(newTempImage == 100) > currentFgPixels:
            leftEdge = tempLeftEdge
            rightEdge = tempRightEdge
            tempImage = fgMask[upperEdge:lowerEdge, leftEdge:rightEdge]
    
    return upperEdge, lowerEdge, leftEdge, rightEdge
    
main()