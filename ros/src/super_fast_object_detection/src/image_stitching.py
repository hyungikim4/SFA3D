import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2 as cv
import cv2
import imutils
import numpy as np

def detectAndDescribe(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    # convert the keypoints from KeyPoint objects to NumPy
    # arrays
    kps = np.float32([kp.pt for kp in kps])
    # return a tuple of keypoints and features
    return (kps, features)
def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    # compute the raw matches and initialize the list of actual
    # matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
        
    # loop over the raw matches
    for m in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
                
    # computing a homography requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            reprojThresh)
                
                
        # return the matches along with the homograpy matrix
        # and status of each matched point
        return (matches, H, status)
            
    # otherwise, no homograpy could be computed
    return None

concat_img = cv2.imread('/home/khg/Python_proj/SFA3D/dataset/veloster/training/front_image/000000.png')
h, total_w, c = concat_img.shape
w = int(total_w/3)

imageA = concat_img[:, :w, :]
imageB = concat_img[:, w:2*w, :]
fr_img = concat_img[:, 2*w:, :]

(kpsA, featuresA) = detectAndDescribe(imageA)
(kpsB, featuresB) = detectAndDescribe(imageB)

# match features between the two images
ratio=0.75
reprojThresh=4.0

M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
(matches, H, status) = M
cv2.imshow('ori_img', imageA)
print(H)
result = cv2.warpPerspective(imageA, H, (2*imageA.shape[1], 2*imageA.shape[0]))

cv2.imshow('asdf',result)
result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

cv2.imshow('test',result)
cv2.waitKey(0)
# fl_img = cv2.resize(fl_img, (int(w/4), int(h/4)))
# front_img = cv2.resize(front_img, (int(w/4), int(h/4)))
# fr_img = cv2.resize(fr_img, (int(w/4), int(h/4)))

# # cv2.imshow('left', fl_img)
# # cv2.imshow('front', front_img)
# # cv2.imshow('right', fr_img)
# cv2.setNumThreads(1)
# modes = (cv2.Stitcher_PANORAMA, cv2.Stitcher_SCANS)
# imgs = [fl_img, front_img, fr_img]
# stitcher = cv2.createStitcher()
# status, pano = stitcher.stitch(imgs)

# # print(statue)
# # cv2.imshow('test', pano)
# # cv2.waitKey(0)



# FLANN_INDEX_LSH    = 6

# def anorm2(a):
#     return (a*a).sum(-1)
# def anorm(a):
#     return np.sqrt( anorm2(a) )

# def matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2):

#     flann_params= dict(algorithm = FLANN_INDEX_LSH,
#                        table_number = 6, # 12
#                        key_size = 12,     # 20
#                        multi_probe_level = 1) #2



#     matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
#     raw_matches = matcher.knnMatch(descriptors1, descriptors2, k = 2) #2

    

#     matches = []
#     for m in raw_matches:
#         if len(m) == 2 and m[0].distance < m[1].distance * 0.79:
#             matches.append((m[0].trainIdx, m[0].queryIdx))


#     if len(matches) >= 4:

#         keyPoints1 = np.float32([keyPoints1[i] for (_, i) in matches])
#         keyPoints2 = np.float32([keyPoints2[i] for (i, _) in matches])


#         H, status = cv.findHomography(keyPoints1, keyPoints2, cv.RANSAC,4.0)

#         print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
#     else:
#         H, status = None, None
#         print('%d matches found, not enough for homography estimation' % len(p1))


#     return matches, H, status


   
# def drawMatches(image1, image2, keyPoints1, keyPoints2, matches, status):


#     h1, w1 = image1.shape[:2]
#     h2, w2 = image2.shape[:2]



#     img_matching_result = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")



#     img_matching_result[0:h2, 0:w2] = image2
#     img_matching_result[0:h1, w2:] = image1



#     for ((trainIdx, queryIdx), s) in zip(matches, status):

#         if s == 1:
#             keyPoint2 = (int(keyPoints2[trainIdx][0]), int(keyPoints2[trainIdx][1]))
#             keyPoint1 = (int(keyPoints1[queryIdx][0]) + w2, int(keyPoints1[queryIdx][1]))
#             cv.line(img_matching_result, keyPoint1, keyPoint2, (0, 255, 0), 1)



#     return img_matching_result


# def main():


#     # img1 = cv.imread('.\\images\\B.jpg') 
#     # img2 = cv.imread('.\\images\\A.jpg')
#     concat_img = cv.imread('/home/khg/Python_proj/SFA3D/dataset/veloster/training/front_image/000000.png')
#     h, total_w, c = concat_img.shape
#     w = int(total_w/3)

#     img1 = concat_img[:, :w, :]
#     img2 = concat_img[:, w:2*w, :]
#     fr_img = concat_img[:, 2*w:, :]
    
    

#     gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
#     gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    
 
#     detector = cv.BRISK_create()
#     keyPoints1, descriptors1 = detector.detectAndCompute(gray1, None)
#     keyPoints2, descriptors2 = detector.detectAndCompute(gray2, None)
#     print('img1 - %d features, img2 - %d features' % (len(keyPoints1), len(keyPoints2)))


    
#     keyPoints1 = np.float32([keypoint.pt for keypoint in keyPoints1])
#     keyPoints2 = np.float32([keypoint.pt for keypoint in keyPoints2])
    


#     matches, H, status = matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2)



#     img_matching_result = drawMatches(img1, img2, keyPoints1, keyPoints2, matches, status)



#     result = cv.warpPerspective(img1, H,
#         (img1.shape[1] + img2.shape[1], img1.shape[0]))
#     result[0:img2.shape[0], 0:img2.shape[1]] = img2


#     cv.imshow('result', result)
#     cv.imshow('matching result', img_matching_result)

#     cv.waitKey(0)

#     print('Done')

# if __name__ == "__main__":
#     main()
#     cv.destroyAllWindows()
    