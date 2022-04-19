# (1) import the necessary packages
import argparse
import imutils
import cv2

# (2) construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", 
                required=True, 
                help="path to the input image")
args = vars(ap.parse_args())

# (3) load the image, 
# convert it to grayscale, 
# blur it slightly,
# and threshold it

'''
 Apply grayscale 
 -> Gaussian smoothing sử dụng 5x5 kernel, 
 và cuối cùng là thresholding. 
 
 Output của bước tiền xử lý là 
 1 ảnh nền đen và các hình màu trắng nổi lên. 
'''
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('/home/anhp/Documents/coding/learnOpenCV/tìm_điểm_chính_giữa_của_1hình/out/shapes_gray.jpg', gray)
cv2.imshow("Gray",gray)
cv2.waitKey(0)


blurred = cv2.GaussianBlur(gray, (5,5), 0)
cv2.imwrite('/home/anhp/Documents/coding/learnOpenCV/tìm_điểm_chính_giữa_của_1hình/out/shapes_blurred.jpg', blurred)
cv2.imshow('Blurred', blurred)
cv2.waitKey(0)



thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('/home/anhp/Documents/coding/learnOpenCV/tìm_điểm_chính_giữa_của_1hình/out/shapes_thresh.jpg', thresh)
cv2.imshow('Thresh', thresh)
cv2.waitKey(0)



# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), 
                        cv2.RETR_EXTERNAL,
	                    cv2.CHAIN_APPROX_SIMPLE)
# Đoạn code trên trả về 1 list 
# tương ứng với số đa giác (mảng màu trắng) 
# tìm kiếm được trên hình.
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
print(cnts)

