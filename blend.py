import cv2
import numpy as np

image_paths = ['images/1_fullsize.jpg', 'images/2_fullsize.jpg', 'images/3_fullsize.jpg']
images = []
for name in image_paths:
    image = cv2.imread(name)
    images.append(image)

# generate Gaussian pyramid for images[0]
G = images[0].copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)
# show the piramid
# for i in range(7):
#     cv2.imshow(str(i), gpA[i])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# generate Gaussian pyramid for images[1]
G = images[1].copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)
# generate Gaussian pyramid for images[2]
G = images[2].copy()
gpC = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpC.append(G)


depth_of_laplacian = 2
# generate Laplacian Pyramid for images[0]
lpA = [gpA[depth_of_laplacian]]
for i in range(depth_of_laplacian,0,-1):
    GE = cv2.pyrUp(gpA[i])
    # print(GE.shape, gpA[i-1].shape)
    # resize GE to gpA[i-1].shape
    GE = cv2.resize(GE, gpA[i-1].shape[:2][::-1])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

# generate Laplacian Pyramid for images[1]
lpB = [gpB[depth_of_laplacian]]
for i in range(depth_of_laplacian,0,-1):
    GE = cv2.pyrUp(gpB[i])
    GE = cv2.resize(GE, gpA[i - 1].shape[:2][::-1])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

# generate Laplacian Pyramid for images[2]
lpC = [gpC[depth_of_laplacian]]
for i in range(depth_of_laplacian,0,-1):
    GE = cv2.pyrUp(gpC[i])
    GE = cv2.resize(GE, gpA[i - 1].shape[:2][::-1])
    L = cv2.subtract(gpC[i-1],GE)
    lpC.append(L)


# human defined boundary
ratio1 = 755/2053
ratio2 = 1182/2053

# Now add left and right halves of images in each level
LS = []
for la,lb,lc in zip(lpA,lpB,lpC):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:int(cols * ratio1)], lb[:,int(cols * ratio1):int(cols * ratio2)], lc[:,int(cols * ratio2):]))
    LS.append(ls)
# now reconstruct
ls_ = LS[0]
for i in range(1,depth_of_laplacian+1):
    ls_ = cv2.pyrUp(ls_)
    # print(ls_.shape, LS[i].shape)
    ls_ = cv2.resize(ls_, LS[i].shape[:2][::-1])
    ls_ = cv2.add(ls_, LS[i])
# image with direct connecting each half
real = np.hstack((images[0][:,:int(cols * ratio1)],images[1][:,int(cols * ratio1):int(cols * ratio2)], images[2][:,int(cols * ratio2):]))
cv2.imwrite('Pyramid_blending.jpg',ls_)
cv2.imwrite('Direct_blending.jpg',real)