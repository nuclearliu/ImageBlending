import cv2
import numpy as np

image_paths = ['images/1_fullsize.jpg', 'images/2_fullsize.jpg', 'images/3_fullsize.jpg']
images = []
for name in image_paths:
    image = cv2.imread(name).astype(np.float32)
    images.append(image)

# generate Gaussian pyramid for images[0]
G = images[0].copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    # print(G.dtype)
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


depth_of_laplacian = 5
# generate Laplacian Pyramid for images[0]
lpA = [gpA[depth_of_laplacian]]
for i in range(depth_of_laplacian,0,-1):
    GE = cv2.pyrUp(gpA[i])
    # print(GE.shape, gpA[i-1].shape)
    # resize GE to gpA[i-1].shape
    GE = cv2.resize(GE, gpA[i-1].shape[:2][::-1])
    # L = cv2.subtract(gpA[i-1],GE)
    # print(GE.dtype)
    L = gpA[i-1] - GE
    # print(gpA[i-1].shape, GE.shape, L.shape)
    lpA.append(L)

# generate Laplacian Pyramid for images[1]
lpB = [gpB[depth_of_laplacian]]
for i in range(depth_of_laplacian,0,-1):
    GE = cv2.pyrUp(gpB[i])
    GE = cv2.resize(GE, gpA[i - 1].shape[:2][::-1])
    # L = cv2.subtract(gpB[i-1],GE)
    L = gpB[i-1] - GE
    lpB.append(L)

# generate Laplacian Pyramid for images[2]
lpC = [gpC[depth_of_laplacian]]
for i in range(depth_of_laplacian,0,-1):
    GE = cv2.pyrUp(gpC[i])
    GE = cv2.resize(GE, gpA[i - 1].shape[:2][::-1])
    # L = cv2.subtract(gpC[i-1],GE)
    L = gpC[i-1] - GE
    lpC.append(L)


# human defined boundary
ratio1 = 755/2053
ratio2 = 1182/2053

mix_ratio = 1/80

# blend the piramids
LS = []
for la,lb,lc in zip(lpA,lpB,lpC):
    rows,cols,dpt = la.shape

    boundary1 = int(cols * ratio1)
    boundary2 = int(cols * ratio2)
    halfwidth = int(cols * mix_ratio/2)
    maska = np.concatenate([np.ones(boundary1-halfwidth), np.linspace(1,0,2*halfwidth), np.zeros(cols-boundary1-halfwidth)]).reshape(1,cols,1).repeat(rows, axis=0).repeat(3, axis=2)
    maskb = np.concatenate([np.zeros(boundary1-halfwidth), np.linspace(0, 1, 2*halfwidth), np.ones(boundary2-boundary1-2*halfwidth), np.linspace(1,0,2*halfwidth), np.zeros(cols-boundary2-halfwidth)]).reshape(1,cols,1).repeat(rows, axis=0).repeat(3, axis=2)
    maskc = np.concatenate([np.zeros(boundary2-halfwidth), np.linspace(0, 1, 2*halfwidth), np.ones(cols-boundary2-halfwidth)]).reshape(1,cols,1).repeat(rows, axis=0).repeat(3, axis=2)
    # ls = np.hstack((la[:, 0:int(cols * ratio1)], lb[:, int(cols * ratio1):int(cols * ratio2)], lc[:, int(cols * ratio2):]))
    ls = la * maska + lb * maskb + lc * maskc
    LS.append(ls)
# reconstruct
ls_ = LS[0]
for i in range(1,depth_of_laplacian+1):
    ls_ = cv2.pyrUp(ls_)
    # print(ls_.dtype, LS[i].dtype)
    ls_ = cv2.resize(ls_, LS[i].shape[:2][::-1])
    # ls_ = cv2.add(ls_, LS[i])
    ls_ = ls_ + LS[i]
ls_ = np.clip(ls_, 0, 255).astype(np.uint8)
# image with direct connecting each half
real = np.hstack((images[0][:,:int(cols * ratio1)],images[1][:,int(cols * ratio1):int(cols * ratio2)], images[2][:,int(cols * ratio2):]))
cv2.imwrite('Pyramid_blending.jpg',ls_)
cv2.imwrite('Direct_blending.jpg',real)