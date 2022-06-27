import math

import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import (EssentialMatrixTransform,
                               FundamentalMatrixTransform)


class SelectVideo(object):
    def __init__(self) -> None:
        pass

    def select(self):
        # cap = cv2.VideoCapture('labeled/0.hevc')  #NIGHT HIGHWAY
        # cap = cv2.VideoCapture('labeled/1.hevc')  #NIGHT HIGHWAY
        # cap = cv2.VideoCapture('labeled/2.hevc')  #Day + Blur
        cap = cv2.VideoCapture('labeled/3.hevc')  # NIGHT CITY
        # cap = cv2.VideoCapture('labeled/4.hevc')  #Day
        # cap = cv2.VideoCapture('unlabeled/5.hevc')#Day HIGHWAY
        # cap = cv2.VideoCapture('unlabeled/6.hevc')#DAY RAIN
        # cap = cv2.VideoCapture('unlabeled/7.hevc')#NIGHT Suburb
        # cap = cv2.VideoCapture('unlabeled/8.hevc')#DAY HIGHWAY
        # cap = cv2.VideoCapture('unlabeled/9.hevc')#SNOW

        return cap


class FeatureDetector(object):
    def __init__(self, K) -> None:
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        self.last = None

    def normalize(self, pts):
        newMat = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        return np.dot(self.Kinv, newMat.T).T[:, 0:2]

    def denormalize(self, pt):
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        return int(round(ret[0])), int(round(ret[1]))

    def extractRt(self, E, needJustR=False):

        W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        U, s, Vh = np.linalg.svd(E)

        assert np.linalg.det(U) > 0

        if np.linalg.det(Vh) < 0:
            Vh *= -1.0
        R = np.dot(np.dot(U, W), Vh)

        if np.sum(R.diagonal()) < 0:
            R = np.dot(np.dot(U, W.T), Vh)

        t = U[:, 2]
        Rt = np.concatenate([R, t.reshape(3, 1)], axis=1)
        if needJustR:
            return R
        else:
            return Rt

    def extract(self, img, rotMatRequired=False):
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(
            np.uint8), 3000, qualityLevel=0.01, minDistance=3)

        # extract
        try:
            kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        except:
            kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)

        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            for m, n in matches:
                if m.distance < 0.55*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))

        Rt, R = None, None
        if len(ret) > 0:
            ret = np.array(ret)

            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])

            try:
                model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                        EssentialMatrixTransform,
                                        # FundamentalMatrixTransform,
                                        min_samples=8,
                                        residual_threshold=0.006,
                                        max_trials=200)
                ret = ret[inliers]

                if rotMatRequired:
                    R = self.extractRt(model.params, rotMatRequired)
                else:
                    Rt = self.extractRt(model.params)
            except:
                pass

        self.last = {"kps": kps, "des": des}

        if rotMatRequired:
            return ret, R
        else:
            return ret, Rt

    def yawpitchrolldecomposition(self, R):
        if R is not None:
            sin_x = math.sqrt(R[2, 0] * R[2, 0] + R[2, 1] * R[2, 1])
            validity = sin_x < 1e-6
            my_sin_x = math.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2])
            if not validity:
                z1 = math.atan2(R[1, 0], R[0, 0])
                x = math.atan2(-R[2, 0], my_sin_x)
                z2 = math.atan2(R[2, 1], R[2, 2])
            else:
                z1 = 0                               
                x = math.atan2(sin_x,  R[2, 2])   
                z2 = 0                               
        else:
            z1, x, z2 = 0, 0, 0

        return np.array([[z1], [x], [z2]])


class Processing(object):
    def __init__(self, fe):
        self.fe = fe

    def process_frame(self, img):
        height, width, z = img.shape
        bigFrame = True

        if bigFrame:
            img = cv2.resize(img, (width, height))
            img = img[0:640, :]
        else:
            img = cv2.resize(img, (width//2, height//2))
            img = img[0:320, :]

        matches, pose = self.fe.extract(img, True)
        if pose is None:
            pass

        for pt1, pt2 in matches:
            u1, v1 = self.fe.denormalize(pt1)
            u2, v2 = self.fe.denormalize(pt2)

            if round(sum([((u1, v1)[x] - (u2, v2)[x]) ** 2 for x in range(len((u1, v1)))]) ** 0.5) > 40:
                pass
            else:
                cv2.circle(img, (u1, v1), color=(0, 0, 200), radius=2)
                cv2.line(img, (u1, v1), (u2, v2), color=(255, 255, 255))

        ypr = self.fe.yawpitchrolldecomposition(pose)

        return img, ypr[0], ypr[1]


if __name__ == "__main__":
    def drawImg(title, frame):
        cv2.imshow(title, frame)
        cv2.waitKey(1)

    F = 910
    height = 880
    width = 1200

    K = np.array(([F, 0, width//2], [0, F, height//2], [0, 0, 1]))
    fe = FeatureDetector(K)
    check = Processing(fe)

    cap = SelectVideo().select()

    isFirstFrame = True
    while cap.isOpened():
        ret, currentFrame = cap.read()
        if ret == True:
            img, y, p = check.process_frame(currentFrame)
            drawImg("Keypoints on Img", img)
        else:
            break
