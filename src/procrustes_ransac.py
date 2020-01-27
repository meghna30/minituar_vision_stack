
import numpy as np
import pdb


def ProcrustesRansac(src_pnts, dst_pnts):

    err_thresh = 0.005
    prev_inliers = 0
    idx_all = np.arange(0,len(dst_pnts))
    no_iterations = 1000


    for iter in range(0, no_iterations):

        idx = np.random.randint(len(dst_pnts), size = 4)
        dst = dst_pnts[idx, :]
        src = src_pnts[idx, :]

        R, t = ComputeH(src, dst)
        dst_predicted = np.matmul(R,dst_pnts.T).T + np.reshape(t,[1,-1])
        err = np.sqrt((src_pnts[:,0] - dst_predicted[:,0])**2 +  (src_pnts[:,1] - dst_predicted[:,1])**2 + (src_pnts[:,2] - dst_predicted[:,2])**2)
        # err = np.sqrt(err[:,0]**2 + err[:,1]**2 + err[:,2]**2)


        inlier_idx = idx_all[err < err_thresh]



        if len(inlier_idx) > prev_inliers:
            prev_inliers = len(inlier_idx)
            src_inliers = src_pnts[inlier_idx,:]
            dst_inliers = dst_pnts[inlier_idx,:]
    if prev_inliers > 4:
        R,t = ComputeH(src_inliers, dst_inliers)
        print(len(src_inliers), len(src_pnts))
    else:
        R = np.eye(3)
        t = np.zeros(3)
        print("no inlier")

    return R,t


def ComputeH(src, dst):

    dst_c = np.mean(dst, axis = 0)
    src_c = np.mean(src, axis = 0)

    dst_centered = dst - dst_c
    src_centered = src - src_c

    H = np.matmul(dst_centered.T, src_centered)
    U,S,Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = np.matmul(Vt.T, U.T)

    t = np.mean(src -  np.matmul(R,dst.T).T , axis = 0)
    # t = src_c -  np.matmul(R,dst_c)

    return R, t


def umeyama(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t
