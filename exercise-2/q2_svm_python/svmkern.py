import numpy as np
from kern import kern
import cvxopt


def svmkern(X, t, C, p):
    # Non-Linear SVM Classifier
    #
    # INPUT:
    # X        : the dataset                        (num_samples x dim)
    # t        : labeling                           (num_samples x 1)
    # C        : penalty factor the slack variables (scalar)
    # p        : order of the polynom               (scalar)
    #
    # OUTPUT:
    # sv       : support vectors (boolean)          (1 x num_samples)
    # b        : bias of the classifier             (scalar)
    # slack    : points inside the margin (boolean) (1 x num_samples)

    #####Insert your code here for subtask 2d#####
  # Get dimensions of data
    num_samples, dim = X.shape

    # Construct quadratic matrix of the problem
    H = np.zeros((num_samples, num_samples))
    for n in range(num_samples):
      H[n,n] = t[n] * t[n] * kern(X[n, :], X[n, :],p)
      for m in range(n+1, num_samples):
        H[n,m] = t[n] * t[m] * kern(X[n, :], X[m, :],p)
        H[m,n] = H[n,m]
 
    # vector to sum the lagrange multipliers in the objective function
    q = (-1) * np.ones(num_samples)

    # equation for the upper and lower boundaries
    G = np.vstack([-np.eye(num_samples,dtype=np.float),
     np.eye(num_samples, dtype=np.float)])
    h = np.hstack([np.zeros((num_samples)),
     C*np.ones((num_samples))]).reshape(-1,1).astype(np.float)

    # complementary slackness
    A = t.reshape(1,-1).astype(np.float)
    b = 0.0

    # get lagrange multipliers
    alpha = cvxopt.solvers.qp(P=cvxopt.matrix(H), 
                              q=cvxopt.matrix(q), 
                              G=cvxopt.matrix(G), 
                              h=cvxopt.matrix(h), 
                              A=cvxopt.matrix(A), 
                              b=cvxopt.matrix(b))
    alpha = np.array(alpha["x"])
    
    # support vectors and weights
    sv = (alpha > 1e-8).reshape(-1)
    w = np.zeros((dim,))
    for n in range(num_samples):
      w += alpha[n] * t[n] * X[n, :]

    # determining the bias term
    sv_idxs = np.array(range(len(sv)))
    for idx in sv_idxs[sv]:
        b += t[idx] - np.dot(w, X[idx, :])
    b /= t[sv].shape[0]

    # support vectors on the border
    slack = alpha == C
    result = np.dot(X, w) + b

    return alpha, sv, b, result, slack
