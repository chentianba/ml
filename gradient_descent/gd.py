import numpy as np
import matplotlib.pyplot as plt 

sx = np.arange(-5, 5, 0.5)
error = np.random.uniform(-1, 1, (len(sx)))
sy = sx + error

def gd(x, y, alpha = 0.0001):
    # format: x.shape=(m, n) y.shape=(m, 1)
    # alpha is step
    print(x.shape, y.shape)
    m = x.shape[0]
    theta = np.random.rand(x[0].shape[0]+1, 1)
    one = np.ones((m, 1))
    X = np.hstack((one, x))
    print(theta)
    theta_old = np.zeros(theta.shape)
    loss = []
    while not (np.abs(theta - theta_old) < 0.0001).all():
        theta_old = theta
        theta = theta - alpha*np.transpose(X).dot(X.dot(theta)-y)
        print(np.linalg.norm(theta - theta_old), theta-theta_old)
        loss.append(np.linalg.norm(X.dot(theta)-y))
    plt.figure()
    plt.plot(range(len(loss)), loss)
    return theta
if not True:
    # pattern one: linear, one feature
    sx1 = np.array([[x] for x in sx])
    sy1 = np.array([[y] for y in sy])
    theta = gd(sx1, sy1, 0.01)
    print("theta: ", theta)
    plt.figure()
    plt.scatter(sx1[:, 0], sy1[:, 0])
    plt.plot(sx1[:, 0], sx1[:, 0]*theta[1, 0]+theta[0, 0], c='r')
    plt.show()
if True:
    # pattern two: 2D, two features
    x1 = np.linspace(0, 1, 20)[:, np.newaxis]
    x2 = np.random.rand(20, 1)
    x = np.hstack((x1, x2))
    y = np.ones(x1.shape)
    y[x[:, 0] < x[:, 1]] = -1
    theta = gd(x, y, 0.001)
    print(x)
    print(theta)
    mp = np.array(['r', 'b'])
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=mp[(y == -1).astype(int)].transpose()[0])
    xx = np.linspace(0, 1, 20)
    plt.plot(xx, -xx*theta[1, 0]/theta[2,0]+(0-theta[0, 0])/theta[2,0], c='y')
    # plt.plot(xx, -xx*theta[1, 0]/theta[2,0]+(1-theta[0, 0])/theta[2,0], c='gray')
    # plt.plot(xx, -xx*theta[1, 0]/theta[2,0]+(-1-theta[0, 0])/theta[2,0], c='gray')
    plt.show()
