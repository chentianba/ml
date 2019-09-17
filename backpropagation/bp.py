import numpy as np
import matplotlib.pyplot as plt

fa = lambda x : np.maximum(0, x)
dfa = lambda x : 1
alpha = 0.0001

w1 = np.random.rand(5, 2)
b1 = np.random.rand(5, 1)
w2 = np.random.rand(1, b1.shape[0])
b2 = np.random.rand(1, 1)

def output(x):
    global w1, w2
    y1 = fa(np.dot(w1, x) + b1)
    return fa(np.dot(w2, y1) + b2)

def update(x, z):
    global w1, w2, b1, b2
    y1 = fa(np.dot(w1, x) + b1)
    y = fa(np.dot(w2, y1) + b2)

    delta = y - z
    dy1 = np.dot(np.transpose(w2), delta)
    diff2 = alpha*np.dot(delta, np.transpose(y1))
    bdiff2 = alpha*np.sum(delta, axis=1)[:, np.newaxis]
    w2 = w2 - diff2
    b2 = b2 - bdiff2
    diff1 = alpha*np.dot(dy1, np.transpose(x))
    bdiff1 = alpha*np.sum(dy1, axis=1)[:, np.newaxis]
    w1 = w1 - diff1
    b1 = b1 - bdiff1
    print(diff1, diff2, bdiff1, bdiff2)
    print("="*50)
    if (np.abs(diff1) < 0.001).all() and (np.abs(diff2) < 0.001).all() and (np.abs(bdiff2) < 0.001).all() and (np.abs(bdiff1) < 0.001).all():
        return True
    else:
        return False

def picture(x, z, c1='r', c2='b'):
    # x.shape=(?, 2) z.shape=(1, ?)
    # plt.figure()
    print(x.shape, z.transpose().shape)
    for e in zip(x, z.transpose()):
        a,b,c = e[0][0], e[0][1], e[1][0]
        # print(e, a, b, c)
        if c > 0.5:
            plt.scatter(a, b, c=c1)
        else:
            plt.scatter(a, b, c=c2)

n = 1000
# x.shape = (2, ?) z.shape = (1, ?)
x = np.random.rand(2, n)
""" pattern one: multiple lines """
z = []
for e in np.transpose(x):
    tmp = 0
    if e[0] > 0.5 and e[0]*2 - e[1] > 1:
        tmp = 1
    if e[0] < 0.5 and e[0]*2 + e[1] < 1:
        tmp = 1
    z.append(tmp)
z = np.array([z])
""" pattern two: one line """
z = np.array([[(1 if e[0] < e[1] else 0) for e in np.transpose(x)]])
# print(x, z)
while True:
    flag = update(x, z)
    if flag:
        break
print(w1, w2)
print("*"*30)
x_ = np.linspace(0, 1, 50)
y_ = np.linspace(0, 1, 50)
xx = []
zz = []
for i in x_:
    for j in y_:
        xx.append([i, j])
xx = np.array(xx)
zz = output(xx.transpose())
picture(xx, zz, c1='y', c2='g')
picture(x.transpose(), z)
plt.show()
