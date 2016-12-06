from sklearn import datasets
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import math
from random import randint

data = datasets.load_iris()
# X = data.data[:100,:2]
# Y = data.target[:100]
# #X_full = data.data[:100,:]
#X, Y = make_blobs(n_samples=5000, centers=2, n_features=2,cluster_std=10.0 ,center_box=(-10.0,10.0) ,shuffle = True, random_state= 10)

#data1
X, Y = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=3, center_box=(-10.0, 10.0), shuffle=True,random_state=1)
#data 2
#X, Y = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=15, center_box=(-10.0, 10.0), shuffle=True,random_state=1)
X_test30 = (X[900:,:])
Y_test30 = Y[900:]
#To add error at the end
# for i in range(Y_test30.size):
#     a = X_test30[i][0]
#     b = X_test30[i][1]
#     c =0
#     if(randint(0,1)==0):
#         a = a+ randint(0,9)+ randint(0,9)
#         b = b- randint(0,9)- randint(0,9)
#         c = 1
#     else:
#         a = a- randint(0,9)- randint(0,9)
#         b = b+ randint(0,9)+ randint(0,9)
#         c =0
#     X[900+i][0] = a
#     X[900+i][1] = b
#     Y[900+i] = c

#To add noise in intermidiate data
# k = 0
# for j in range(100):
#     a = X[k][0]
#     b = X[k][1]
#     c =0
#     if(randint(0,1)==0):
#         a = a+ randint(0,9)+ randint(0,9)
#         b = b- randint(0,9)- randint(0,9)
#         c = 1
#     else:
#         a = a- randint(0,9)- randint(0,9)
#         b = b+ randint(0,9)+ randint(0,9)
#         c =0
#     X[k][0] = a
#     X[k][1] = b
#     Y[k] = c
#     k = k+9




# X_new = np.zeros((X.shape[0],3))
# i =0
# for item in X:
#     # X_new[i][0] = X[i][0]
#     # X_new[i][1] = X[i][1]
#     X_new[i][0] = X[i][0]*X[i][0]
#     X_new[i][1] = X[i][0]*X[i][1]
#     X_new[i][2] = X[i][1]*X[i][1]
#     i = i +1
# X = X_new

#alpha = 100 # GD
alpha = 200
#code for logit - start
def grad_desc(theta_values, X, y, L2= False, lr = 0.01, converge_change = 0.0001):
    #standardizing X

    X = (X-np.mean(X,axis=0)) / np.std(X,axis=0)
    bias = 1
    X = np.hstack ((X, [[bias]] * len (X) ))
    cost_iter = []
    theta_iter1 = []
    theta_iter2 = []
    theta_iter3 = []
    cost = cost_func(theta_values,X,y, L2= L2)
    cost_iter.append([0,cost])
    change_cost = 1
    i =1
    while(change_cost > converge_change):
        old_cost = cost
        theta_values = theta_values - (lr * log_gradient(theta_values, X, y,L2=L2))
        theta_iter1.append(([i,theta_values[0]]))
        theta_iter2.append(([i,theta_values[1]]))
        theta_iter3.append(([i,theta_values[2]]))
        cost = cost_func(theta_values, X, y, L2= L2)
        cost_iter.append([i, cost])
        change_cost = old_cost - cost
        i = i+ 1
    print "Total iterations:", i
    print cost_iter[i-1]
    return theta_values, np.array(cost_iter), np.array(theta_iter1),np.array(theta_iter2),np.array(theta_iter3)


def logistic_func(thetas, X):
    # bias = 1
    # x_withbias = np.hstack ((X, [[bias]] * len (X) ))
    return float(1)/(1+math.e**(-X.dot(thetas)))

def cost_func(thetas, X,y,L2= False):
    log_func_v = logistic_func(thetas,X)
    y = np.squeeze(y)
    step1 = y * np.log(log_func_v)
    step2 = (1-y) * np.log(1 - log_func_v)
    final = -step1 - step2
    if L2 == False:
        return np.mean(final)
    else:
        L2_factor = float((alpha/2)*(thetas*thetas).sum(axis=0))
        return np.mean(final) + L2_factor
    #return np.sum(final)

def log_gradient(theta, x, y, L2= False):
    first_calc = logistic_func(theta, x) - np.squeeze(y)
    # bias = 1
    #
    # x_withbias = np.hstack ((x, [[bias]] * len (x) ))
    final_calc = first_calc.T.dot(x)
    if L2 == False:
        return final_calc
    else:
        L2_factor = alpha*theta
        return final_calc + L2_factor

def pred_values(theta, X, hard=True):
    #normalize
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    bias = 1
    X = np.hstack ((X, [[bias]] * len (X) ))
    pred_prob = logistic_func(theta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    if hard:
        return pred_value
    return pred_prob
def pred_values_hinge(theta, X, hard=True):
    #normalize
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    pred_prob = logistic_func(theta, X)
    pred_value = np.where(pred_prob >= .5, 1, -1)
    if hard:
        return pred_value
    return pred_prob
#code for logit - stop

#code for hinge loss - start
def grad_desc_hinge(theta_values, X, y, L2 = False, lr = 0.001, converge_change = 0.0001):
    #standardizing X
    X = (X-np.mean(X,axis=0)) / np.std(X,axis=0)
    y[y==0] = -1
    cost_iter = []
    cost = cost_func_hinge(theta_values,X,y, L2= L2)
    cost_iter.append([0,cost])
    change_cost = 1
    i =1
    while(change_cost > converge_change):
        old_cost = cost
        theta_values = theta_values - (lr * hinge_gradient(theta_values, X, y, L2= L2))
        cost = cost_func_hinge(theta_values, X, y, L2= L2)
        cost_iter.append([i, cost])
        change_cost = old_cost - cost
        i = i+ 1
    print "Total iterations:", i
    print cost_iter[i-1]
    return theta_values, np.array(cost_iter)

def cost_func_hinge(thetas, X,y, L2 = False):
    #IMPL 1
    # y = np.squeeze(y)
    # hinge_func_v = y*(X.dot(thetas)) #y*np.dot(thetas,X)
    # zero_th = np.zeros(X.shape[0]) # 1-hinge_func_v
    # final = np.maximum(zero_th, 1-hinge_func_v)
    # #return np.mean(final)
    # return np.sum(final)

    #IMPL 2
    loss = 0
    bias = -1

    x_withbias = X
        #np.hstack ((X, [[bias]] * len (X) ))
    for (x_,y_) in zip(x_withbias,y):
        v = y_*np.dot(thetas,x_)
        loss += max(0,1-v)
    if L2 == False:
        return loss/1000
    else:
        L2_factor = float((alpha/2)*(thetas*thetas).sum(axis=0))
        return loss + L2_factor

def hinge_gradient(thetas, X, y, L2 = False):
    grad = 0
    bias = -1
    x_withbias = X
        #np.hstack ((X, [[bias]] * len (X) ))

    for (x_,y_) in zip(x_withbias,y):
        v = y_*np.dot(thetas,x_)
        grad += 0 if v > 1 else -y_*x_

    # hinge_func_v =  y*(X.dot(thetas))
    # for
    # temp_prod = -y*X
    # grad = []
    # for a,b in hinge_func_v,temp_prod:
    #     g = 0 if a>1 else b
    #     grad.append(g)
    # # for a,b,v in X,y, hinge_func_v:
    # #     g =0 if v > 1 else -b*a
    # #     grad.append(a)
    #grad = grad/np.linalg.norm(grad)
    if L2 == False:
        grad = grad/np.linalg.norm(grad)
    else:
        L2_factor = alpha*thetas
        grad = grad/np.linalg.norm(grad)+ L2_factor
        #grad = grad/np.linalg.norm(grad)
    return grad
#code for hinge loss - stop

#code for Adagrad - start
def grad_desc_adagrad(theta_values, X, y,L2 = False, lr = 0.01, converge_change = 0.001, e = 1e-8):
    #standardizing X

    X = (X-np.mean(X,axis=0)) / np.std(X,axis=0)
    bias = 1
    X = np.hstack ((X, [[bias]] * len (X) ))
    cost_iter = []
    cost = cost_func(theta_values,X,y,L2 = L2)
    cost_iter.append([0,cost])
    change_cost = 1
    j =1
    G_matrix = np.zeros((theta_values.size,theta_values.size))
    while(change_cost > converge_change):
        old_cost = cost
        g = log_gradient(theta_values, X, y,L2 = L2)
        for i in range(theta_values.size):
            G_matrix[i,i] = G_matrix[i,i]+(g[i]*g[i])
            G_denominator = G_matrix[i,i] + e
            G_denominator = math.sqrt(G_denominator)
            theta_values[i] = theta_values[i] - (lr * g[i]/G_denominator)
        cost = cost_func(theta_values, X, y,L2 = L2)
        cost_iter.append([j, cost])
        change_cost = old_cost - cost
        j+=1
    print "Total iterations:",j
    print cost_iter
    print cost_iter[j-1]
    return theta_values, np.array(cost_iter)
#code for Adagrad - stop
def hinge_grad_desc_adagrad(theta_values, X, y,L2 = False, lr = 0.001, converge_change = 0.001, e = 1e-8):
    #standardizing X

    X = (X-np.mean(X,axis=0)) / np.std(X,axis=0)
    bias = 1
    X = np.hstack ((X, [[bias]] * len (X) ))
    cost_iter = []
    cost = cost_func_hinge(theta_values,X,y,L2 = L2)
    cost_iter.append([0,cost])
    change_cost = 1
    j =1
    G_matrix = np.zeros((theta_values.size,theta_values.size))
    while(change_cost > converge_change):
        old_cost = cost
        g = hinge_gradient(theta_values, X, y,L2 = L2)
        for i in range(theta_values.size):
            G_matrix[i,i] = G_matrix[i,i]+(g[i]*g[i])
            G_denominator = G_matrix[i,i] + e
            G_denominator = math.sqrt(G_denominator)
            theta_values[i] = theta_values[i] - (lr * g[i]/G_denominator)
        cost = cost_func_hinge(theta_values, X, y,L2 = L2)
        cost_iter.append([j, cost])
        change_cost = old_cost - cost
        j+=1
    print "Total iterations:",j
    print cost_iter
    print cost_iter[j-1]
    return theta_values, np.array(cost_iter)
#code for RMSProp - start
def grad_desc_rmsprop(theta_values, X, y,L2= False, lr = 0.001, converge_change = 0.001, e = 1e-8, gamma = 0.99):
    #standardizing X
    X = (X-np.mean(X,axis=0)) / np.std(X,axis=0)
    bias = 1
    X = np.hstack ((X, [[bias]] * len (X) ))
    cost_iter = []
    cost = cost_func(theta_values,X,y,L2 = L2)
    cost_iter.append([0,cost])
    change_cost = 1
    j =1
    G_matrix = np.zeros((theta_values.size,theta_values.size))
    while(change_cost > converge_change):
        old_cost = cost
        g = log_gradient(theta_values, X, y,L2 = L2)
        for i in range(theta_values.size):
            G_matrix[i,i] = gamma*G_matrix[i,i]+(1.0-gamma)*(g[i]*g[i])
            G_denominator = G_matrix[i,i] + e
            G_denominator = math.sqrt(G_denominator)
            theta_values[i] = theta_values[i] - (lr * g[i]/G_denominator)
        cost = cost_func(theta_values, X, y,L2 = L2)
        cost_iter.append([j, cost])
        change_cost = old_cost - cost
        j =j+1
    print "Total iterations:",j
    print cost_iter[j-1]
    return theta_values, np.array(cost_iter)
#code for RMSProp - stop

#code for Adam - start
def grad_desc_adam(theta_values, X, y,L2 = False, lr = 0.01, converge_change = 0.001, e = 1e-8, b1 = 0.8, b2 = 0.1):
    #standardizing X
    X = (X-np.mean(X,axis=0)) / np.std(X,axis=0)
    bias = 1
    X = np.hstack ((X, [[bias]] * len (X) ))
    cost_iter = []
    cost = cost_func(theta_values,X,y,L2 = L2)
    cost_iter.append([0,cost])
    change_cost = 1
    j =1
    mom = np.zeros(theta_values.size)
    vel = np.zeros(theta_values.size)

    while(change_cost > converge_change):
        old_cost = cost
        g = log_gradient(theta_values, X, y,L2 = L2)
        for i in range(theta_values.size):
            mom[i] = b1 * mom[i] + (1 - b1) * g[i]
            vel[i] = b2 * vel[i] + (1 - b2) * (g[i]*g[i])
            mom_bias_corrected_val = mom[i]/(1-math.pow(b1,i+1))
            vel_bias_corrected_val = vel[i]/(1-math.pow(b2,i+1))
            denominator = math.sqrt(vel_bias_corrected_val)+e
            theta_values[i] = theta_values[i] - (lr* mom_bias_corrected_val)/denominator
        cost = cost_func(theta_values, X, y,L2 = L2)
        cost_iter.append([j, cost])
        change_cost = old_cost - cost
        j=j+1
    print "Total iterations:",j
    print cost_iter[j-1]
    return theta_values, np.array(cost_iter)
#code for Adam - stop

# main code
shape = X.shape[1]
#y_flip = np.logical_not(Y)
#betas_hinge = np.zeros(shape)
#betas = np.array([0.1, 0.1, 0.1,0.1])
betas = np.array([0.1, 0.1, 0.1])
    #np.zeros(shape+1)
# fitted_values, cost_iter = grad_desc(betas, X, y_flip)
# print(fitted_values)
# predicted_y = pred_values(fitted_values, X)
# print predicted_y

# for hinge loss
#fitted_values, cost_iter = grad_desc_hinge(betas_hinge, X, Y)
#predicted_y = pred_values_hinge(fitted_values, X)
# enthropy loss
#fitted_values, cost_iter, weight_iter1,weight_iter2,weight_iter3 = grad_desc(betas, X, Y)
# adagrad
fitted_values, cost_iter = grad_desc_adagrad(betas, X, Y)
# rmsprop
#fitted_values, cost_iter = grad_desc_rmsprop(betas, X, Y)
#adam
#fitted_values, cost_iter = grad_desc_adam(betas, X, Y)
print(fitted_values)

predicted_y = pred_values(fitted_values, X)




#print Y
#print predicted_y
count = 0
failures = 0
for i in range(Y.size):
    if Y[i]== predicted_y[i]:
        count = count + 1
    else:
        failures = failures + 1
print "Success:", count
print "failures:",failures
# print np.amax(weight_iter2[:,1])
# print np.amin(weight_iter2[:,1])

# print "w1", weight_iter1[:,1]
# print "w2", weight_iter2[:,1]
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(weight_iter1[:,1], weight_iter2[:,1],weight_iter3[:,1],linestyle = '-.', label='parametric curve')
# ax.set_xlabel('W1')
# ax.set_ylabel('W2')
# ax.set_zlabel('W3')
# ax.legend()
# plt.show()

# plt.plot(weight_iter[:,0], weight_iter[:,1])
# plt.ylabel("W1")
# plt.xlabel("Iteration")
# sns.despine()
# plt.show()


# plt.plot(cost_iter[:,0], cost_iter[:,1])
# plt.ylabel("Cost")
# plt.xlabel("Iteration")
# sns.despine()
# plt.show()


# plt.plot([0, bias_vector[0]/weight_matrix[0][1]],
#          [ bias_vector[1]/weight_matrix[0][0], 0], c = 'g', lw = 3)
X_dash1 = []
X_dash2 = []
for i in range(Y.size):
    if Y[i] == 1:
        X_dash1.append(X[i])
    else:
        X_dash2.append(X[i])
# X_dash1 = np.array(X_dash1)
# X_dash2 = np.array(X_dash2)
# x_axis = plt.scatter(X_dash1[:,0], X_dash1[:,1], c='b')
# y_axis = plt.scatter(X_dash2[:,0], X_dash2[:,1], c='r')
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend((x_axis, y_axis), ("X", "Y"))
# sns.despine()
# plt.show()

