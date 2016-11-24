from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

data = datasets.load_iris()
X = data.data[:100,:2]
Y = data.target[:100]
X_full = data.data[:100,:]

#setosa = plt.scatter(X[:50,0],X[:50,1], c='b')
#versicolor = plt.scatter(X[50:,0],X[50:,1],c='r')
#sns.despine()
#plt.show()

#code for logit - start
def grad_desc(theta_values, X, y, lr = 0.001, converge_change = 0.001):
    #standardizing X
    X = (X-np.mean(X,axis=0)) / np.std(X,axis=0)
    cost_iter = []
    cost = cost_func(theta_values,X,y)
    cost_iter.append([0,cost])
    change_cost = 1
    i =1
    while(change_cost > converge_change):
        old_cost = cost
        theta_values = theta_values - (lr * log_gradient(theta_values, X, y))
        cost = cost_func(theta_values, X, y)
        cost_iter.append([i, cost])
        change_cost = old_cost - cost
        i+=1
    return theta_values, np.array(cost_iter)


def logistic_func(thetas, X):
    return float(1)/(1+math.e**(-X.dot(thetas)))

def cost_func(thetas, X,y):
    log_func_v = logistic_func(thetas,X)
    y = np.squeeze(y)
    step1 = y * np.log(log_func_v)
    step2 = (1-y) * np.log(1 - log_func_v)
    final = -step1 - step2
    return np.mean(final)

def log_gradient(theta, x, y):
    first_calc = logistic_func(theta, x) - np.squeeze(y)
    final_calc = first_calc.T.dot(x)
    return final_calc

def pred_values(theta, X, hard=True):
    #normalize
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    pred_prob = logistic_func(theta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    if hard:
        return pred_value
    return pred_prob
#code for logit - stop

#code for hinge loss - start
def grad_desc_hinge(theta_values, X, y, lr = 0.001, converge_change = 0.001):
    #standardizing X
    X = (X-np.mean(X,axis=0)) / np.std(X,axis=0)
    cost_iter = []
    cost = cost_func_hinge(theta_values,X,y)
    cost_iter.append([0,cost])
    change_cost = 1
    i =1
    while(change_cost > converge_change):
        old_cost = cost
        theta_values = theta_values - (lr * hinge_gradient(theta_values, X, y))
        theta_values = theta_values/i
        cost = cost_func_hinge(theta_values, X, y)
        cost_iter.append([i, cost])
        change_cost = old_cost - cost
        i+=1
    return theta_values, np.array(cost_iter)

def cost_func_hinge(thetas, X,y):
    y = np.squeeze(y)
    hinge_func_v = y*(X.dot(thetas)) #y*np.dot(thetas,X)
    zero_th = np.zeros(X.shape[0]) # 1-hinge_func_v
    final = np.maximum(zero_th, 1-hinge_func_v)
    return np.mean(final)

def hinge_gradient(thetas, X, y):
    grad = 0
    for (x_,y_) in zip(X,y):
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
    grad = grad/np.linalg.norm(grad)
    return grad

#code for hinge loss - stop
# main code
shape = X.shape[1]
y_flip = np.logical_not(Y)
betas = np.zeros(shape)
# fitted_values, cost_iter = grad_desc(betas, X, y_flip)
# print(fitted_values)
# predicted_y = pred_values(fitted_values, X)
# print predicted_y

# for hinge loss
#fitted_values, cost_iter = grad_desc_hinge(betas, X, y_flip)
# enthropy loss
fitted_values, cost_iter = grad_desc(betas, X, y_flip)
print(fitted_values)
predicted_y = pred_values(fitted_values, X)
print predicted_y