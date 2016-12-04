from sklearn import datasets
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

data = datasets.load_iris()
# X = data.data[:100,:2]
# Y = data.target[:100]
# #X_full = data.data[:100,:]
X, Y = make_blobs(n_samples=5000, centers=2, n_features=2,cluster_std=10.0 ,center_box=(-10.0,10.0) ,shuffle = True, random_state= 10)

#alpha = 100 # GD
alpha = 2
#code for logit - start
def grad_desc(theta_values, X, y, L2= False, lr = 0.001, converge_change = 0.001):
    #standardizing X

    X = (X-np.mean(X,axis=0)) / np.std(X,axis=0)
    bias = 1
    X = np.hstack ((X, [[bias]] * len (X) ))
    cost_iter = []
    cost = cost_func(theta_values,X,y, L2= L2)
    cost_iter.append([0,cost])
    change_cost = 1
    i =1
    while(change_cost > converge_change):
        old_cost = cost
        theta_values = theta_values - (lr * log_gradient(theta_values, X, y,L2=L2))
        cost = cost_func(theta_values, X, y, L2= L2)
        cost_iter.append([i, cost])
        change_cost = old_cost - cost
        i = i+ 1
    print "Total iterations:", i
    print cost_iter[i-1]
    return theta_values, np.array(cost_iter)


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
def grad_desc_hinge(theta_values, X, y, L2 = False, lr = 0.001, converge_change = 0.001):
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
def grad_desc_adam(theta_values, X, y,L2 = False, lr = 0.01, converge_change = 0.001, e = 1e-8, b1 = 0.99, b2 = 0.1):
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
betas_hinge = np.zeros(shape)
betas = np.array([0.1, 0.2, 0.1])
    #np.zeros(shape+1)
# fitted_values, cost_iter = grad_desc(betas, X, y_flip)
# print(fitted_values)
# predicted_y = pred_values(fitted_values, X)
# print predicted_y

# for hinge loss
#fitted_values, cost_iter = grad_desc_hinge(betas_hinge, X, Y)
#predicted_y = pred_values_hinge(fitted_values, X)
# enthropy loss
#fitted_values, cost_iter = grad_desc(betas, X, Y)
# adagrad
#fitted_values, cost_iter = grad_desc_adagrad(betas, X, Y, L2 = True)
# rmsprop
#fitted_values, cost_iter = grad_desc_rmsprop(betas, X, Y,  L2 = True)
#adam
fitted_values, cost_iter = grad_desc_adam(betas, X, Y,  L2 = True)
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

plt.plot(cost_iter[:,0], cost_iter[:,1])
plt.ylabel("Cost")
plt.xlabel("Iteration")
sns.despine()
plt.show()


# plt.plot([0, bias_vector[0]/weight_matrix[0][1]],
#          [ bias_vector[1]/weight_matrix[0][0], 0], c = 'g', lw = 3)
# setosa = plt.scatter(X[:50,0], X[:50,1], c='b')
# versicolor = plt.scatter(X[50:,0], X[50:,1], c='r')
# plt.xlabel("Sepal Length")
# plt.ylabel("Sepal Width")
# plt.legend((setosa, versicolor), ("Setosa", "Versicolor"))
# sns.despine()
# plt.show()

