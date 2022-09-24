import numpy as np
import matplotlib.pyplot as plot

#Tonya Shulkey (018145619)
#CECS 456 Machine Learning
#Linear Regression with gradient descent
#3/11/22

list1 = []
list2 = []

f = open('ex1data1.txt')
data = f.readlines()
f.close()
for i in data:
    line = i.strip().split(',')     #Strip will get rid of the \n in the y list
    list1.append(line[0])
    list2.append(line[1])

x = np.array(list1)
y = np.array(list2)

#Previously these were seen as string and would make a straight line we want floats
x = x.astype(float)
y = y.astype(float)

#Set the title, x, and y labels
plot.title("Training Data")
plot.xlabel("Population of City in 10,000s")
plot.ylabel("Profit in $10,000s")

# 'rx' is red x mark
plot.plot(x, y, 'rx')
# plot.show()



x = [np.ones(len(x)), x]        #add another collumn of 1's in x


theta = np.array([0.0, 0.0])
iterations = 1000
alpha = 0.01



# The cost function formula
def cost(x, y, theta):
    m = len(x[0])                   #number of training examples
    sumation = 0.0                  #The values are floats so sumation should be float

    for i in range(m):
        # h(x) = theta0 + theta1x

        hypothesis = theta[0] * x[0][i] + theta[1] * x[1][i]
        sumation += (hypothesis - y[i]) ** 2
    cost = sumation / (2.0 * m)

    return cost


# print("This is the cost: ",cost(x, y, theta))


# Batch gradient descent
def batch_gradient_descent(x, y, theta, alpha, iterations):
    m = len(x[0])                   #number of training examples

    J = []
    J.append(cost(x, y, theta))

    cost_it = 0

    theta0 = []
    theta1 = []
    theta0.append(theta[0])
    theta1.append(theta[1])

    it = 0          #how many iterations occured out of the total iterations
    while it < iterations:

        sumation1 = 0.0
        sumation2 = 0.0

        for j in range(m):
            hypothesis = theta[0] * x[0][j] + theta[1] * x[1][j]
            sumation1 += (hypothesis - y[j])
            sumation2 += (hypothesis - y[j]) * x[1][j]

        # simulatanious update
        theta[0] = theta[0] - (alpha * (sumation1 / m))
        theta[1] = theta[1] - (alpha * (sumation2 / m))

        theta0.append(theta[0])
        theta1.append(theta[1])

        J.append(cost(x, y, theta))
        cost_it += 1

        it += 1

    return theta, J, theta0, theta1



theta, J, theta0, theta1 = batch_gradient_descent(x, y, theta, alpha, iterations)

print("This is the cost values: \n", J)

print("Theta found by gradient descent: ", theta)




#-------Plot the new line-----------
plot.plot(x[1], theta[0] + theta[1] * x[1])


#--------Plot J(theta) in terms of theta1---------
# plot.plot(theta1, J)

#------Plot the cost-------

# plot.plot(J)




#-----------for the next two graphs-----------------
# theta0_vals = np.linspace(-10,10,100)
# theta1_vals = np.linspace(-1,4,100)
# Jval = np.zeros((len(theta0_vals),len(theta1_vals)))
# for i, t0 in enumerate(theta0_vals):
#   for j, t1 in enumerate(theta1_vals):
#        Jval[i,j] = cost(x, y, [t0, t1])
# T0, T1 = np.meshgrid(theta0_vals,theta1_vals)
#
# fig = plot.figure()

#----------Contour Graph-------------
# plot.contour(T0,T1,Jval, np.logspace(-2, 3, 20))
# plot.plot(theta[0], theta[1], 'rx', ms = 10, lw = 3)    #plot point of found theta0 and theta1


#------------3d Graph----------------

# ax = fig.add_subplot(projection = '3d')
# ax.plot_surface(T0, T1, Jval)
# plot.plot(theta[0], theta[1], 'rx', ms = 10, lw = 3)    #plot point of found theta0 and theta1

plot.show()