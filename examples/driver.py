import numpy as np 
import matplotlib.pyplot as plt
import src.muktiregression as multiregression

def main(): 
    data = np.loadtxt("data/Question_2_DATA_rho_vp.txt") 
    p = data[:,0] 
    vp = data[:,1] 


    plt.figure(figsize=(8, 5))
    plt.scatter(p, vp, marker='o')
    plt.title("Velocity vs Density")
    plt.xlabel("Density")
    plt.ylabel("Velocity")
    plt.show() 
    plt.savefig("figures/Question_2_velocity_vs_density.png")
    # the data is increasing exponentially, so we will take the ln of the velocity
    # to makie it linear, visuallly the data matches this trend. Equation (3) also 
    # suggests this is the case. 
    # so we will take the ln of the velocity to make it linear.

    # this means our parameters are: 
    # a1 = ln(V0) 
    # a2 = k

    vp = np.log(vp) 
    z = np.vstack((np.ones_like(vp)),vp).T

    aCoeff, em, R2 = multiregression.multiregression(vp, z) 
    print("Coefficients: ", aCoeff)
    print("Residuals: ", em)
    print("R^2: ", R2)


if __name__ == "__main__": 
    main()