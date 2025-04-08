import numpy as np 
import matplotlib.pyplot as plt
import muktiregression as multiregression

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
    # to makie it linear, visuallly the data matches this trend 

    vp = np.log(vp) 
    v0 = np.zeros_like(vp)



if __name__ == "__main__": 
    main()