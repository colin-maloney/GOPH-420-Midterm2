import numpy as np 
import matplotlib.pyplot as plt
import GOPH_420_Midterm2.muktiregression as multiregression

def main(): 
    data = np.loadtxt("data/Question_2_DATA_rho_vp.txt") 
    p = data[:,0] 
    vp = data[:,1] 


    plt.figure(figsize=(8, 5))
    plt.scatter(p, vp, marker='o')
    plt.title("Velocity vs Density")
    plt.xlabel("Density")
    plt.ylabel("Velocity")
    plt.savefig("figures/Plot_1_Original_data.png")
    # the data is increasing exponentially, so we will take the ln of the velocity
    # to makie it linear, visuallly the data matches this trend. Equation (3) also 
    # suggests this is the case. 
    # so we will take the ln of the velocity to make it linear.

    # this means our parameters are: 
    # a1 = ln(V0) 
    # a2 = k

    y = np.log(vp) 
    z = np.vstack((np.ones_like(p),p)).T

    aCoeff, em, R2 = multiregression.multiregression(y, z) 
    print("Coefficients: ", aCoeff)
    print("Residuals: ", em)
    print("R^2: ", R2)

    model_vp = z @ aCoeff 
    

    plt.figure(figsize=(8, 5)) 
    plt.scatter(p, y, marker='o', label='data') 
    plt.plot(p, model_vp, label='fitted line' ,color='red')
    plt.title("Linerized Vp vs Density with linearized data with regression line") 
    plt.xlabel("Density") 
    plt.ylabel("Velocity") 
    plt.legend()   
    plt.text(0.75, 0.05, f"ln(V0): {aCoeff[0]:.3f}, k: {aCoeff[1]:.3f} ", fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
    plt.savefig("figures/Plot3_regression_line_in_lnVp_space.png") 

    V0 = np.exp(aCoeff[0]) 
    k = aCoeff[1] 

    aCoefflinear = np.array([V0, k])
    y_model = V0 * np.exp(k * p) 
    sorted_indicies = np.argsort(p) 
    p_sorted = p[sorted_indicies] 
    y_model_sorted = y_model[sorted_indicies]
    
    

    plt.figure(figsize=(8, 5)) 
    plt.scatter(p, vp, marker='o', label='data') 
    plt.plot(p_sorted, y_model_sorted, label='fitted line' ,color='red')
    plt.title("Original plot with regression line") 
    plt.xlabel("Density") 
    plt.ylabel("Velocity") 
    plt.legend() 
    plt.text(0.75, 0.05, f"V0: {aCoefflinear[0]:.3f}, k: {aCoefflinear[1]:.3f} ", fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
    plt.savefig("figures/Plot4_Regression_line_in_original_vp_sapce.png") 
    
    # recover true model parameter values 
    V0 = np.exp(aCoeff[0]) 
    k = aCoeff[1] 
    print(f"V0 [m/s]: {V0} ")  
    print(f"k [cm^3/g]: {k} ") 


if __name__ == "__main__": 
    main()