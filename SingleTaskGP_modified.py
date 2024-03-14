##https://gpcam.readthedocs.io/en/latest/examples/basic_test.html

from gpcam import AutonomousExperimenterGP
import numpy as np
import numpy as np
import matplotlib.pyplot as plt




#this function defines the break condition of the autonomous experimenter
def break_condition(my_ae):
    if my_ae.data.dataset[0]['x_data']>=3:
        return True  # Stop the loop after some iterations
    else:
        return False  # Continue the loop

def plot_graphs(my_ae,acquisition_function):
# Extract x_data and y_data from the list of dictionaries
    x_data = [data['x_data'] for data in my_ae.data.dataset]
    y_data = [data['y_data'] for data in my_ae.data.dataset]
    print(x_data,"====\n",y_data)

# Plotting
    plt.figure(figsize=(5, 5))
    for i in range(len(x_data)):
        plt.scatter(x_data[i], y_data[i], color='blue')  # Plotting x_data[0] vs y_data
#plt.scatter(x_data[i][1], y_data[i], color='red')   # Plotting x_data[1] vs y_data

    plt.xlabel('x_data')
    plt.ylabel('y_data')
    plt.title('Plot of x_data vs y_data '+acquisition_function)
    plt.legend(['[x_data,y_data]'])
    plt.grid(True)
    plt.show()
    
def instrument(data):
    for entry in data:
        print("I want to know the y_data at: ", entry["x_data"])
        entry["y_data"] = entry["x_data"]+entry["x_data"]
#print(entry["x_data"][0])
        print("I received ",entry["y_data"])
        print("")
    return data

##set up your parameter space
## this is for setting the data
## all the parameters bounds we give here will apply to the x data
parameters = np.array([[int(3),int(10)]])

##set up some hyperparameters, if you have no idea, set them to 1 and make the training bounds large
init_hyperparameters = np.array([100,100,100])
hyperparameter_bounds =  np.array([[90,100],[90,100.0],[90,100]])


##variable "expected_improvement","target probability":
acquisition_function = ["ucb","lcb","maximum","minimum", "variance",
        "relative information entropy","relative information entropy set",
        "probability of improvement", "gradient","total correlation"]

acquisition_function = ["ucb"]



##let's initialize the autonomous experimenter ...
for acq in acquisition_function:
    try:
        my_ae = AutonomousExperimenterGP(parameters, init_hyperparameters,
                                         hyperparameter_bounds,instrument_function = instrument,
                                         acquisition_function = acq,
                                         init_dataset_size=2, info=False)
    except:
        print(acq,"is not valid")
        continue
#...train...
    my_ae.train()

#...and run. That's it. You successfully executed an autonomous experiment.
    my_ae.go(N = 20,checkpoint_filename ="checkpoint_filename.txt",break_condition_callable = break_condition)
    plot_graphs(my_ae,acq)



