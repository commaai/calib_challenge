import matplotlib.pyplot as plt
import numpy as np

START_X_VALUE = 0.0318534
START_Y_VALUE = 0.0358204
START_DX = 5.99e-06
START_DY = 1.36e-05

# ----------- Note: 2023-9-20 -----------
# If x or y is out of range, then increase residual weight

def g_h_filter(data, x0, dx, dt=1., max_allowed_deviation=0.1, g=3./5, h=2./3):
    state_estimate = x0
    results = []
    
    previous_state = x0
    
    for z in data:
        # prediction step

        if(abs(z - previous_state) > max_allowed_deviation):
            g = 9./10

        state_prediction = state_estimate + (dx * dt)
        dx = dx

        # update step
        residual = z - state_prediction
        dx = dx + h * (residual) / dt
        state_estimate = state_prediction + g * residual
        results.append(state_estimate)
        g = 3./5
    return np.array(results)

def test():
    # Generate actual state trends, increasing from 160 - 172 throughout 12 increments
    start_value, end_value = 160, 172
    num_increments = 12

    x_values = np.arange(num_increments)
    y_values = np.linspace(start_value, end_value, num=num_increments)
    plt.plot(x_values, y_values, label='Actual State', linestyle='-')

    # Measurement data for states
    states = np.array([158.0, 164.2, 160.3, 159.9, 162.1, 164.6, 169.6, 167.4, 166.4, 171.0, 171.2, 172.6])
    plt.plot(range(len(states)), states, label='Measured State', marker='o')

    # Apply the g-h filter to the data
    filtered_data = g_h_filter(data=states, x0=160., dx=1., g=6./10, h=2./3, dt=1.)
    plt.plot(range(len(filtered_data)), filtered_data, label='Filtered State', linestyle='--', marker='x')


    plt.xlabel('Time Step')
    plt.ylabel('State')
    plt.title('Figure 1: Comparison of Measured and Filter state against actual state')
    plt.legend()

    plt.grid(True)
    plt.show()

    print("Measured States:", states)
    print("Filtered States:", filtered_data)    

def filter_data():
    states = np.loadtxt('data/3_1695528148.466924.txt')

    x_coordinates = states[:,0]
    y_coordinates = states[:,1]

    filtered_x = g_h_filter(data=x_coordinates, x0=START_X_VALUE, dx=START_DX, dt=1.)
    filtered_y = g_h_filter(data=y_coordinates, x0=START_Y_VALUE, dx=START_DY, dt=1.)

    filtered_states = np.column_stack((filtered_x, filtered_y))
    np.savetxt('data/03_2691722614_filtered.txt', filtered_states, fmt='%.6f')



if __name__ == '__main__':

    #test()
    filter_data()
