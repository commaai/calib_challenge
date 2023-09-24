import numpy as np
import sys
import matplotlib.pyplot as plt
import math

#TEST_DIR = './data/03_1691722614_filtered.txt'
TEST_DIR = './data/3_1695183450.706041.txt'
GT_DIR = 'labeled/'

#Finding Mean Squared Error
def get_mse(gt, test):
  test = np.nan_to_num(test) #handles missing NaN (not a number) values with 0
  return np.mean(np.nanmean((gt - test)**2, axis=0))

def accuracy(gt, test):
  zero_mses = []
  mses = []

  zero_mses.append(get_mse(gt, np.zeros_like(gt)))
  mses.append(get_mse(gt, test))
  percent_err_vs_all_zeros = 100*np.mean(mses)/np.mean(zero_mses)
  print(f'YOUR ERROR SCORE IS {percent_err_vs_all_zeros:.2f}% (lower is better)') 


def gather_data():
  actual_state_x = []
  actual_state_y = []

  prediction_state_x = []
  prediction_state_y = []

  avg_x = 0
  prev_x = 0
  avg_y = 0
  prev_y = 0

  dx_total_x = 0
  dx_total_y = 0

  control_x = 0.0318534
  control_y = 0.0358204
  max_x = 0
  max_y = 0

  adjusted_length = len(gt)

  for index in range(len(test)):

    if(max_x < abs(gt[index][0]-control_x)):
      max_x = abs(gt[index][0]-control_x)
    
    if(max_y < abs(gt[index][1]-control_y)):
      max_y = abs(gt[index][1]-control_y)

    if(index == 0):
      prev_x = gt[index][0]
      prev_y = gt[index][1]
    else:
      prev_x = gt[index-1][0]
      prev_y = gt[index-1][1]

    if(math.isnan(gt[index][1]) or math.isnan(test[index][1])):
      adjusted_length-=1
      gt[index][1] = 0
      gt[index][0] = 0

    actual_state_x.append(gt[index][0])
    actual_state_y.append(gt[index][1])

    prediction_state_x.append(test[index][0])
    prediction_state_y.append(test[index][1])

    avg_x += gt[index][0]
    avg_y += gt[index][1]

    dx_total_x += gt[index][0] - prev_x
    dx_total_y += gt[index][1] - prev_y

  avg_x /= adjusted_length
  avg_y /= adjusted_length
  dx_total_x /= adjusted_length
  dx_total_y /= adjusted_length

  print("Average X: ", avg_x)
  print("Average Y: ", avg_y)
  print("dx_total_x: ", abs(dx_total_x))
  print("dx_total_y: ", abs(dx_total_y))
  print("max_x: ", max_x)
  print("max_y: ", max_y)

  return actual_state_x, actual_state_y, prediction_state_x, prediction_state_y

def plot_comparison(actual_state_x, actual_state_y, prediction_state_x, prediction_state_y, display_x):

  plotIndex = int(len(actual_state_x)/1)
  print(plotIndex)
  print(len(actual_state_x))

  if(display_x):
    plt.plot(range(plotIndex), actual_state_x[:plotIndex], label='Actual State X', linestyle='-')
    plt.plot(range(plotIndex), prediction_state_x[:plotIndex], label='Predicted State X', linestyle='--')
  else:
    plt.plot(range(plotIndex), actual_state_y[:plotIndex], label='Actual State Y', linestyle='-')
    plt.plot(range(plotIndex), prediction_state_y[:plotIndex], label='Predicted State Y', linestyle='--')

  plt.xlabel('Time Step')
  plt.ylabel('State')
  plt.title('Figure 1: Comparison of Measured and Filter state against actual state')
  plt.legend()

  plt.grid(True)
  plt.show()

if __name__ == '__main__':

  gt = []
  test = []

  label_index = '3'
  print(GT_DIR + label_index + '.txt')
  gt = np.loadtxt(GT_DIR + label_index + '.txt')
  test = np.loadtxt(TEST_DIR)

  #accuracy(gt, test)

  actual_state_x, actual_state_y, prediction_state_x, prediction_state_y = gather_data()
  plot_comparison(actual_state_x, actual_state_y, prediction_state_x, prediction_state_y, True)

