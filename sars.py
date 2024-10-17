import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


# database reading
file_path = 'C:\\Users\\administration\\programming\\SARS_cases_by_district.xlsx'
pf = pd.read_excel(io=file_path, sheet_name='Sheet1')


# number of districts
n = 18
""" HongKong District List & population
1   Central & Western   261884
2   Eastern             616199
3   Southern            290240
4   Islands             86667
5   Kwai Tsing          477092
6   Kowloon City        381352
7   Kwun Tong           562427
8   North               298657
9   Sai Kung            327689
10  Sham Shui Po        353550
11  Sha Tin             628634
12  Tuen Mun            488831
13  Tai Po              310879
14  Tsuen Wan           275527
15  Wan Chai            167146
16  Wong Tai Sin        444630
17  Yuen Long           449070
18  Yau Tsim Mong       287915
"""


# parameter calculation (working)
beta = 0.062
beta_array = np.array([beta, 0.370 * beta, 0.059 * beta])
mu = 0.2
c = {'cw': 1, 'cc': 0.57, 'cm': 0.02}

def delta(i,j):
    if i == j:
        return 1
    else:
        return 0


# M matrix calculation
class M:
    """ 
    Mc : a short-distance region matrix 
         In this matrix, the element is 1 for adjacent regions that are not the same, and 0 otherwise.
    Mm : a long-distance region matrix
         In this matrix, the element is 1 for non-identical adjacent regions, and 1 otherwise. 
    """
    def __init__(self, n):
        self.Mc = np.zeros((n,n))
        self.Mm = np.zeros((n,n))

        self.Mc[0, [2, 14, 17]] = 1
        self.Mc[1, [2, 6, 14]] = 1
        self.Mc[2, [3, 14]] = 1
        self.Mc[3, [4, 11, 13]] = 1
        self.Mc[4, [9, 10, 13]] = 1
        self.Mc[5, [6, 9, 10, 14, 15, 17]] = 1
        self.Mc[6, [8, 15]] = 1
        self.Mc[7, [12, 16]] = 1
        self.Mc[8, [10, 15]] = 1
        self.Mc[9, [10, 17]] = 1
        self.Mc[10, [12, 13, 15]] = 1
        self.Mc[11, [13, 16]] = 1
        self.Mc[12, [13, 16]] = 1
        self.Mc[13, 16] = 1
        self.Mc[14, 17] = 1

        for i in range (n):
            for j in range (n):
                if self.Mc[i,j] == 1:
                    self.Mc[j,i] = 1

        self.Mm = 1 - self.Mc
        for i in range (n):
            self.Mm[i,i] = 0


# N matrix calculation
N = np.array(pf['Population'].tolist())
""" N : population data by region.
"""


# K matrix calculation
K_matrix = np.zeros((n,n))
M_matrix = M(n)

for j in range (n):
    for i in range (n):
        K_matrix[j,i] = (c['cw'] * delta(i,j) + 
                         c['cc'] * M_matrix.Mc[i,j] + 
                         c['cm'] * M_matrix.Mm[i,j]) / N[j]


# S matrix calculation
S_matrix = np.zeros((n,n))

for i in range (n):
    for j in range (n):
        S_matrix[i,j] = N[i] * K_matrix[j,i]

eigenvalues, eigenvectors = np.linalg.eig(S_matrix)
S = np.max(eigenvalues.real)


# r calculation
# T calculation
def T(i):
    r_matrix = np.array([0.6, 1, 0.2, 0.2]) # r_matrix = [r_I=r_A, r_Y, r_HR, r_HD]
    
    T_Y = np.array([4.85, 3.83, 3.67]) # T_Y value list
    T_matrix = np.array([1.01, T_Y[i], 23.5, 35.9]) # T_matrix = [T_I, T_Y, T_HR, T_HD]
    """ 
    T_matrix : 

    T_I + T_L = 6.37 Days
    Epidemiological determinants of spread of causal agent of severe acute respiratory syndrome in Hong Kong
    : Infection to onset of symptom average time is 6.37 Days

    Transmission Dynamics of the Etiological Agent of SARS in HongKong:Impact of Public Health Interventions
    Infection to onset of symptom average time is equal to T_L + T_I
    since paper used T_L value as 5.36 Days, T_I = 1.01 Days
    """

    T_value = r_matrix * T_matrix
    T_value[2] = (1-mu) * T_value[2]
    T_value[3] = mu * T_value[3]
    return T_value


# Rt value calculation
Rt_XSS = []
for i in range (3):
    T_value = T(i)
    Rt0_XSS = beta * np.sum(T_value) * S
    Rt_XSS.append((beta_array[i] / beta) * Rt0_XSS)

print(np.sum(T_value))
print(S)
print(Rt_XSS)


# ploting
dates = np.array(['26-Feb', '5-Mar', '12-Mar', '19-Mar', '26-Mar', '2-Apr', '9-Apr', '16-Apr', '23-Apr', '30-Apr'])

# Figure 2B
"""
daily_case_red = np.array(pf.loc[6].to_list()[2:])
daily_case_blue = np.array(pf.loc[10].to_list()[2:]) + daily_case_red
daily_case_green = np.array(pf.loc[12].to_list()[2:]) + daily_case_blue
all_case = np.array(pf.loc[0].to_list()[2:])
for i in range(n-1):
    all_case += np.array(pf.loc[i+1].to_list()[2:])
plt.plot(dates, daily_case_red, 'r', label=r'7 area')
plt.plot(dates, daily_case_blue, 'b', label=r'11 area')
plt.plot(dates, daily_case_green, 'g', label=r'13 area')
plt.plot(dates, all_case, 'black', label=r'all case')
plt.fill_between(dates, daily_case_green, color='g')
plt.fill_between(dates, daily_case_blue, color='b')
plt.fill_between(dates, daily_case_red, color='r')
plt.xlabel('Date')
plt.ylabel('Daily Incidence')
plt.legend()
plt.show()
"""

# Figure 2C

plot_Rt = [Rt_XSS[0]]*4 + [Rt_XSS[1]]*3 + [Rt_XSS[2]]*3
plt.plot(dates, plot_Rt, 'black', drawstyle='steps-post')
plt.xlabel('Date')
plt.ylabel('Rt_XSS')
plt.show()


