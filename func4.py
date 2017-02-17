import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
from scipy import signal
from textwrap import wrap
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'

# Reads a number from a line, then removes that number and returns the line

def NumExtr(x): 
    
    output = ""
    
    while x[0] != ",": #Reads the file up to a comma
        output = output + x[0]
        x = x[1:]
    
    x = x[1:]
    
    return (x,output)

def NumExtr(x): 
    
    output = ""
    
    while x[0] != ",": #Reads the file up to a comma
        output = output + x[0]
        x = x[1:]
    
    x = x[1:]
    
    return (x,output)

# Extracts the zaccel value, which is complicated by the fact that there's no comma at the end.

def NumExtr2(x):
    
    output = ""
    j = 0
    
    while j < 2: #Reads the file up to the end of zaccel. Number is determined by the length of zaccel
        output = output + x[0]
        x = x[1:]
        j = j + 1
    
    x = x[1:]
    
    return (x,output)

# Reads a line for the new file format, read from the internet instead of from a text file.
# Major change is that it now uses the NumExtr2 function to extract the zaccel value.
# Also returns the file, but with the line that has been read deleted. 

def LineRead(x, time, yomega, xomega, zomega, yaccel, xaccel, zaccel, acalib, gcalib): 
    
    x, time1 = NumExtr(x) # Extracts the time value
    time1 = int(time1)
    time = np.concatenate((time, [time1]), axis = 0)

    x, yomega1 = NumExtr(x) # Extracts the yomega value
    yomega1 = float(yomega1)
    yomega = np.concatenate((yomega, [yomega1]), axis = 0)

    x, xomega1 = NumExtr(x) # Extracts the xomega value
    xomega1 = float(xomega1)
    xomega = np.concatenate((xomega, [xomega1]), axis = 0)
    
    x, zomega1 = NumExtr(x) # Extracts the zomega value
    zomega1 = float(zomega1)
    zomega = np.concatenate((zomega, [zomega1]), axis = 0)
    
    x, yaccel1 = NumExtr(x) # Extracts the yaccel value 
    yaccel1 = float(yaccel1)
    yaccel = np.concatenate((yaccel, [yaccel1]), axis = 0)

    x, xaccel1 = NumExtr(x) # Extracts the xaccel value
    xaccel1 = float(xaccel1)
    xaccel = np.concatenate((xaccel, [xaccel1]), axis = 0)

    x, zaccel1 = NumExtr(x) # Extracts the zaccel value
    zaccel1 = float(zaccel1)
    zaccel = np.concatenate((zaccel, [zaccel1]), axis = 0)
    
    x, acalib1 = NumExtr(x) # Extracts the acceleration calibration value
    acalib1 = float(acalib1)
    acalib = np.concatenate((acalib, [acalib1]), axis = 0)

    x, gcalib1 = NumExtr2(x) # Extracts the acceleration calibration value
    gcalib1 = float(gcalib1)
    gcalib = np.concatenate((gcalib, [gcalib1]), axis = 0)

    return (x, time, yomega, xomega, zomega, yaccel, xaccel, zaccel, acalib, gcalib)

# Reads the entire internet file. Significant changes are that instead of using the number of lines, the file is treated as one long string. Also the previously altered functions are used 

def FileRead(file): 
    
    j = 0
    output = ""
    time = []
    yomega = []
    xomega = []
    zomega = []
    xaccel = []
    yaccel = []
    zaccel = []
    acalib = []
    gcalib = []
    
    while len(file) > 0:

        file, time, yomega, xomega, zomega, yaccel, xaccel, zaccel, acalib, gcalib = LineRead(file, time, yomega, xomega, zomega, yaccel, xaccel, zaccel, acalib, gcalib)
        
    time = time/[1000]
    time = time - [time[0]]
    #omega = omega - [omega[0]]
    #phi = phi * (np.pi/180)
    
    return time, yomega, xomega, zomega, yaccel, xaccel, zaccel, acalib, gcalib

# This function removes the linear drift from the signal. It assumes that the car is at rest when the data ends

def DriftDel(x, time): 
    
    drift = x[-1] - x[0]
    
    driftstep = float(drift)/float((len(time)))
    
    j = 1
    
    while time[j] < time[-1]:
        
        x[j] = x[j] - j*driftstep
        
        j = j + 1
    
    x[-1] = x[-1] - j*driftstep
    
    return x

# The following function is a variant on the previous drift removal function. It seeks to give a more accurate estimate of 
# drift by taking the average of the first few values. This assumes that the car is at rest both initially and at the end.

def DriftDel2(x, time): 
    
    drift = np.mean([x[-10:-1]]) - np.mean([x[0:10]])
    
    driftstep = drift/(len(time))
    
    j = 1
    
    while time[j] < time[-1]:
        
        x[j] = x[j] - j*driftstep
        
        j = j + 1
    
    x[-1] = x[-1] - j*driftstep
    
    return x


# This is another variant on the drift removal function. This is a test, aimed specifically at xaccel

def DriftDel3(x, time): 
    
    drift = x[-1] - (-0.75)
    
    driftstep = float(drift)/float((len(time)))
    
    
    j = 1
    
    while time[j] < time[-1]:
        
        x[j] = x[j] - j*driftstep
        
        j = j + 1
    
    x[-1] = x[-1] - j*driftstep
    
    return x

# This is a Butterworth filter, which filters noise from a signal. Cutoff refers to the cutoff frequency (all frequencies
# above this will be removed) and fs is the sampling frequency

def Butter2(x, cutoff, fs, order=3):
    nyq = 0.5 * fs
    normalcutoff = cutoff / nyq
    
    b, a = signal.butter(order, normalcutoff)
    y = signal.filtfilt(b, a, x, padlen=100)
    
    return y

# This removes the offset. 

def OffsetRem(x):
    
    x = x - [np.mean(x)]
    
    return x

# This is a variant of the offset removal, that assumes the very first value is the offset

def OffsetRem2(x):
    
    x = x - [x[0]]
    
    return x

# Calculates the offset from calibration data

def Offset(x): 
    
    offset = np.mean(x[0:15])
    
    return offset

#This removes the delay from the data

def DelayRem(x,time): 
    j = 0
    
    while time[j+10] < time[-1]:
        r = np.arange(j,j+10,1)
        
        if abs(np.mean(x[r])) < 0.05:
            x = np.delete(x,j)
            time = np.delete(time,j)
            j = j - 1
        
        j = j + 1
    
    return x, time

# Filters calibration data

def Filter(x,time): 
    
    x_filt = Butter2(x, 0.0625, 40)
    x_filt = OffsetRem(x_filt) #This step is not a good idea for actual data. It'll work only with calibration data.
    x_filt = np.concatenate((x_filt, [0]))
    x_filt = Butter2(x_filt, 0.0625, 40)
    x_filt = DriftDel(x_filt,time)
    
    return x_filt

# A more general filter

def Filter2(x,time,offset,cutoff): 
    
    x = Butter2(x, cutoff, 40)
    x = x - [offset] #This step is not a good idea for actual data. It'll work only with calibration data.
    #x = np.concatenate((x, [0]))
    #x = np.concatenate((x, [x[-1]])) # A temporary move. 
    x = Butter2(x, cutoff, 40)
    x = DriftDel2(x,time)
    
    return x

# Trapezium Integral function

def TrapInt(time,accel): 
    j = 1
    integ = [0]
    
    while time[j] < time[-1]:
        integ = np.concatenate((integ, [np.trapz([accel[j-1], accel[j]], [time[j-1], time[j]])]), axis=0)
        j = j + 1
    
    integ = np.concatenate((integ, [integ[-1]]), axis=0)
    
    j = 1
    
    while time[j] < time[-1]:
        integ[j] = integ[j-1] + integ[j] 
        j = j + 1
    
    integ[-1] = integ[len(integ)-2] + integ[-1] 
    
    return integ

# Carries out a trapezium integration twice

def DoubleTrapInt(time,accel): 
    vel = TrapInt(time,accel)
    disp = TrapInt(time, vel)
    
    return disp

# Euler Integral function

def EulerInt(time, accel, timeinterval): 
    j = 1
    velocity = [0]

    while time[j] < time[-1]: #Euler integration
        velocity = np.concatenate((velocity, [velocity[j-1] + accel[j-1]*timeinterval]), axis=0)
        j = j + 1
    
    velocity = np.concatenate((velocity, [velocity[-1]]), axis=0)
    
    return velocity

# Carries out an Euler integration twice

def DoubleEulerInt(time, accel, timeinterval):
    
    vel = EulerInt(time,accel, timeinterval)
    disp = EulerInt(time,vel, timeinterval)
    
    return disp

# Converts from intrinsic coordinates into cartesian ones

def IntToCartX(s, phi,time): 
    x = [0] #This is the initial value of x
    j = 1
    
    while time[j] < time[-1]:
        x = np.concatenate((x, [(s[j]-s[j-1])*np.cos(phi[j])]), axis=0)
        j = j + 1 
    
    j = 1
    
    while time[j] < time[-1]:
        x[j] = x[j-1] + x[j]
        j = j + 1
    
    x = np.concatenate((x, [x[-1]]), axis = 0)
    
    return x

# Converts from intrinsic coordinates into cartesian 

def IntToCartY(s, phi,time): 
    y = [0] #This is the initial value of y
    j = 1
    
    while time[j] < time[-1]:
        y = np.concatenate((y, [(s[j]-s[j-1])*np.sin(phi[j])]), axis=0)
        j = j + 1
    
    j = 1
    
    while time[j] < time[-1]:
        y[j] = y[j-1] + y[j]
        j = j + 1
    
    y = np.concatenate((y, [y[-1]]), axis = 0)
    
    return y

# Derives omega from the y accel data

def OmegaDeriv(yaccel, xvel, time):
    
    j = 1
    omega = [0]
    
    while time[j] < time[-1]:
        
        omega = np.concatenate((omega, [yaccel[j]/xvel[j]]))
        j = j + 1
    
    return omega