import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import math
from scipy.optimize import curve_fit

def func(x,a,b,d):
    xx=x/a
    xx2=xx*xx
    y=np.log(1e-49)+np.log(xx**(b)+ d*(1-xx2+0.5*xx2*xx2)/(1.+0.5*xx2))
    return y

def aLIGO_PSD(f):
    cutoff = -109.35 + math.log(2e10)
    logpsd=np.zeros(len(f))
    if f[0]==0:
        f[0]=f[1]

    for i in range(len(f)):
        x = f[i]/215.
        x2 = x*x
        logpsd[i] = np.log(1e-49) + np.log(x**(-4.14) -5./x2 + 111.*(1-x2+0.5*x2*x2)/(1.+0.5*x2))
        
        if logpsd[i]>cutoff:
            logpsd[i]=cutoff
    output=np.exp(logpsd)
    return output

def aLIGO_PSD2(f):
    fS1 = 5.e-26
    fS2 = 1.e-40
    fS3 = 1.4e-46
    fS4 = 2.7e-51
    FCUT = 10.

#    output = fS1/np.power(f, 56.) + fS2/np.power(f, 4.56) + fS3 + fS4*np.power (f/FCUT, 2.0) ;
    output =  fS1/np.power(f, 20.) + fS2/np.power(f, 4.05) + fS3/np.power(f,.5) + fS4*np.power (f/FCUT, 2.0) ;
    return output

input = np.loadtxt("../PSD/aLIGODesign.txt", dtype='f', delimiter='  ')
xdata=input[:,0]
ydata=input[:,1]

#popt, pcov = curve_fit(func, xdata, ydata)

Fs=20000
n=1*Fs
ff=Fs/n*np.arange(0,10000)
psd=aLIGO_PSD(ff)
psd2=aLIGO_PSD2(ff)

plt.figure(1)
plt.loglog(xdata,ydata,'r',label='aLIGO')
plt.loglog(ff,np.sqrt(psd),'b',label='param')
plt.loglog(ff,np.sqrt(psd2),'g',label='param2')
plt.grid(True)
plt.show()

