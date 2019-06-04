import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import math
import scipy
from scipy import interpolate
from scipy.interpolate import interp1d

def snr(freq, hf, freq_psd, psd, df, f1, f2):

    # print(f1,f2,len(freq), freq[0], freq[1], freq[2], freq[3], freq[4])
    # get the indices in freq that includes [f1 f2]
    ind=np.where(freq>f1)
    ind1=ind[0][0]
    ind=np.where(freq<f2)
    ind2=ind[0][-1]
    # print(ind1, ind2, freq[ind1],freq[ind2])
    
    freq1=freq[ind1:ind2]
    hf1=hf[ind1:ind2]


    # dont assume that freq and freq_psd is identical
    ind=np.where(freq_psd>f1)
    ind1=ind[0][0]
    ind=np.where(freq_psd<f2)
    ind2=ind[0][-1]
    # print(ind1, ind2, freq_psd[ind1], freq_psd[ind2])
    
    freq_psd1=freq_psd[ind1:ind2]
    psd1=psd[ind1:ind2]

    if len(psd1) != len(hf1):
        #print("hf and psd vectors have different lengths. PLease check Nyquist freqs.")
        #print("PSD Nyquist freq:", freq_psd[-1])
        #print("WVF Nyquist freq:", freq[-1])
        index=np.minimum(len(freq_psd1),len(freq1))
        # We will assume this is because fN are different for wvf and psd and
        # stop the integral whenever min(fN,fN) is reached.

        p=np.zeros(index)
        for i in range(0,index):
            p[i]=np.power(hf1[i],2)/psd1[i]
        #print(ind1, ind2, len(psd1) , len(hf1))
    else:
        # compute SNR
        p=np.power(hf1,2)/psd1

    snr=np.sqrt(4*sum(p)*df)
    return snr

def fourier(hplus, hcross, time, dt, deltaf):
    
    fs=1./dt

    if fs.is_integer() == False or fs < len(hplus):
        if fs < len(hplus):
            #print ('resampling the waveform to have deltaf=1 Hz')
            fs=len(hplus)
            dtnew=1./fs
            tnew=np.arange(time[0],time[-1],dtnew);
            #            print(dt, dtnew,fs, time[1], time[-1], len(time), len(tnew))
        else:
            #print ('resampling the waveform')
            dtnew=1./np.floor(1./dt)
            tnew=np.arange(time[0],time[-1],dtnew);
            
        hp1 = interpolate.interp1d(time,hplus)
        hc1 = interpolate.interp1d(time,hcross)

        hp_res=hp1(tnew)
        hc_res=hc1(tnew)
    else:
        #print ('nothing to do')
        hp_res=hplus
        hc_res=hcross
        tnew=time
        
    Nt = len(hp_res)
    w=np.hanning(Nt)
    hp=hp_res*w
    hc=hc_res*w
    fs=1./dtnew
    N=np.int(fs/deltaf)
    #print("wvf sampling:", fs, "Hz. wvf samples nb:", Nt, "segment sample nb:", N)
    
    zeropad=N-Nt
    if zeropad<0:
        print('NEGATIVE padding!!!')
        
    X=np.pad(hp,(0,zeropad),'constant')
    Y=np.pad(hc,(0,zeropad),'constant')

    freqs=np.round(np.fft.fftfreq(N, dtnew),3)       # round precision to 3 digits to avoid
                                                     # rounding errors later
    m=int(math.ceil(N/2.))

    freqs=freqs[0:m]

    hpf=np.fft.rfft(X)     # FFT normalized by sqrt(Nt)
    hcf=np.fft.rfft(Y)     # FFT normalized by sqrt(Nt)
    spec=np.sqrt(np.power(np.abs(hpf),2)+np.power(np.abs(hcf),2))

    # One sided spectrum is |FT|/(fs/2)
    spec=spec/(fs/2.)

    if N % 2 == 0:
        spec=spec[0:m]
    #print("freqs sample nb:", len(freqs), "Fourier sample nb:", len(spec), 'Wvf Nyquist:', np.int(fs/2.))
    return freqs, spec


def aLIGO_PSD_new(f,type):
# aLIGO sensitivity curve: fit the data point from https://dcc.ligo.org/LIGO-T1800044/public
# Type=1 --> one-sided PSD.
# Type=2 --> two-sided PSD. 

    S1 = 5.0e-26
    S2 = 1.0e-40
    S3 = 1.4e-46
    S4 = 2.7e-51
    fcut = 10.
    cutoff=1e-38
    output=np.zeros(len(f))

    # to avoid issue with f=0
    if f[0]==0:
        f[0]=f[1]

    if type == 1:
        for i in range(len(f)):
            x=np.abs(f[i])
            output[i] =  S1/np.power(x, 20.) + S2/np.power(x, 4.05) + S3/np.power(x,.5) + S4*np.power (x/fcut, 2.0)
            if output[i]>cutoff:
                output[i]=cutoff
    else:
        for i in range(int(len(f)/2)+1):
            x=np.abs(f[i])
            output[i] = S1/np.power(x, 20.) + S2/np.power(x, 4.05) + S3/np.power(x,.5) + S4*np.power (x/fcut, 2.0)
            if output[i]>cutoff:
                output[i]=cutoff

            if i>0:
                output[len(f)-i]=output[i]

        output=output/2.          # Two sided PSD
                
    return output

# Main: SNR compute

wvf = np.loadtxt("../waveforms/KURODA_SFHX_H_resampled.dat", dtype='f', delimiter=' ')
t=wvf[:,0]
hp=wvf[:,1]
hc=wvf[:,2]

dt=t[1]-t[0]
Fs=round(1./dt)
deltaf=.1

n=int(Fs/deltaf)

print(dt,Fs,n)

freq=Fs*np.fft.fftfreq(n)         # two-sided frequency vector
ff=freq[range(0,int(n/2))]        # one-sided frequency vector
psd=aLIGO_PSD_new(ff,1)        # one-sided PSD

freqWVF,specWVF = fourier(hp, hc, t, dt, deltaf)

hcWVF=2.*freqWVF*specWVF

f1=2
f2=np.floor(Fs/2.)-1

rho=snr(freqWVF, specWVF, ff, psd, deltaf, f1, f2)
print('aLIGO SNR [',f1,'-',f2,']Hz =', rho)

fig1 = plt.figure(1)
plt.loglog(ff,np.sqrt(psd),'g',label='aLIGO')
plt.plot(freqWVF,hcWVF,'r',label='KURODA 10kpc')
plt.xlabel('Frequency [Hz]')
plt.ylabel('ASD / hchar')
plt.show()


#hpf=np.fft.fft(hp)
#hcf=np.fft.fft(hc)

#hpfr=hpf[range(int(n/2))]
#hcfr=hcf[range(int(n/2))]

#integrand=np.power(np.absolute(hpfr+hcfr),2)/psd

#print(len(integrand))

#snr = 4*np.trapz(integrand, ff)
#print(np.sqrt(snr))
