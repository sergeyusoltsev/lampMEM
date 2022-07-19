#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 09:31:24 2022

@author: Sergey Usoltsev
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from scipy import interpolate
from os.path import exists

filenamelist = []
fig, axs = plt.subplots(2, 2, figsize=(18, 9), gridspec_kw={'height_ratios': [5, 1]})
inset = fig.add_axes([0.758, 0.6, 0.125, 0.25])

decaylist = ['1_605_40mkl.dat', 
            '2_605_80mkl.dat', 
            '3_605_160mkl.dat',
            '4_605_240mkl.dat', 
            '5_605_320mkl.dat']

fileprefix = '605_volhstock_upsample3_100lts_'

names = [r'$0.2\cdot10^{-5}$',
          r'$0.4\cdot10^{-5}$',
          r'$0.8\cdot10^{-5}$',
          r'$0.16\cdot10^{-4}$',
          r'$0.32\cdot10^{-4}$']

sqrtmachineepsilon = (7./3 - 4./3 -1) ** (1/2)
irf = '000_605_volstock_irf.dat'

colorindex = np.linspace(0.2,0.8,len(decaylist))
plotslegend = []
MEMslegend = []

def plotdecay(x, y, col, zord, name = '_nolegend_'):
    global plotslegend
    plotslegend.append(str(name))
    kwargs = dict(color = col, linestyle = "solid",
                  linewidth = 2, zorder = zord)
    axs[0][0].plot(x, y, **kwargs)

def acf(x):
    length = len(x)
    return np.array([0]+[np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, length)])


def MEM(x, y, xr, yr, col, zord, name = '_nolegend_', timestart = 0,
        timelimit = 100, numiter = 500, numlts = 100, upsample = 3, 
        mode = 'lin', simpleinterp = False, plotonly=True):
    
    irfmax = list(yr).index(max(yr))
    decaymax = list(y).index(max(y))
    binwidth = max(xr)/len(xr)
    
    # watch carefully for background level determination,
    # as this is somewhat ad-hoc solution
    prepulse = np.mean(y[:irfmax-40])
    
    valtime = np.floor(timelimit / binwidth).astype(int)
    threshold = max(yr) * 0.005 # 0.5% treshold to include in fit
    for n, i in enumerate(y):
        if (i < threshold or i < prepulse) and n > decaymax and n < valtime:
            valtime = n # fit up to ... (look from the end)
            break
    
    # tail trim
    x = np.array(x[0:valtime])
    y = np.array(y[0:valtime])
    xr = np.array(xr[0:valtime])
    yr = np.array(yr[0:valtime])
    
    # iterpolation
    if upsample != 1:
        print('----\nUpsampling %i bins...' % int(valtime - decaymax))
        yn = interpolate.interp1d(x, y, fill_value='extrapolate')
        yrn = interpolate.interp1d(xr, yr, fill_value='extrapolate')
        x = np.arange(0, x[-1], x[-1]/(len(x) * upsample))
        xr = x
        y = yn(x)
        if simpleinterp == False: # apply poisson noise to interpolated data?
            print('With Poisson noise added to linearly interpolated data')
            y = np.random.poisson([y if y > 0 else 0 for y in y])
        yr = yrn(x)
        if simpleinterp == False:
            yr = np.random.poisson([y if y > 0 else 0 for y in yr])
        
        # new range - new maxima and binwidth
        binwidth = max(xr)/len(xr)
        irfmax = list(yr).index(max(yr))
        decaymax = list(y).index(max(y))
    
    # we don't cut the dataset, we are fitting to its' part, otherwise
    # the FFTC border effects will affect the result severely
    decaystart = decaymax + np.floor(timestart/binwidth).astype(int) # shift start by this number
    
    print('Have %i bins to work with' % (len(x) - decaymax))
    print('----\nbkg level %f' % prepulse)
    
    global tau
    if mode == 'lin':
        tau = np.linspace(0.01, 10, num=numlts)
    elif mode == 'log':
        tau = np.logspace(-2, 1, num=numlts)
        
    if not exists(fileprefix + 'mem_results.txt'):
        with open(fileprefix + 'mem_results.txt', 'w') as f:
            for t in tau:
                f.write('%.4f\t' % t)
            f.write('\n')
        
    ai = [1/(sum(tau)) for i in range(0, len(tau))] # initial amplitudes 
    ai.append(1) # IRF_SCALING initial value
    ai.append(0) # IRF_SHIFT initial value

    # Fast Fourier transform convolution nojit 10.21 jit 10.65
    def convolve(a, b):
        aa = np.fft.fft(a)
        bb = np.fft.fft(b)
        conv = np.real(np.fft.ifft(aa*bb))
        return conv

    # Residuals
    def resid (fc, fe): 
        resid = [(point - fc[n])/point for n, point in enumerate(fe)]
        return resid

    # Chi-squared 
    def chisq (fc, fe):  
        return (fc - fe) ** 2 / fe
    
    # # working variant before optimization
    # def multiexponent(ai, xr=xr, tau=tau, bkg=prepulse):
    #     exponent = [np.e ** (-(xr - ai[-1])/t) for t in tau]
    #     convolvent = np.zeros(len(xr)).astype('complex128')
    #     for num, exp in enumerate(exponent):
    #         convolvent += convolve(yr, exp) * ai[num] 
    #     convolvent = bkg + convolvent * ai[-2] # R = A + B . I
    #     return convolvent[decaystart:]
    
    # # working variant after optimization (20% faster)
    def multiexponent(ai, xr=xr, tau=tau, bkg=prepulse):
        exponent = [np.e ** (-(xr - ai[-1])/t) for t in tau]
        amplituded = np.multiply(convolve(yr, exponent).T, ai[:-2])
        convolvent = bkg + np.sum(amplituded.T, axis = 0) * ai[-2]
        return convolvent[decaystart:]

    # Chi-squared (Obligatory constraint)
    def chisqconstr (ai):
        fc = multiexponent(ai)
        point = y[decaystart:].astype('complex128')
        chisq = np.sum((fc - point) ** 2 / point)
        return 1/((2 if upsample != 1 else 1) * len(point)) * chisq - 1

    # Sum of lt amplitudes (Optional constraint) 
    def sumaconstr (ai):
        fc = np.sum(ai[:-2])
        return 1 - fc # 1 - fc >= 0 -> 1 >= fc for ineq constraint
   
    # Shannon-Jaynes entropy (Objective !!!)  
    def shanjaymini (ai):
        pi = ai[:-2] #[a/sum(ai[:-2]) for a in ai[:-2]]
        return np.sum(pi * np.log(pi))
    
    def readmem(finame, reqnum):
    # 605_volh2o_interp2_unscaled_mem_results.txt
        with open(finame, 'r') as f:
            for num, line in enumerate(f):
                if num == 0:
                    tau = np.array(line.strip().split('\t')).astype(float)
                elif num == reqnum:
                    ai = np.array(line.strip().split('\t')).astype(float)
        return tau, ai
    
    # Values' bounds:
        
    bnd = (1e-10, 1)
    bnds = [bnd for i in ai[:-2]]
    bnds.append((1e-10, 1))
    bnds.append((-1, max([(decaymax - irfmax) * binwidth, 1e-10])))
    bnds = tuple(bnds) 
    
    # Constraints:
    
    cons = ({'type' : 'eq', 'fun' : chisqconstr},{'type' : 'ineq', 'fun' : sumaconstr})
    slsqpopts = {'maxiter':numiter, 'disp':True, 'ftol':1e-5, 'eps': sqrtmachineepsilon}# * a for a in ai]}
    
    # Minimization and plotting the result or importing it:
    if plotonly == False:
        res = minimize(shanjaymini, x0=ai, method='SLSQP', bounds=bnds, jac='cs',
                       constraints=cons, options=slsqpopts)
        ai = res.x
    else:
        print('reading ' + fileprefix + 'mem_results.txt decay ' + str(zord + 1))
        tau, ai = readmem(fileprefix + 'mem_results.txt', int(zord + 1))
        
    T = multiexponent(ai)
    chisquared = 1/((2 if upsample != 1 else 1) * len(y)) * np.sum(chisq(T, y[decaystart:]))
    ent = - shanjaymini([1e-10 if a <= 0 else a for a in ai[:-2]])
       
    print('----')
    print('Chi-squared %f' % chisquared)
    print('Shannon-Jaynes %f' % ent)
    print('Sum of amplitudes %f' % np.sum(ai[:-2]))
    print('IRF Shift %.3f (max-irf %.3f)' % (ai[-1], (decaymax - irfmax) * binwidth))
    print('Scaling factor %.3f' % ai[-2])
   
    pi = [a/sum(ai) for a in ai[:-2]]

    plotslegend.append(str(name))
    kwargs = dict(color = col, linestyle = "solid",
                  linewidth = 2, zorder = zord)
    
    if mode == 'log':
        axs[0][1].set_xscale('log')
    axs[0][0].plot(x[decaystart:], T, color = 'white', linestyle='--', linewidth = 1, zorder = zord + 0.5)
    axs[0][1].plot(tau, pi, **kwargs)
    axs[1][0].plot(x[decaystart:], resid(T, y[decaystart:]), **kwargs)
    axs[1][1].plot(x[decaystart:], acf(resid(T, y[decaystart:])), **kwargs)
    axs[0][0].axhline(prepulse, color = col, linestyle='--', linewidth = 1,
                      label ='_nolegend_')
    axs[0][0].axvline(x[decaystart], color = 'lightgray', linestyle='--', linewidth = 1,
                      label ='_nolegend_')
    insetais.append(ai[:-2] ** (1/3))
    if plotonly == False:
        with open(fileprefix + 'mem_results.txt', 'a') as f:
            for a in ai:
                f.write('%f\t' % a)
            f.write('\n')
    
def main():

    global insetais
    insetais = []
    
    axs[0][0].set_yscale('log')
    axs[0][0].set_ylim([1, 20000])
    axs[0][0].set_xlim([0, 60])
    axs[1][0].set_xlim([0, 60])
    axs[1][0].set_ylim([-3, 3])
    axs[1][1].set_ylim([-1, 1])

    
    xr, yr = np.genfromtxt(irf, unpack=True)
    # yr = [1 if val <= 1 else val for val in yr]
    
    plotdecay(xr, yr, 'lightgray', -999, 'IRF')
    
    for i, j in enumerate(decaylist):
        x, y = np.genfromtxt(j, unpack=True)
        
        y = [1 if val <= 1 else val for val in y]
        y = [yi * 10 if max(y) < 10000 else yi for yi in y]
        
        plotdecay(x, y, plt.cm.viridis(colorindex[i]), i, '%s' % names[i])
        MEM(x, y, xr, yr, plt.cm.viridis(colorindex[i]), i, '_nolegend_')
      
    ais = np.array(insetais)
    x, y = np.meshgrid(tau, range(1, len(decaylist)+1))
    levels = np.linspace(ais.min(), ais.max(), 50)
    inset.contourf(x, y, ais, cmap=plt.cm.viridis, antialiased=False, levels = levels)
    inset.locator_params(axis='y', nbins=len(names))
    inset.set_yticklabels(names)
    inset.set_title('Top view')


    axs[0][0].legend(plotslegend, loc = 'upper right')
    
    axs[0][0].set_ylabel(r'Intensity, a.u.')
    axs[0][0].set_xlabel(r't, ns.')
    axs[1][0].set_ylabel(r'Resid., a.u.')
    axs[0][1].set_xlabel(r'$\tau$, ns.')
    axs[1][1].set_ylabel(r'$ACF$, a.u.')
    axs[0][1].set_ylabel(r'$P(\tau)$, a.u.') #/p_{max}
    
    axs[0][0].grid(color = 'k', alpha = 0.6, linestyle = ':')
    axs[1][0].grid(color = 'k', alpha = 0.6, linestyle = ':')
    axs[1][1].grid(color = 'k', alpha = 0.6, linestyle = ':')
    axs[0][1].grid(color = 'k', alpha = 0.6, linestyle = ':')
        
    fig.savefig(fileprefix + 'mem.png')

if __name__ == '__main__':
    starttime = time.time()
    main()
    print("----\ntotal run time\n%s seconds\n" % (time.time() - starttime))

