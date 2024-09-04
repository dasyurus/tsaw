import numpy as np
import pandas as pd
from . import pycwt_fork as wavelet
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats import norm
from spectrum import pmtm, dpss
import scipy.special as sc
from scipy.signal import find_peaks

# returns maximum period within cone of influence,
# i.e. a maximum useful period
# inputs: length of time series [N],
#         wavelet [wv],
#         time increment [dt]
# output: maximum period
def get_max_T(N, wv, dt):
	coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
	coi = wv.flambda() * wv.coi() * dt * coi
	return max(coi)

# detrends and normalizes input data
# inputs: time (independent variable) [t],
#         time series (dependent variable) [x]
# output: detrended and normalized time series 	
def detrend(t, x):
	p = np.polyfit(t - t[0], x, 1)
	dat_notrend = x - np.polyval(p, t - t[0])
	std = dat_notrend.std()

	return dat_notrend / std

# calculates a bunch of time-related variables
# inputs:  time (independent variable) [t],
#          maximum period of interest [T_end]
#          ** T_end can be user-specified
# outputs: time increment [dt],
#          starting (Nyquist) period [T_start],
#          ending (maximum) period [T_end],
#          array of frequencies for wavelet transforms [freqs_in]
def	get_timing(t, T_end):
	dt = t[1] - t[0]
	T_start = 2 / dt # Nyquist period
	wv = wavelet.Morlet(6) # use Morlet wavelet with w0 = 6 
	if T_end < 0:
		T_end = get_max_T(len(t), wavelet.Morlet(6), dt)
	freqs_in = 1/(wv.flambda() * np.geomspace(T_start, T_end, num=len(t))) 
	
	return dt, T_start, T_end, freqs_in

# calculates continuous wavelet transform 
# inputs: time (independent variable) [t],
#         time series (dependent variable) [x],
#         ** it is strongly recommended that x be detrended.
#             see detrend() function.
#         maximum period of interest [T_end] (optional)
#         ** default is calculated from the wavelet transform
#            cone of influence.
#         significance level [significance_level] (optional),
#         ** default is 0.90 (90% significance) 
# outputs: frequencies of interest [freqs],
#          spectral power [power],
#          significance bounds [sig95], 
#          cone of influence [coi], 
#          continuous wavelet transform coefficients [coefs], 
#          AR1 autocorrelation coefficient [alpha]
def cwt(t, x, T_end = -99, significance_level = 0.9):	
	dt, T_start, T_end, freqs_in = get_timing(t, T_end)
	
	# s0/dj/J setup is included as an example
	# this is a very common way to do a cwt in the literature!
	
	#s0 = 2 * dt
	#dj = 1/12
	#J = 49
	#mother = wavelet.Morlet(6)

	# Do CWT
	#coefs, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(, dt, dj, s0, J, mother)
	
	# note that cwt from pycwt is used -- recall that this cwt() function is a wrapper
	coefs, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(x, dt, wavelet=wavelet.Morlet(6), freqs=freqs_in)
	power = (np.abs(coefs)) ** 2

	# significance test
	alpha, _, _ = wavelet.ar1(x)
	signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
											 significance_level=significance_level,
											 wavelet=wavelet.Morlet(6))
										
	sig95 = np.ones([1, len(x)])  *signif[:, None]
	sig95 = power / sig95
	
	return freqs, power, sig95, coi, coefs, alpha

# calculates cross wavelet transform 
# inputs: time (independent variable) [t],
#         time series (dependent variable) [x],
#         ** it is strongly recommended that x be detrended.
#             see detrend() function.
#         ** this is the comparison time series
#         continuous wavelet transform coefficients [coefs], 
#         AR1 autocorrelation coefficient [alpha]	
#         maximum period of interest [T_end] (optional),
#         ** default is calculated from the wavelet transform
#            cone of influence. 
#         ** IMPORTANT: T_end selected MUST be consistent between
#                       both time series. Because the lengths of both
#                       time series in a XWT must be equal, the
#                       default T_end will be equal too.
#         significance level [significance_level] (optional)
#         ** default is 0.90 (90% significance)               
# outputs: frequencies of interest [xfreqs],
#          spectral power [xpower],
#          significance bounds [xsig95], 
#          cone of influence [xcoi],
#          cross wavelet transform coefficients [xcoefs],
#          phase of xcoefs, [a_xcoefs]
# IMPORTANT: the continuous wavelet transform (cwt) for the 
#            main time series must be calculated first! 
#            You will need the coefs and alpha variables from
#            the cwt() function to feed into the xwt() function.
def xwt(t, x, coefs, alpha, T_end = -99, significance_level = 0.9 ):
	# define periods of interest
	dt, T_start, T_end, freqs_in = get_timing(t, T_end)

	coefs_, scales_, freqs_, coi_, fft_, fftfreqs_ = wavelet.cwt(x, dt, wavelet=wavelet.Morlet(6), freqs=freqs_in)
	xpower = (np.abs(coefs)) ** 2

	xcoefs = coefs * coefs_.conj()
	xfreqs = freqs_
	xcoi = coi_
	# significance test
	xalpha, _, _ = wavelet.ar1(x)

	Pk1 = wavelet.ar1_spectrum(xfreqs * dt, alpha)
	Pk2 = wavelet.ar1_spectrum(xfreqs * dt, xalpha)
	dof = wavelet.Morlet().dofmin
	significance_level = 0.8646 * significance_level/.95 # consistent results for Grinsted 95%
	PPF = chi2.ppf(significance_level, dof)

	xsignif = ((Pk1 * Pk2) ** 0.5 * PPF / dof)
	xsig95 = np.ones([1, len(x)]) * xsignif[:, None]
	xsig95 = np.abs(xcoefs) / xsig95 # pycwt uses power, grinsted used coefs. Units coefs here would consistent to power in CWT?

	a_xcoefs = np.angle(xcoefs)
	
	return xfreqs, xpower, xsig95, xcoi, xcoefs, a_xcoefs	

# helper functions that returns base-2 period ticks for a graph
# input: array of periods of interest [period]
# output: list of base-2 periods which will be labelled
def get_base2_ticks(period):
	min_T  = min(period)
	max_T = max(period)
	# get second-ish-next power of 2 from min 
	start_T  = int(np.ceil(np.log(min_T)/np.log(2) + 0.5))
	end_T = int(np.floor(np.log(max_T)/np.log(2))) 
	return [2**n for n in range(start_T,end_T+1)]

# plots wavelet transforms (both CWT and XWT)
# inputs: time (independent variable) t,
#         frequencies of interest (independent variable) [freqs],
#         wavelet transform coefficients (dependent variable) [coefs],
#         significance bounds [sig95],
#         cone of influence [coi],
#         filename to which to save figure [fn] (optional),
#         ** default is False, in which case, nothing is saved to file
#         whether or not a xwt is plotted [xwt] (optional),
#         ** default is False, i.e. plot cwt.
#         whether or not phase information is plotted [arrows] (optional),
#         ** default is False, do not plot arrows. 
#         ** NOTE: this option is meant for plotting xwts
#         whether or not a colorbar is included with the figure [colorbar] (optional)
#         ** default is False, no colorbar is plotted.
def plot_wt(t, freqs, coefs, sig95, coi, fn=False, xwt=False, arrows=False, colorbar=False):
	fig, axs = plt.subplots(1,1)
	period = 1 /freqs
	# main plot
	if xwt==False:
		pcm = axs.pcolormesh(t, period, (np.abs(coefs)) ** 2, cmap=plt.colormaps["Spectral_r"])
	else:
		pcm = axs.pcolormesh(t, period, np.abs(coefs), cmap=plt.colormaps["Spectral_r"])
	axs.set_yscale("log",base=2)
	axs.set_xlabel("Year")
	axs.set_ylabel("Period (yr)")

	extent = [t.min(), t.max(), 2, max(period)]
	axs.contour(t, period, sig95, [-99, 1], colors='k', linewidths=2, extent=extent)

	min_T = min(period)
	max_T = max(period)
	min_t = min(t)
	max_t = max(t)
	#print(min_T)
	coi_ = coi
	#dt = t[1]-t[0]
	delta_t = t[1]-t[0]
	delta_T = period[1]/period[0]
	coi[coi_<min(period)] = min(period)/delta_T #- 0.03
	
	# pad around the pcolormesh to remove unwanted "border"
	axs.fill(np.concatenate([t, t[-1:] + delta_t, t[-1:] + delta_t,
							 t[:1] - delta_t, t[:1] - delta_t]),
			 np.concatenate([coi, [min_T/(delta_T)], period[-1:]*(delta_T),
							 period[-1:]*(delta_T), [min_T/(delta_T)]]),
			 'k', alpha=0.2)

	if arrows==True:
		a_xcoefs = np.angle(coefs)
		# plot arrows
		x_quiv = t
		y_quiv = period
		a_quiv = a_xcoefs
		R = int(np.round(len(t)/25))
		if R < 1:
			R = 1
		XX, YY = np.meshgrid(x_quiv[::R],y_quiv[::R])
		G=2.
		Q = axs.quiver(XX, YY, G*np.cos(a_quiv[::R, ::R]), G* -np.sin(a_quiv[::R, ::R]),units='xy', scale=100/len(t), headwidth=4.)

					   
	axs.get_yaxis().set_major_formatter(
		ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
 
	axs.set_yticks(get_base2_ticks(period))
	
	# remove overshoot from the fill
	axs.set_xlim(min_t, max_t)  
	axs.set_ylim(min_T, max_T)
	
	axs.invert_yaxis()
	
	plt.tight_layout()
	if colorbar==True:
		fig.colorbar(pcm, ax=axs)
	if fn!=False:
		plt.savefig(fn, dpi=200, bbox_inches='tight', pad_inches=0.5)
	else:
		plt.show()
			
	
#---- periodogram functions
# calculates lag-1 autocorrelation coefficient
# input: time series [datax]
# output: autocorrelation coefficient [rho]
def rhoAR1(datax):
    nrho=len(datax)           
    rho=0
    sommesup=0                                                                                       
    sommeinf=0
    moy=np.sum(datax)/nrho                                   
    datam=datax-moy                                       
    for i in range(1, nrho):
        j=i-1;                             
        sommesup=sommesup+(datam[i]*datam[j])
        sommeinf=sommeinf+((datam[j])**2)
                                                                         
    rho=sommesup/sommeinf
    
    return rho

# wrapper for inverse to the regularized lower incomplete gamma function
# replicates chi-square inverse cumulative distribution function
def chi2invPMTK(p, v):
    return 2*sc.gammaincinv(v/2,p) # note inputs are flipped relative to identical matlab fxn

# calculates confidence level (red noise, pct confidence levels)
# inputs: number of tapers used by MTM [nw],
#         area of data power spectrum [Ax],
#         length of padded input data [npr],
#         time increment [dt],
#         autocorrelation coefficient [rho],
#         array of frequencies of interest [fr],
#         array of confidence levels [conf]
# outputs: array of spectral power limits for each frequency, 
#          for each confidence level [tabtchi],
#          array of spectral power limits for each frequency, 
#          for red noise level [theored]
def conflevel(nw,Ax,npr,dt,rho,fr, conf):
    # Estimate of the Chi-square confidence levels for the mean spectrum of the 
    # nsim red noise signals.
    
    tabchi=np.zeros((npr, len(conf)))
    
    # The degrees of freedom for a mtm analysis is equal to 2*number of tapers
    # (Tompson, 1982).
    nw2=2*(2*nw-1);
    
    facchi = np.zeros(len(conf))
    for idx, lvl in enumerate(conf):
         facchi[idx] = chi2invPMTK(lvl,nw2)/nw2

    # estimate of the theoretical red noise spectrum
    fnyq=1/(2*dt);
    theored = (1-rho**2)/(1-(2*rho*np.cos(np.pi*fr/fnyq))+rho**2)

    # normalisation of the spectrum
    theoredun=theored[0]
    theored[0]=0
    Art=np.sum(theored)/npr
    theored[0]=theoredun
    theored=theored*(Ax/Art)


    # Estimate of the Chi-square confidence levels theoretical red noise
    # spectrum.
    
    tabtchi=np.zeros((npr, len(conf)))
    
    for idx in range(0,len(conf)):
        tabtchi[:, idx] = theored.flatten()*facchi[idx]

    return tabtchi,theored

# pad array with zeroees to next power of 2
# input: original time series [x]
# output: padded time series    
def pad_arr(x):
    N_ = len(x)
    N = 2** np.ceil(np.log(N_)/np.log(2))
    ds = N.astype(int) - N_
    return np.concatenate((x, np.zeros(ds, dtype=x.dtype)))

# calculate Thomson’s multitaper power spectral density (PSD) estimate
# functions as a wrapper for spectrum.pmtm, but also replicates
# functionality of Matlab pmtm() function.
# inputs: padded time series [x],
#        DPSS windows [tapers],
#        oncentration ratios for the windows [eigen]
# outputs: Thomson’s multitaper power spectral density (PSD) estimate [Sk],
#          frequencies of interest [xf]  
# NOTE: search documenation for spectrum.dpss and spectrum.pmtm for
#        more details on this operation.
def pmtm2(x, tapers, eigen):
    N = len(x)
    Sk_complex, weights, eigenvalues=pmtm(x, e=eigen, v=tapers, show=False) #from spectrum

    Sk = abs(Sk_complex)**2
    Sk = np.mean(Sk * np.transpose(weights), axis=0) # scale might be funny re. matlab but frequency-specific behaviour ok?
    xf = np.arange(0, N, 1) * np.pi /N
    return Sk[0:N], xf

# Calculate periodogram of a time series
# inputs: time series [x],
#         time increment [dt] (optional),
#         ** default is 1
#         number of tapers used by MTM [nw] (optional),
#         ** default is 2
#         confidence level [conf] (optional)
#         ** default are 0.85, 0.90, 0.95, and 0.99
# outputs: frequencies of interest [fd]
#          Thomson’s multitaper power spectral density (PSD) estimate [po],
#          array of spectral power limits for each frequency, 
#          for each confidence level [tabtchi],
#          array of spectral power limits for each frequency, 
#          for red noise level [theored]
def pdgram(x, dt = 1, nw = 2, conf=[0.85,0.90,0.95,0.99]):
    timex = np.arange(0,len(x),1)
    
    #AR1 lag coefficient
    rho = rhoAR1(x)
    
    # normalize data
    nd=len(x)
    dataxm=np.sum(x)/nd
    datan=x-dataxm
    
    pad_datan = pad_arr(datan)
    # spectral analysis of the data
    [tapers, eigen] = dpss(len(pad_datan), nw) # may need to do only once if we properly pad? doublecheck.  # from spectrum
    po, w = pmtm2(pad_datan, tapers,eigen)

    fd=w/(2*np.pi*dt)
    npo=len(po)       
    
    # calculation of the area of the data power spectrum
    Ax=(np.sum(po))/npo
    
    npr = len(pad_arr(x))
        
    tabtchi,theored=conflevel(nw,Ax,npr,dt,rho,fd, conf)

    return fd, po, tabtchi, theored

# Plots periodogram with red noise and confidence level estimates
# inputs: frequencies of interest [fd]
#         Thomson’s multitaper power spectral density (PSD) estimate [po],
#         array of spectral power limits for each frequency, 
#         for each confidence level [tabtchi],
#         array of spectral power limits for each frequency, 
#         for red noise level [theored],
#         relative to which confidence level 
#         to show peaks [show_peaks] (optional),
#         ** defaults to False, no peaks
#         ** shows peaks if a value is chosen from conf
#         list of confidence levels [conf] (optional),
#         maximum period of interest [T_end] (optional),
#         ** by default, maximum period is chosen with same logic as for
#            the wavelet transforms
#         filename to which to save figure [fn] (optional),
#         ** default is False, in which case, nothing is saved to file
#         whether or not legend is displayed [legend] (optional),
#         ** default is True, show legend.
def plot_pdgram(fd, po ,tabtchi, theored, t, show_peaks=False, conf=None, T_end =-99, fn=False, legend=True):
	dt, T_start, T_end, _ = get_timing(t, T_end)
	fd_ = fd
	T = 1/fd_[1:]

	po_ = po[1:]

	peaks, _ = find_peaks(po_)
	tabtchi_ = tabtchi[1:,:]

	fig, axs = plt.subplots(1,1)

	axs.plot(T,po_.T)

	# plot red noise
	axs.plot(T,theored[1:], label='Red noise', color='red')
	# plot significance levels
	
	# Get the default color cycle
	default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
	
	for idx in range(0,len(tabtchi_[0,:])):
		color_idx = idx+1
		if idx>=2: # color idx 3 is also red
			color_idx = idx+2
		axs.plot(T,tabtchi_[:,idx], label="{:.2f}".format(conf[idx]), color=default_colors[color_idx])
		



	if show_peaks != False and show_peaks in conf:
		for peak in peaks:
			if po_[peak] >= tabtchi_[peak,conf.index(show_peaks)]:
				plt.annotate(f'{T[peak]:.1f}', (T[peak], po_[peak]), textcoords="offset points", xytext=(0,10), ha='center')

	axs.set_xscale('log', base=2)

	axs.set_xticks(get_base2_ticks(T))
	axs.get_xaxis().set_major_formatter(
		ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
	plt.xlim(T_start, T_end)
	#axs.legend(loc='lower right')
	#axs.legend(facecolor='white', framealpha=1, edgecolor='black', loc='best', fontsize=10)
	# Get the handles and labels
	if legend==True:
		handles, labels = plt.gca().get_legend_handles_labels()
		axs.legend(handles[::-1], labels[::-1], facecolor='white', framealpha=1, edgecolor='black', loc='lower right', fontsize=10)
	
	if fn!=False:
		plt.savefig(fn, dpi=200, bbox_inches='tight', pad_inches=0.5)
	else:
		plt.show()

