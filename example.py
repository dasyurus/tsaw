import pandas as pd
import tsaw as F

def main():
	df = pd.read_csv('data/campsie_winter_snow.csv', header=None)
	t = df[0]
	x = df[1]
	x = F.detrend(t,x)
	
	df_pdo = pd.read_csv('data/pdo.csv', header=None)	
	x_pdo = df_pdo[df_pdo[0].tolist().index(min(t)):df_pdo[0].tolist().index(max(t))+1][1]
	x_pdo = F.detrend(t,x_pdo)
	
	# cwt example
	freqs, power, sig95, coi, coefs, alpha = F.cwt(t, x)
	F.plot_wt(t, freqs, coefs, sig95, coi, fn='cwt_test.png')
	
	# xwt example
	xfreqs, xpower, xsig95, xcoi, xcoefs, a_xcoefs = F.xwt(t, x_pdo, coefs, alpha, max(coi))
	F.plot_wt(t, xfreqs, xcoefs, xsig95, xcoi, fn='xwt_test.png', xwt=True, arrows=True)
	
	# periodogram example
	conf = [0.85, 0.90, 0.95, 0.99]
	fd,po,tabtchi,theored = F.pdgram(x, conf=conf)
	F.plot_pdgram(fd, po ,tabtchi, theored, t, show_peaks=0.9, conf=conf, fn='pdgram_test.png', legend=False)
	
	
main()
