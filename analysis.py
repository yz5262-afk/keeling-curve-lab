"""Reproduce analysis from the archived NOAA CSV. Run with Python 3.13.
Dependencies: numpy pandas scipy matplotlib statsmodels.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.signal import welch
from statsmodels.tsa.stattools import acf

P = Path(__file__).resolve().parent
(P/'figures').mkdir(exist_ok=True)
df = pd.read_csv(P/'data/co2_mm_mlo.csv', comment='#')
# Complete years, NOAA-era observations, before the 2022 eruption/site change.
d = df.loc[df.year.between(1975,2021)].copy().reset_index(drop=True)
d['date'] = pd.to_datetime(dict(year=d.year, month=d.month, day=15))
d['observed'] = (d.average > 0) & (d.unc >= 0) & (d.sdev >= 0)
d.to_csv(P/'data/analysis_months.csv',index=False)
assert len(d)==47*12 and not d.duplicated(['year','month']).any()
t = (d.year + (d.month-.5)/12 - 1998.5).to_numpy()
y = d.average.to_numpy()
valid = d.observed.to_numpy()

def design(t, months, degree=2, harmonics=2):
    cols=[np.ones(len(t))]+[t**k for k in range(1,degree+1)]
    phase=(np.asarray(months)-.5)/12
    for k in range(1,harmonics+1):
        cols += [np.sin(2*np.pi*k*phase),np.cos(2*np.pi*k*phase)]
    return np.column_stack(cols)

def fit(mask=valid, degree=2, harmonics=2, lag=24):
    X=design(t,d.month,degree,harmonics)
    # Calendar-time Newey-West covariance: zero scores at missing months.
    # This preserves actual lag distances rather than compressing time gaps.
    mask=np.asarray(mask) & valid
    base=sm.OLS(y[mask],X[mask]).fit()
    scores=np.zeros_like(X)
    scores[mask]=X[mask]*base.resid[:,None]
    meat=scores.T@scores
    for j in range(1,lag+1):
        cross=scores[j:].T@scores[:-j]
        meat+=(1-j/(lag+1))*(cross+cross.T)
    bread=np.linalg.inv(X[mask].T@X[mask])
    cov=bread@meat@bread * (mask.sum()/(mask.sum()-X.shape[1]))
    robust=base.get_robustcov_results(cov_type='HAC',maxlags=lag,use_correction=True)
    robust.cov_params_default=cov
    return robust,X

model,X=fit()
linear,XL=fit(degree=1)
res=y-X@model.params
season_X=design(np.zeros(12),np.arange(1,13))
season=season_X[:,3:]@model.params[3:]
trend=design(t,d.month)[:,:3]@model.params[:3]
ci=model.conf_int()
acc=2*model.params[2]
acc_ci=(2*ci[2]).tolist()
amp=float(np.ptp(season))
# Propagate HAC covariance to seasonal peak-to-trough uncertainty by simulation.
rng=np.random.default_rng(5262)
draws=rng.multivariate_normal(model.params,model.cov_params(),size=5000)
sdraw=draws[:,3:]@season_X[:,3:].T
amp_ci=np.quantile(np.ptp(sdraw,axis=1),[.025,.975]).tolist()
growth={}
for yr in [1975,1998,2021]:
    tt=yr+.5-1998.5
    v=np.array([0,1,2*tt,0,0,0,0])
    val=float(v@model.params); se=float(np.sqrt(v@model.cov_params()@v))
    growth[str(yr)]={'estimate':val,'ci':[val-1.96*se,val+1.96*se]}
sens=[]
for lag in [12,24,60]:
    m,_=fit(lag=lag)
    sens.append({'check':f'HAC lag {lag} months','acceleration':float(2*m.params[2]),'ci':(2*m.conf_int()[2]).tolist()})
for h in [1,3]:
    m,_=fit(harmonics=h)
    sens.append({'check':f'{h} seasonal harmonic(s)','acceleration':float(2*m.params[2]),'ci':(2*m.conf_int()[2]).tolist()})
for lo,hi in [(1980,2021),(1975,2015)]:
    m,_=fit(mask=(d.year.between(lo,hi).to_numpy() & valid))
    sens.append({'check':f'Window {lo}-{hi}','acceleration':float(2*m.params[2]),'ci':(2*m.conf_int()[2]).tolist()})
# Chronological holdout; same fixed model specification, never shuffle time.
train=(d.year<=2011).to_numpy() & valid; test=(d.year>=2012).to_numpy() & valid
cv={}
for deg in [1,2]:
    A=design(t,d.month,deg)
    m=sm.OLS(y[train],A[train]).fit()
    cv[str(deg)]=float(np.sqrt(np.mean((y[test]-A[test]@m.params)**2)))
# Welch spectra use 12 equally spaced samples/year, and quadratic-only detrending.
detr=y-trend
spectra={}
for nseg in [120,240]:
    f,p=welch(detr,fs=12,window='hann',nperseg=nseg,noverlap=nseg//2,detrend='constant',scaling='density')
    eligible=(f>=.2)&(f<=3)
    peak=float(f[eligible][np.argmax(p[eligible])])
    spectra[str(nseg)]={'peak_frequency':peak,'period_months':12/peak,'bin_spacing':12/nseg}
acf1=float(acf(res,nlags=1,fft=True)[1])
# Independent NOAA annual-growth product: endpoint-based seasonal correction,
# NOT the derivative of our fitted quadratic. Compare, do not call them identical.
growth_file=P/'data/co2_gr_mlo.csv'
official=pd.read_csv(growth_file,comment='#').rename(columns={'ann inc':'noaa_growth','unc':'noaa_unc'})
official=official.loc[official.year.between(1975,2021)].copy()
assert len(official)==47 and official.year.is_unique and official.noaa_growth.notna().all()
official['quadratic_growth']=model.params[1]+2*model.params[2]*(official.year+.5-1998.5)
official['difference']=official.quadratic_growth-official.noaa_growth
official.to_csv(P/'data/growth_comparison.csv',index=False)
comparison={'source':'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_gr_mlo.csv',
            'sha256':hashlib.sha256(growth_file.read_bytes()).hexdigest(),
            'n':len(official),'correlation':float(official.noaa_growth.corr(official.quadratic_growth)),
            'bias_model_minus_noaa':float(official.difference.mean()),
            'rmse':float(np.sqrt(np.mean(official.difference**2))),
            'largest_absolute_difference_year':int(official.iloc[official.difference.abs().argmax()].year)}
# Disjoint time windows fit their own trend and harmonics. The last window is
# 17 years (not 15); seasonal range is the fitted 12 monthly midpoints.
windows=[]
for start,end in [(1975,1989),(1990,2004),(2005,2021)]:
    mask=d.year.between(start,end).to_numpy() & valid
    local_t=t-(start+end)/2+1998.5  # center within this window
    A=design(local_t,d.month)
    fit_local=sm.OLS(y[mask],A[mask]).fit()
    robust_local=fit_local.get_robustcov_results(cov_type='HAC',maxlags=12)
    S=design(np.zeros(12),np.arange(1,13))[:,3:]
    seasonal=S@fit_local.params[3:]
    local_draws=np.random.default_rng(5262+start).multivariate_normal(
        fit_local.params[3:],robust_local.cov_params()[3:,3:],size=5000)
    ranges=np.ptp(local_draws@S.T,axis=1)
    windows.append({'start':start,'end':end,'n':int(mask.sum()),
                    'peak_to_trough':float(np.ptp(seasonal)),
                    'ci':np.quantile(ranges,[.025,.975]).tolist(),
                    'peak_month':int(np.argmax(seasonal)+1),
                    'trough_month':int(np.argmin(seasonal)+1)})
summary={'source':'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv','sha256':hashlib.sha256((P/'data/co2_mm_mlo.csv').read_bytes()).hexdigest(),'n':len(d),'observed_n':int(valid.sum()),'excluded_n':int((~valid).sum()),'start':str(d.date.iloc[0].date()),'end':str(d.date.iloc[-1].date()),'annual_mean_1975':float(d.loc[d.year==1975,'average'].mean()),'annual_mean_2021':float(d.loc[d.year==2021,'average'].mean()),'acceleration':float(acc),'acceleration_ci':acc_ci,'seasonal_peak_to_trough':amp,'seasonal_amplitude_ci':amp_ci,'peak_month':int(np.argmax(season)+1),'trough_month':int(np.argmin(season)+1),'growth':growth,'linear_rmse':float(np.sqrt(np.mean(linear.resid**2))),'quadratic_rmse':float(np.sqrt(np.mean(res[valid]**2))),'residual_lag1_acf':acf1,'holdout_rmse':cv,'sensitivity':sens,'spectra':spectra,'parameters':model.params.tolist(),'noaa_comparison':comparison,'seasonal_windows':windows}
(P/'results.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
fig,axs=plt.subplots(2,1,figsize=(9,6),layout='constrained')
axs[0].plot(d.date,y,color='#427785',lw=.85,label='Monthly mean (input data)')
axs[0].plot(d.date,trend,color='#c26a36',lw=2,label='Fitted quadratic trend')
axs[0].scatter(d.loc[~d.observed,'date'],y[~valid],s=40,facecolors='white',edgecolors='#8b3b42',linewidths=1.3,zorder=5,label='NOAA interpolated (excluded from fit)')
axs[0].set(ylabel='CO2 dry-air mole fraction (ppm)',title='A  |  NOAA Mauna Loa record, 1975-2021'); axs[0].legend(loc='upper left')
sel=d.year.between(2017,2021)
axs[1].plot(d.loc[sel,'date'],y[sel],'.-',color='#427785',label='Monthly mean')
axs[1].plot(d.loc[sel,'date'],(X@model.params)[sel],color='#c26a36',label='Trend + seasonal harmonics')
axs[1].set(ylabel='CO2 (ppm)',xlabel='Year',title='B  |  Five-year close-up');axs[1].legend()
fig.savefig(P/'figures/01_raw_data.svg');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
V=np.column_stack([np.zeros(len(t)),np.ones(len(t)),2*t,np.zeros((len(t),4))])
g=V@model.params; se=np.sqrt(np.einsum('ij,jk,ik->i',V,model.cov_params(),V))
axs[0].plot(d.date,g,color='#c26a36');axs[0].fill_between(d.date,g-1.96*se,g+1.96*se,alpha=.25,color='#c26a36')
axs[0].set(ylabel='Fitted growth rate (ppm/year)',xlabel='Year',title='A  |  Increasing trend slope (95% HAC band)')
axs[1].plot(range(1,13),season,'o-',color='#427785');axs[1].axhline(0,color='gray',lw=.6)
axs[1].set(xticks=range(1,13),xlabel='Month',ylabel='Seasonal deviation (ppm)',title='B  |  Fitted average seasonal cycle')
axs[1].annotate('Peak',xy=(np.argmax(season)+1,max(season)),xytext=(7,3),arrowprops={'arrowstyle':'->'})
axs[1].annotate('Trough',xy=(np.argmin(season)+1,min(season)),xytext=(5,-3),arrowprops={'arrowstyle':'->'})
fig.savefig(P/'figures/02_key_results.svg');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
for nseg in [120,240]:
    f,p=welch(detr,fs=12,window='hann',nperseg=nseg,noverlap=nseg//2,detrend='constant',scaling='density')
    axs[0].semilogy(f[1:],p[1:],label=f'{nseg}-month segments')
axs[0].set(xlim=(0,3),xlabel='Frequency (cycles/year)',ylabel='PSD (ppm² / (cycles/year))',title='A  |  Spectrum after removing quadratic trend');axs[0].legend(fontsize=8)
axs[0].annotate('Annual cycle',xy=(1,100),xytext=(1.5,30),arrowprops={'arrowstyle':'->'})
axs[1].plot(d.date,res,lw=.7,color='#427785');axs[1].axhline(0,color='gray',lw=.5)
axs[1].set(xlabel='Year',ylabel='Residual (ppm)',title='B  |  Residual structure remains')
fig.savefig(P/'figures/03_diagnostics.svg');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
axs[0].plot(official.year,official.noaa_growth,'o-',ms=3,lw=.8,color='#427785',label='NOAA annual growth')
axs[0].plot(official.year,official.quadratic_growth,lw=2,color='#c26a36',label='Quadratic trend derivative')
axs[0].set(xlabel='Year',ylabel='CO2 growth (ppm/year)',title='A  |  Independent NOAA benchmark')
axs[0].legend(fontsize=8)
labels=[f"{w['start']}–{w['end']}" for w in windows]
vals=[w['peak_to_trough'] for w in windows]
err=np.array([[v-w['ci'][0] for w,v in zip(windows,vals)],
              [w['ci'][1]-v for w,v in zip(windows,vals)]])
axs[1].errorbar(labels,vals,yerr=err,fmt='o-',capsize=4,color='#427785')
axs[1].set(xlabel='Disjoint window',ylabel='Fitted seasonal range (ppm)',
           title='B  |  Window-specific seasonal cycle')
fig.savefig(P/'figures/04_replication_extensions.svg');plt.close(fig)
print(json.dumps(summary,indent=2))
