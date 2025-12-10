import numpy as np
import dynesty
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing
import matplotlib as mpl
import time
import os
import pandas as pd
import corner
from scipy.stats import uniform, norm, truncnorm, loguniform
# from models import ModelFunc, linearModel
from modelsAdvanced import ModelFunc, linearModel
import json
import julian


class TTVFit:
    def __init__(self, target, data, period0, bounds, names=[], kind='Linear' ,rej_sigma=None,multiprocessing = False, sampler='dynesty',sampler_kwargs={},lin_parameters=None, fixed_params=None):
        self.target = target
        self.data = data.sort_values('T_mid').reset_index(drop=True)
        if 'type' not in self.data.columns:
            print('Assuming transit data')
            self.data['type'] = 'transit'
        self.period0 = period0
        self.lin_parameters = lin_parameters
        self.kind = kind
        self.data.index = range(len(self.data))
        self.Prior = self.NestedPrior

        self.data['Epoch'] = np.array(np.round((self.data['T_mid']-self.data['T_mid'].min())/period0),dtype=int)
        self.multiprocessing = multiprocessing
        self.bounds = bounds
        self.names = names
        self.sampler_name = sampler
        self.sampler_kwargs = sampler_kwargs
        if 'Label' not in self.data.columns:
            self.data['Label'] = ['data']*len(self.data)
        MF = ModelFunc(kind, fixed_params=fixed_params)
        self.Model = MF.get_model()
        self.fixed_params = fixed_params
        self.N = self.data['Epoch'].max()
        self.rej_sigma = rej_sigma
        self.data['Valid'] = True

    def getJudgements(self):
        valid_mask = self.data['Valid']
        obs = np.array(self.data[valid_mask]['T_mid'])
        pre = np.array(self.data[valid_mask]['T_mid_pre'])
        error = np.array(self.data[valid_mask]['Uncertainty'])
        k = len(self.bounds)
        nu = len(obs) - k
        c=[]
        for i in range(len(obs)):
            c.append((obs[i]-pre[i])**2/error[i]**2)
        c=np.array(c)
        chi2 = c.sum()
        judgements = {'chi2':chi2,'chi2nu':chi2/nu,'nu':nu,'BIC':chi2+k*np.log(len(obs)),'AIC':chi2+2*k,'k':k,'N':len(obs)}
        return judgements
    
    def Likelihood(self, params):
        valid_mask = self.data['Valid']
        t_obs = np.array(self.data[valid_mask]['T_mid'],dtype=object)
        t_err = np.array(self.data[valid_mask]['Uncertainty'],dtype=object)
        epochs = np.array(self.data[valid_mask]['Epoch'],dtype=object)
        t_pre = self.Model(params,self.N)
        t_pre = np.array(t_pre,dtype=object)
        x = []
        for i in range(len(t_obs)):
            xi = (t_obs[i]-t_pre[epochs[i]])**2/t_err[i]**2
            x.append(xi)
        x = np.array(x,dtype=object)
        return -0.5*np.sum(x)
    
    def NestedPrior(self, params):
        params_list = []
        for i, par in enumerate(params):
            if self.bounds[i][-1]=='U':
                params_list.append(uniform.ppf(par, self.bounds[i][0], self.bounds[i][1]-self.bounds[i][0]))
            elif self.bounds[i][-1]=='N':
                params_list.append(norm.ppf(par, self.bounds[i][0], self.bounds[i][1]))
            elif self.bounds[i][-1]=='TN':
                mean, std ,low, high =   self.bounds[i][:-1]
                low_n, high_n = (low - mean) / std, (high - mean) / std  # standardize
                params_list.append(truncnorm.ppf(par, low_n, high_n, loc=mean, scale=std))
            elif self.bounds[i][-1]=='LU':
                params_list.append(loguniform.ppf(par, self.bounds[i][0], self.bounds[i][1]))
            else:
                ValueError('Prior distribution not recognized, should be "U", "N", "TN", "LU"')
        return np.array(params_list)

    def Probability(self, params):
        lp = self.Prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.Likelihood(params)
    
    def Method(self):
        sampler_kwargs = self.sampler_kwargs
        N_threads = max(multiprocessing.cpu_count() - 1,1)
        if self.multiprocessing == True:pool = Pool(processes=N_threads)
        else:pool = None

        if self.multiprocessing == True:pool = Pool()
        else:pool = None
        ndim = len(self.bounds)

        start = time.time()
        if self.sampler_name == 'dynesty':
            print('Start dynesty')
            sampler = dynesty.NestedSampler(
                self.Likelihood,
                self.Prior,
                ndim,
                bound = 'multi', sample='rwalk', pool = pool, queue_size=N_threads,\
                nlive=sampler_kwargs['nlive'],  # number of live points
            )
            # print('The citations of %s:\n'%self.sampler_name,sampler.citations)
            sampler.run_nested(dlogz=sampler_kwargs['dlogz'],print_progress=True)
            print('Finish dynesty')
        elif self.sampler_name == 'dynamic_dynesty':
            print('Start dynamic dynesty')
            sampler = dynesty.DynamicNestedSampler(
                self.Likelihood, 
                self.Prior,
                ndim,
                bound = 'multi', sample='rwalk', pool = pool, queue_size=N_threads,\
            )
            # print('The citations of %s:\n'%self.sampler_name,sampler.citations)
            sampler.run_nested(print_progress=True,**self.sampler_kwargs)
            print('Finish dynamic dynesty')
        else:
            raise ValueError("Sampler not recognized")

        end = time.time()
        total_time = end - start
        if self.multiprocessing == True:
            pool.close()
            print("Multiprocessing took {0:.1f} seconds".format(total_time))
        else:
            print("Serial took {0:.1f} seconds".format(total_time))
        
        results = sampler.results
        samples, weights = results.samples, results.importance_weights()
        self.mean, self.cov = dynesty.utils.mean_and_cov(samples, weights)
        self.logz = results.logz[-1]
        self.logzerr = results.logzerr[-1]
        samples = dynesty.utils.resample_equal(results.samples, weights)
        self.samples = samples
        self.parameters_pre = [(v[1], v[2]-v[1], v[1]-v[0]) for v in zip(*np.percentile(self.samples, [16, 50, 84],axis=0))]
        self.mean, self.cov = dynesty.utils.mean_and_cov(samples, weights)
        self.logz = results.logz[-1]
        self.logzerr = results.logzerr[-1]
        self.params = [x[0] for x in self.parameters_pre]
        self.getTmids()

        if self.rej_sigma is not None:
            self.sigma = self.data[self.data['Valid']]['Residual'].std(ddof=1)*self.rej_sigma
            m = len(self.data[self.data['Valid']])
            for i in range(len(self.data)):
                self.data.loc[i,'Valid'] *= (np.abs(self.data['Residual']) <= self.sigma)[i]
            n = len(self.data[self.data['Valid']])
            return m - n
        else:
            return 0

    def getTmids(self):
        t_tras = self.Model(self.params,self.N)
        t_mids = []
        for i in range(len(self.data['Epoch'])):
            t_mids.append(t_tras[self.data['Epoch'][i]])
        self.data['T_mid_pre'] = np.array(t_mids,dtype=object)
        self.data['Residual'] = np.array(self.data['T_mid'] - self.data['T_mid_pre'], dtype=object)

    def runMethod(self, path=None, suffix=''):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        key = self.Method()
        keys = [key]
        n = 1
        while key>0:
            key = self.Method()
            keys.append(key)
            n += 1
            # if (keys[-1]==keys[-2] or keys[-1]>15):
            #     break
        if self.rej_sigma is not None:
            if n > 1:
                print('Outliers are removed with rejection = %.5f, after %d runs.'%(self.sigma,n))
                print('The keys of outliers are:',keys)
            else:
                print('No data points are removed.')
        np.save(path+'samples%s.npy'%suffix,self.samples)
        all_params = {}
        all_params['target'] = self.target
        all_params['names'] = self.names
        all_params['bounds'] = self.bounds
        all_params['period0'] = self.period0
        all_params['kind'] = self.kind
        all_params['sampler'] = self.sampler_name
        all_params['sampler_kwargs'] = self.sampler_kwargs
        all_params['lin_parameters'] = self.lin_parameters
        all_params['fixed_params'] = self.fixed_params
        all_params['params'] = self.params
        all_params['parameters_pre'] = self.parameters_pre
        all_params['logz'] = self.logz
        all_params['logzerr'] = self.logzerr
        all_params['mean'] = list(self.mean)
        all_params['cov'] = [list(c) for c in self.cov]
        all_params['judgements'] = self.getJudgements()
        all_params['Date'] = time.ctime()+' (UTC+8)'

        if path != None:
            self.data.to_csv(path+'posterior_data%s.csv'%suffix,index=False)
            with open(path+'posteriors%s.json'%suffix,'w') as f:
                json.dump(all_params,f,indent=2)
        return n

    def plotFit(self, path=None, suffix='', ax=None, unitt='s'):
        # Configure scientific plotting style
        plt.rcParams.update({
            'font.size': 18,
            'font.family': 'serif',
            'axes.linewidth': 1.4,
            'xtick.major.size': 8,
            'ytick.major.size': 8,
            'xtick.minor.size': 4,
            'ytick.minor.size': 4,
            'axes.labelpad': 8,
            'lines.linewidth': 1.8,
            'lines.markersize': 10,
            'axes.titlepad': 15,
            'legend.frameon': True,
            'legend.framealpha': 0.8,
            'xtick.major.width': 2.0,
            'ytick.major.width': 2.0,
            'xtick.minor.width': 1.5,
            'ytick.minor.width': 1.5,
        })

        self.unitt = unitt
        unit_conversion = {'s': 86400, 'min': 1440, 'h': 24}
        multi = unit_conversion.get(unitt, 1)

        # Color management
        cmap = plt.cm.get_cmap('tab10')
        unique_labels = sorted(set(self.data['Label']))
        color_dict = {label: cmap(i/len(unique_labels)) for i, label in enumerate(unique_labels)}
        
        mask = self.data['Valid']
        numbers = self.data[mask].index

        if ax is None:
            fig, ax = plt.subplots(figsize=(24, 10), dpi=100)
            fig.subplots_adjust(left=0.1, right=0.85)
        else:
            fig = ax.figure

        # Common plot elements
        base_zorder = 10
        errorbar_kwargs = {
            'ms': 8,
            'capsize': 6,
            'capthick': 2,
            'elinewidth': 2,
            'zorder': base_zorder
        }

        if 'Linear' in self.kind:
            # Linear model plotting
            valid_data = self.data[mask]
            for i in numbers:
                ax.errorbar(valid_data.loc[i,'Epoch'], (valid_data.loc[i,'T_mid']-valid_data.loc[i,'T_mid_pre'])*multi, fmt='o' if valid_data.loc[i,'Reference'] == 'This work' else 'd',
                            yerr=valid_data.loc[i,'Uncertainty']*multi,label=valid_data.loc[i,'Label'],
                            color=color_dict[valid_data.loc[i,'Label']],
                            ecolor='#666666',**errorbar_kwargs)

            if not np.all(mask):
                invalid_data = self.data[~mask]
                ax.errorbar(invalid_data['Epoch'],
                        (invalid_data['T_mid'] - invalid_data['T_mid_pre']) * multi,
                        yerr=invalid_data['Uncertainty'] * multi,
                        color='#AAAAAA',
                        ecolor='#888888',
                        label='Outliers',
                        alpha=0.6,
                        fmt='x',
                        **errorbar_kwargs)

            ax.axhline(0, c='k', lw=1.5, ls='--', zorder=base_zorder-1, label = 'Linear Model')
            
            if self.rej_sigma is not None:
                ax.fill_between(self.data['Epoch'],
                            self.sigma * multi,
                            -self.sigma * multi,
                            color='#1f77b4',
                            alpha=0.15,
                            label=fr'${self.rej_sigma}\sigma$ area')

        else:
            t_pre = np.array(self.Model(self.params,self.N))
            if self.lin_parameters is None:
                t_pre_l = np.array(linearModel(self.params[:2],self.N))
                t_tras_lin = linearModel(self.params[:2],self.N)
                t_mids_lin = []
                for i in range(len(self.data['Epoch'])):
                    t_mids_lin.append(t_tras_lin[self.data['Epoch'][i]])
                data_lin_pre = np.array(t_mids_lin)
            else:
                t_pre_l = np.array(linearModel(self.lin_parameters,self.N))
                t_tras_lin = linearModel(self.lin_parameters,self.N)
                t_mids_lin = []
                for i in range(len(self.data['Epoch'])):
                    t_mids_lin.append(t_tras_lin[self.data['Epoch'][i]])
                data_lin_pre = np.array(t_mids_lin)
            # Non-linear model plotting
            t_pre = np.array(self.Model(self.params, self.N))
            t_pre_l = np.array(linearModel(self.params[:2], self.N)) if self.lin_parameters is None \
                    else np.array(linearModel(self.lin_parameters, self.N))
            
            model_line, = ax.plot(range(self.N), (t_pre[:self.N] - t_pre_l[:self.N]) * multi,
                                color='#66ccff',
                                lw=2,
                                zorder=base_zorder+1,
                                label=f'{self.kind} Model')

            valid_data = self.data[mask]
            for i in numbers:
                ax.errorbar(valid_data.loc[i,'Epoch'], (valid_data.loc[i,'T_mid']-data_lin_pre[i])*multi,fmt='o' if valid_data.loc[i,'Reference'] == 'This work' else 'd',
                            yerr=valid_data.loc[i,'Uncertainty']*multi,
                            label=valid_data.loc[i,'Label'],color=color_dict[valid_data.loc[i,'Label']],
                            ecolor='#666666',
                            **errorbar_kwargs)

            if not np.all(mask):
                invalid_data = self.data[~mask]
                ax.errorbar(invalid_data['Epoch'],
                        (invalid_data['T_mid'] - data_lin_pre[~mask]) * multi,
                        yerr=invalid_data['Uncertainty'] * multi,
                        color='#AAAAAA',
                        ecolor='#888888',
                        label='Outliers',
                        alpha=0.6,
                        fmt='x',
                        **errorbar_kwargs)

            ax.axhline(0, c='k', lw=1.5, ls='--', zorder=base_zorder-1,   label = 'Linear Model')
            
            if self.rej_sigma is not None:
                ax.fill_between(range(self.N),
                            (t_pre[:self.N] - t_pre_l[:self.N] + self.sigma) * multi,
                            (t_pre[:self.N] - t_pre_l[:self.N] - self.sigma) * multi,
                            color='#1f77b4',
                            alpha=0.15,
                            label=fr'${self.rej_sigma}\sigma$ area')

        # Date formatting
        if 'Date' not in self.data.columns:
            self.data['Date'] = [julian.from_jd(x) for x in self.data['T_mid']]
        date_label = pd.date_range(
            start=self.data['Date'][0],
            end=self.data['Date'][len(self.data['Date'])-1],
            periods=10



            
        )
        date_ticks = np.linspace(self.data['Epoch'][0],self.data['Epoch'][len(self.data['Epoch'])-1],10)
        xlim = ax.get_xlim()
        axT = ax.twiny()
        axT.set_xlim(xlim)
        axT.set_xticks(date_ticks,[i.strftime('%Y-%m-%d') for i in date_label])
        axT.tick_params(axis='x', which='major', labelsize=20, direction='in')
        # Axis labeling
        ax.set_ylabel(f"O−C [{unitt}]", fontsize=30, labelpad=10)
        ax.set_xlabel("Epoch", fontsize=30, labelpad=10)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', labelsize=24, direction='in',right=True)
        
        # Grid and ticks
        ax.grid(True, which='major', ls='--', alpha=0.6)
        ax.grid(True, which='minor', ls=':', alpha=0.3)
        
        # Legend handling
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for h, l in zip(handles, labels):
            if l not in unique_labels:
                unique_labels.append(l)
                unique_handles.append(h)
        
        ax.legend(unique_handles, unique_labels,
                borderaxespad=0.,
                fontsize=20,
                )
        if self.kind != 'Linear':
            if self.kind=='Pdot':
                unit = 'ms/yr'
            elif self.kind=='Sine':
                unit = 's'
            else:
                unit = 'radian/epoch'
            ax.set_title(f"{self.target} TTV Fit: {self.kind} Model, " + r'%s$ = %.2f\pm%.3f\ %s$'%(self.names[-1],self.parameters_pre[-1][0],np.abs(np.max(self.parameters_pre[-1][1:])),unit), fontsize=34, pad=15)
        else:
            ax.set_title(f"{self.target} TTV Fit: {self.kind} Model", fontsize=34, pad=15)


        if 'Linear' in self.kind:
            only_data = self.data[self.data['Reference']=='This work'].copy()
            fig_l, ax_l = plt.subplots(figsize=(24, 10), dpi=100)
            fig_l.subplots_adjust(left=0.1, right=0.85)
            ax_l.errorbar(only_data['Epoch'], (only_data['T_mid']-only_data['T_mid_pre'])*multi, 
                        yerr=only_data['Uncertainty']*multi,label=only_data['Label'],
                        ecolor='#666666', fmt='o',**errorbar_kwargs)
            ax_l.axhline(0, c='k', lw=1.5, ls='--', zorder=base_zorder-1)
        if path is not None:
            fig.savefig(f"{path}plot_{self.kind}_{self.target.replace(' ', '-')}{suffix}.png", dpi=300, bbox_inches='tight')
            fig.savefig(f"{path}plot_{self.kind}_{self.target.replace(' ', '-')}{suffix}.pdf", bbox_inches='tight')
            fig.savefig(f"{path}plot_{self.kind}_{self.target.replace(' ', '-')}_transparent{suffix}.pdf", bbox_inches='tight', transparent=True)
            if 'Linear' in self.kind:
                fig_l.savefig(f"{path}plot_only{suffix}.png", dpi=300, bbox_inches='tight')
                fig_l.savefig(f"{path}plot_only{suffix}.pdf", bbox_inches='tight')
                fig_l.savefig(f"{path}plot_only_transparent{suffix}.pdf", bbox_inches='tight', transparent=True)


        plt.rcParams.update(mpl.rcParamsDefault)
        fig_corner = plt.figure(figsize=(3.5*len(self.bounds),3.5*len(self.bounds)))
        corner.corner(self.samples, labels=[x for x in self.names],truths=self.params,fig=fig_corner,
                        show_titles=True,use_math_text=True,quantiles=[0.16, 0.5, 0.84],
                        max_n_ticks=3,title_fmt='.3', title_kwargs={"fontsize": 10},
                        label_kwargs={"fontsize": 10}, fontsize=10,
                        labelpad=0.08)
        # plt.tight_layout()
        plt.savefig(f'{path}corner_{self.kind}_{self.target.replace(" ", "-")}{suffix}.png',dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.savefig(f'{path}corner_{self.kind}_{self.target.replace(" ", "-")}{suffix}.pdf', bbox_inches='tight', pad_inches=0.05)
        plt.savefig(f'{path}corner_{self.kind}_{self.target.replace(" ", "-")}{suffix}_transparent{suffix}.pdf', bbox_inches='tight', pad_inches=0.05, transparent=True)
        # plt.show()
        plt.close('all')