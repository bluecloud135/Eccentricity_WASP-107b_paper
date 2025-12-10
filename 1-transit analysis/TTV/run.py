from ttvfit import *
import matplotlib
matplotlib.use('Agg')
import yaml
import os
import time
start_time = time.time()
models = ['Linear', 'Pdot']
sigma_list = [None]
suffix_list = ['']

with open('init.yaml', 'r') as f:
    init = yaml.safe_load(f)

count = 0
for i,key in enumerate(init):
    target = init[key]['name']
    period0 = init[key]['period']
    outdir = init[key]['outdir']
    for suffix in suffix_list:
        print('suffix',suffix)
        try:
            data = pd.read_csv(outdir+'data%s.csv'%suffix)
            count += 1
        except:
            print('\nTarget %d: %s, data missing, Skipping'%(i+1,target))
            continue
        times0 = data['T_mid'].min()
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        path = outdir + '0%s/'%suffix

        if not os.path.exists(path):
            os.makedirs(path)
        a = 0.001
        times0_est = [times0,a,'N']
        b = 0.00001
        period_est = [period0,b,'N']

        pdot_est = [-200,200,'U']

        sampler = 'dynamic_dynesty'
        sampler_kwargs = {
            'dynamic_dynesty':{
                'nlive_init':10000,
                'dlogz_init':0.001,
            }

        }
        for sigma in sigma_list:
            linear_params = None
            ns = []
            for j,model in enumerate(models):
                print('\nSuffix: %s, Target %d: %s, Threshold: %s sigma, Model %d: %s'%(suffix,i+1,target,sigma,j+1,model))
                bounds =[
                    times0_est,
                    period_est,
                ]
                names = [
                    r'$t_0$',
                    r'$Period$',
                ]
                if model == 'Pdot':
                    names.append(r'$\dot{P}$')
                    bounds.append(pdot_est)
                elif model == 'Linear':
                    pass
                else:
                    raise ValueError("Not a valid model")
                print('\n'.join(['%s : %s'%(k,v) for k,v in zip(names,bounds)]))
                realpath = path+'%ssigma/'%sigma+model+'/'
                TTVFIT = TTVFit(target, data.copy(),period0,bounds,names,model,rej_sigma=sigma,multiprocessing=True,sampler=sampler,sampler_kwargs=sampler_kwargs[sampler],lin_parameters=linear_params)
                n = TTVFIT.runMethod(path=realpath)
                ns.append(n)
                TTVFIT.plotFit(path=realpath)
                # suffix = ''
                title = '%s Transit Timing Variation Analysis'%target
                if model == 'Linear':
                    linear_params = TTVFIT.params
                    print('linear_params',linear_params)
            if np.all([n==1 for n in ns]):
                print('No data points rejected, no need for high sigma run')
                break
end_time = time.time()
print('Total time taken: %.2f seconds'%(end_time-start_time))
print('Total targets processed:',count)