import os
import sys
import yaml
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from builder import CfgLoader
 
def read_info(filename):
    with open(filename, 'r') as f:
        data = yaml.load(f.read(), CfgLoader)
    return data

def get_best_lr(data):
    for fn, fn_data in data.items():
        keys = list(fn_data.keys())
        bests = [data['best'][0] for data in fn_data.values()]
        best_key = keys[bests.index(min(bests))]
        for k in keys:
            if k != best_key: del fn_data[k]
    return data

def main():
  base_dir = 'runs/pycutest/'
  plt.rcParams["font.family"] = "Times New Roman"
  # Set general font size
  plt.rcParams['font.size'] = '28'
  font1 = {'family': 'Times New Roman',
           'weight': 'normal',
           'size': 28,
           }
  color1 = ['deepskyblue', 'cornflowerblue', 'rosybrown', 'tomato', 'darkred']
#  color1 = ['deepskyblue', 'cornflowerblue', 'rosybrown', 'tomato', 'darkred']
#  color2_tmp = [(16,70,128),(49,124,183),(182,215,232),(246,178,147),(220,109,87),(183,34,48),(109,1,31)]
#  color2 = []
#  for tt in color2_tmp:
#      string = '#'
#      for t in tt: string += str(hex(t))[-2:]
#      color2.append(string.upper())
#  print(color2)
  color2 = ['#FBB45D', '#EF8183', '#699ED4', '#A4D9BB', '#B88CC0', '#C61C22', '#BFC0C2']

  ##################
  # plot reuse_rate-time
  ##################
  plt.rcParams['font.size'] = '8'
  color2 = ['#000000', '#f9403f', '#ffa03e', '#b756d7', '#10a37f', '#6098f9', '#fb6f66']
#  color2 = ['blue']
  files = ['ablation_bound_withtime.yaml', 'ablation_bound_withtime_test1.yaml']
  fig = plt.figure(figsize=(5, 2.5))
  # set acc axes
  left, bottom, width, height = 0.15,0.18,0.8,0.8
  ax1 = fig.add_axes([left,bottom,width,height])
  ax1.set_xlabel('Reuse Rate')
  ax1.set_ylabel('Time cost per step')
#  ax1.set_ylim(0.,min(900., max_y))
#  ax1.set_xlim(0,60, 1)

  all_reuse_rates, all_costs = {}, {}
  for _file in files:
      reuse_rates, costs = {}, {}
      bound_data = read_info(os.path.join(base_dir, _file))
      for j, (bound, data) in enumerate(bound_data.items()):
          for i, fn in enumerate(data.keys()):
#              if fn != 'BOXPOWER': continue
              if fn != 'ARGTRIGLS': continue
              print(f'Plot {fn}')
              reuse_rate = sum(data[fn]['num_reuse']) / (data[fn]['num_sample'] * data[fn]['num_iter'])
              reuse_rates.setdefault(fn, []).append(reuse_rate)
              costs.setdefault(fn, []).append(data[fn]['time']/data[fn]['num_iter'])
      for key in reuse_rates.keys():
          all_reuse_rates.setdefault(key, []).append(reuse_rates[key])
          all_costs.setdefault(key, []).append(costs[key])
  for i, fn in enumerate(all_reuse_rates.keys()):
      x = np.array(all_reuse_rates[fn])
      y = np.array(all_costs[fn])
      ax1.plot(x.mean(axis=0), y.mean(axis=0), "o-", label=fn, lw=1, c=color2[i])
      ax1.fill_between(x.mean(axis=0), y.mean(axis=0)-y.std(axis=0), y.mean(axis=0)+y.std(axis=0), alpha=0.5, facecolor=color2[i])

  # set legend
  h1, l1 = ax1.get_legend_handles_labels()
  ax1.legend(h1, l1, loc=1)
  plt.savefig(os.path.join(base_dir, f'ablation_reuseRate_cost.pdf') )

  ##################
  # plot bound
  ##################
  plt.rcParams['font.size'] = '8'
  color2 = ['#000000', '#f9403f', '#ffa03e', '#b756d7', '#10a37f', '#6098f9', '#fb6f66']
  bound_data = read_info(os.path.join(base_dir, 'ablation_bound.yaml'))
  fns = bound_data[0].keys()
  for fn in fns:
      print(f'Plot {fn}')
      fig = plt.figure(figsize=(5, 2.5))
      # set acc axes
      left, bottom, width, height = 0.12,0.18,0.8,0.8
      ax1 = fig.add_axes([left,bottom,width,height])
      ax1.set_xlabel('Iter')
      ax1.set_ylabel('Function Object')
      max_y = 0
      for data in bound_data.values():
          max_y = max(max_y, max(data[fn]['obj'][0]))
      ax1.set_ylim(0.,min(900., max_y))
#      ax1.set_xlim(0,60, 1)

      for i, (bound, data) in enumerate(bound_data.items()):
          print(f'Plot bound={bound}')
          y = data[fn]['obj'][0]
          x = range(0, len(y))
          if bound == 0: label = 'b=0'
          elif bound == 1: label = r'b=$\eta$'
          else: label = r'b=%d$\eta$'%bound
          ax1.plot(x, y, "-", label=label, lw=1, c=color2[i])

      # set legend
      h1, l1 = ax1.get_legend_handles_labels()
      ax1.legend(h1, l1, loc=1)
      plt.savefig(os.path.join(base_dir, f'ablation_bound_{fn}.pdf') )

  ##################
  # plot num_samples
  ##################
  plt.rcParams['font.size'] = '8'
  sample_data = read_info(os.path.join(base_dir, 'ablation_num_sample.yaml'))
  for fn, fn_data in sample_data.items():
      print(f'Plot {fn}')
      fig = plt.figure(figsize=(5, 2.5))
      # set acc axes
      left, bottom, width, height = 0.12,0.18,0.8,0.8
      ax1 = fig.add_axes([left,bottom,width,height])
      ax1.set_xlabel('Iter')
      ax1.set_ylabel('Function Object')
      max_y = 0
      for data in bound_data.values():
          max_y = max(max_y, max(data[fn]['obj'][0]))
      ax1.set_ylim(0.,min(900., max_y))
#      ax1.set_xlim(0,60, 1)

      for i, (num_sample, data) in enumerate(fn_data.items()):
          print(f'Plot num_sample={num_sample}')
          y = data['obj'][0]
          x = range(0, len(y))
          ax1.plot(x, y, "-", label=f'N={num_sample}', lw=1, c=color2[i])

      # set legend
      h1, l1 = ax1.get_legend_handles_labels()
      ax1.legend(h1, l1, loc=1, ncol=2)
      plt.savefig(os.path.join(base_dir, f'ablation_num_sample_{fn}.pdf') )
  plt.close('all')

  ##################
  # plot best lr
  ##################
  plt.rcParams['font.size'] = '28'
#  color2 = ['#ffa03e', 'purple', 'black','blue', 'red']
  color2 = ['#ffa03e', 'purple', 'black','blue', 'red']
  sgd_files = ['runs/pycutest_SGD/ablation_lr_test0.yaml', 'runs/pycutest_SGD/ablation_lr_test1.yaml', 'runs/pycutest_SGD/ablation_lr_test2.yaml']
  zosgd_files = ['runs/pycutest_zosgd_N2/ablation_lr_test0.yaml', 'runs/pycutest_zosgd_N2/ablation_lr_test1.yaml', 'runs/pycutest_zosgd_N2/ablation_lr_test2.yaml']
  zosignsgd_files = ['runs/pycutest_zosign-sgd_N2/ablation_lr_test0.yaml', 'runs/pycutest_zosign-sgd_N2/ablation_lr_test1.yaml', 'runs/pycutest_zosign-sgd_N2/ablation_lr_test2.yaml']
  zoadam_files = ['runs/pycutest_zoadam_N2/ablation_lr_test3.yaml', 'runs/pycutest_zoadam_N2/ablation_lr_test4.yaml', 'runs/pycutest_zoadam_N2/ablation_lr_test5.yaml']
  lizo_files = ['runs/pycutest_N2/ablation_lr_test6.yaml', 'runs/pycutest_N2/ablation_lr_test7.yaml', 'runs/pycutest_N2/ablation_lr_test8.yaml']
  sgd_data, zosgd_data, zosignsgd_data, zoadam_data, lizo_data = [], [], [], [], []
  for sgd_f, zosgd_f, zosignsgd_f, zoadam_f, lizo_f in zip(sgd_files, zosgd_files, zosignsgd_files, zoadam_files, lizo_files):
      _sgd_data = read_info(sgd_f)
      _zosgd_data = read_info(zosgd_f)
      _zosignsgd_data = read_info(zosignsgd_f)
      _zoadam_data = read_info(zoadam_f)
      _lizo_data = read_info(lizo_f)
      sgd_data.append(get_best_lr(_sgd_data))
      zosgd_data.append(get_best_lr(_zosgd_data))
      zosignsgd_data.append(get_best_lr(_zosignsgd_data))
      zoadam_data.append(get_best_lr(_zoadam_data))
      lizo_data.append(get_best_lr(_lizo_data))
  data = {
#          'SGD': sgd_data,
#          'ZO-SGD': zosgd_data,
#          'ZO-signSGD': zosignsgd_data,
          'ZO-AdaMM': zoadam_data,
          'ReLIZO': lizo_data,
          }
  fns = list(lizo_data[0].keys())
  for fn in fns:
      print(f'Plot {fn}')
      fig = plt.figure(figsize=(8, 8))
      # set acc axes
      left, bottom, width, height = 0.18,0.10,0.8,0.85
      ax1 = fig.add_axes([left,bottom,width,height])
      # Set tick font size
      for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
          label.set_fontsize(16)
      ax1.set_xlabel('Iteration')
      ax1.set_ylabel('Function Value F(x)')
#      max_y, min_y = 0, math.inf
#      for tmp in data.values():
#          for tt in tmp[fn].values():
#              max_y = max(max_y, max(tt['obj'][0]))
#              min_y = min(min_y, min(tt['obj'][0]))
#      ax1.set_ylim(0.,min(900., max_y))
#      ax1.set_xlim(0,60, 1)

      for i, (label, tmp) in enumerate(data.items()):
          y = []
          for ttmp in tmp:
              print(f'Plot solver={label}, best_lr={list(ttmp[fn].keys())}')
              ttmp = ttmp[fn]
              y.append(ttmp[list(ttmp.keys())[0]]['obj'][0])

          y = np.array(y)
          y_mean = y.mean(axis=0)
          y_std = y.std(axis=0)
          x = range(0, len(y_mean))
          ax1.plot(x, y_mean, "-", label=label, lw=2, c=color2[i])
          ax1.fill_between(x, y_mean-y_std, y_mean+y_std, alpha=0.5, facecolor=color2[i])
          print(label, min(y_mean), y.min(axis=1))

      # set legend
      h1, l1 = ax1.get_legend_handles_labels()
      ax1.legend(h1, l1, loc=1, ncol=1)
      plt.savefig(os.path.join(base_dir, f'N2_best_lr_{fn}.pdf') )
  plt.close('all')
  assert 0

  ##################
  # plot bound - num_samples
  ##################
  plt.rcParams['font.size'] = '8'
  #color2 = ['#FBB45D', '#EF8183', '#699ED4', '#A4D9BB', '#B88CC0', '#C61C22', '#BFC0C2']
  color2 = ['#000000', '#f9403f', '#ffa03e', '#b756d7', '#10a37f', '#6098f9', '#fb6f66']
  sample_data1 = read_info(os.path.join(base_dir, 'ablation_bound_numsample_withNumReuse.yaml'))
  others = ['ablation_bound_numsample_test1.yaml', 'ablation_bound_numsample_test2.yaml', 'ablation_bound_numsample_test3.yaml']
  other_datas = [read_info(os.path.join(base_dir, name)) for name in others]

  base_dir = 'runs/pycutest_zoadam/'
  files = ['ablation_num_sample_test4.yaml', 'ablation_num_sample_test5.yaml', 'ablation_num_sample_test6.yaml', 'ablation_num_sample_test7.yaml']
  zoadam_datas = [read_info(os.path.join(base_dir, name)) for name in files]
  base_dir = 'runs/pycutest_zosgd/'
  zosgd_datas = [read_info(os.path.join(base_dir, name)) for name in files]
  base_dir = 'runs/pycutest_zosign-sgd/'
  zosignsgd_datas = [read_info(os.path.join(base_dir, name)) for name in files]
  base_dir = 'runs/pycutest/'
  for fn, fn_data in sample_data1.items():
      print(f'Plot {fn}')
      fig = plt.figure(figsize=(5, 2.5))
      # set acc axes
      left, bottom, width, height = 0.15,0.18,0.8,0.8
      ax1 = fig.add_axes([left,bottom,width,height])
      ax1.spines['right'].set_visible(False)
      ax1.spines['top'].set_visible(False)

      ax1.set_xlabel('Number of Samples Per Step, N')
      ax1.set_ylabel('F(x*)')
#      ax1.set_ylim(0.,min(900., max_y))
#      ax1.set_xlim(0,60, 1)

      for tt, (num_sample, tmp) in enumerate(fn_data.items()):
          reuse_rates = dict(data=[], y=[], ytext=[])
          for i, (bound, data) in enumerate(tmp.items()):
              if bound == 10: continue
              print(f'Plot num_sample={num_sample}, bound={bound}')
              y = data['best']
              reuse_rate = sum(data['num_reuse']) / (data['num_sample'] * data['num_iter'])
              reuse_rates['data'].append(reuse_rate)
              for data in other_datas:
                  y.append(data[fn][num_sample][bound]['best'][0])
              y = np.array(y)
              x = [f'{num_sample}']
              if tt == 0:
                  if bound == 0: label = 'b=0'
                  elif bound == 1: label = r'b=$\eta$'
                  else: label = r'b=%d$\eta$'%bound
              else:
                  label = None
#              label = r'b=%d$\eta$'%bound if tt == 0 else None
#              ax1.scatter(x, y, color=color2[i], marker='o', label=label)
              ax1.errorbar(x,[y.mean()],[y.std()], color=color2[i], marker='o', label=label, ms=4, mew=0.5, alpha=0.6)
              reuse_rates['y'].append(y.mean())
              reuse_rates['ytext'].append(y.mean())
          # plot reuse_rate

          ytext = reuse_rates['ytext']
          argidx = np.array(ytext).argsort()
          ylim = ax1.get_ylim()
          unit = (ylim[1] - ylim[0]) / 10
          for idx in range(1, len(argidx)):
              if ytext[argidx[idx]] - ytext[argidx[idx-1]] < 1.2*unit: ytext[argidx[idx]] = ytext[argidx[idx-1]] + 1.2*unit
          reuse_rates['ytext'] = ytext 
          for ttt in range(len(reuse_rates['data'])):
              ax1.annotate('%.2f'%reuse_rates['data'][ttt], xy=(tt,reuse_rates['y'][ttt]), xytext=(tt+0.2,reuse_rates['ytext'][ttt]), arrowprops={'arrowstyle':"->", 'color':color2[ttt]}, color=color2[ttt], fontsize=6, bbox=dict(boxstyle="round", fc="none", ec=color2[ttt]))

      # set legend
      h1, l1 = ax1.get_legend_handles_labels()
      ax1.legend(h1, l1, loc=1, ncol=2)
      plt.savefig(os.path.join(base_dir, f'ablation_bound_numsample_{fn}.pdf') )

      # plot zoadam
      def _plot(datas, label, marker, color):
          print(f'Plot {fn} for {label}')
          for tt, (num_sample, tmp) in enumerate(datas[0][fn].items()):
              print(f'Plot num_sample={num_sample}')
              y = tmp['best']
              for data in datas[1:]:
                  y.append(data[fn][num_sample]['best'][0])
              y = np.array(y)
              x = [f'{num_sample}']
              _label = label if tt == 0 else None
    #          ax1.scatter(x, y, color=color2[i], marker='o', label=label)
              ax1.errorbar(x,[y.mean()],[y.std()], color=color, marker=marker, label=_label, ms=4, mew=0.5, alpha=0.6)
    
      # plot zoadam
      _plot(zoadam_datas, 'ZO-AdaMM', '*', color2[-1])
      # plot zosgd
      _plot(zosgd_datas, 'ZO-SGD', 's', color2[-2])
      # plot zosign-sgd
      _plot(zosignsgd_datas, 'ZO-signSGD', '^', color2[-3])
      # set legend
      h1, l1 = ax1.get_legend_handles_labels()
      ax1.legend(h1, l1, loc=1, ncol=2)
      plt.savefig(os.path.join(base_dir, f'ablation_bound_numsample_allSolver_{fn}.pdf') )

  plt.close('all')

if __name__ == '__main__':
  main()

