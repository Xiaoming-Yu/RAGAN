from options.train_options import TrainOptions
import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from util.visualizer import Visualizer
import sys


opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset_size = len(data_loader) * opt.batchSize
visualizer = Visualizer(opt)

if opt.model == 'supervised':
    from models.SuperNet import SuperNet
    model = SuperNet()
elif opt.model == 'unsupervised':
    from models.USuperNet import USuperNet
    model = USuperNet()
model.initialize(opt)


total_steps = 0
lr = opt.lr
for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    save_result = True
    for i, data in enumerate(data_loader):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.update_model(data)
        
        if save_result or total_steps % opt.display_freq == 0:
            save_result = save_result or total_steps % opt.update_html_freq == 0
            print('model:{} dataset:{}'.format(opt.model,opt.name))
            visualizer.display_current_results(model.get_current_visuals(), epoch, ncols=1, save_result=save_result)
            save_result = False
            if total_steps % (opt.display_freq*5):
                model.sample_attnMap(epoch)
        
        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
                
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')
            
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)
        
    if epoch > opt.niter:
        lr -= opt.lr / opt.niter_decay
        model.update_lr(lr)
        
                