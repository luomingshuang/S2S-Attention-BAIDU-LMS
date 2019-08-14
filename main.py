import torch
from torch.utils.data import Dataset, DataLoader
import tensorboardX
from data import MyDataset
from c3d_resnet_s2s import lipreading
import torch.nn as nn
from torch import optim
import os
import time

import tensorflow as tf

if(__name__ == '__main__'):
    torch.manual_seed(55)
    torch.cuda.manual_seed_all(55)
    opt = __import__('options')
    
def data_from_opt(txt_file, phase):
    dataset = MyDataset(img_root=opt.img_root, 
        annotation_root=opt.annotation_root,
        txt_file=txt_file,
        phase=phase,
        img_padding=opt.img_padding,
        text_padding=opt.text_padding)
    print('txt_file:{},num_data:{}'.format(txt_file,len(dataset.data)))
    loader = DataLoader(dataset, 
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        drop_last=False,
        shuffle=True)   
    return (dataset, loader)


if(__name__ == '__main__'):
    model = lipreading(mode=opt.mode, nClasses=30).cuda()
    
    writer_1 = tf.summary.FileWriter("./logs/plot_1")
    log_var = tf.Variable(0.0)
    tf.summary.scalar("train_loss", log_var)
    writer_op = tf.summary.merge_all()

    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    # optimizer = optim.Adam(model.parameters(),
    #         lr=4e-4,
    #         weight_decay=5e-5)

    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
    #         step_size=500,
    #         gamma=0.8)



    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)        
            
    (train_dataset, train_loader) = data_from_opt(opt.trn_txt, 'train')
    (tst_dataset, tst_loader) = data_from_opt(opt.tst_txt, 'test')
    
    criterion = nn.NLLLoss() 
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    iteration = 0
    for epoch in range(opt.max_epoch):
        start_time = time.time()
        for (i, batch) in enumerate(train_loader):
            (encoder_tensor, decoder_tensor) = batch['encoder_tensor'].cuda(), batch['decoder_tensor'].cuda()
            #print('decoder_tensor size is:', decoder_tensor.size())
            outputs = model(encoder_tensor, decoder_tensor, opt.teacher_forcing_ratio)            
            #print('outputs size is:',outputs.size())
            flatten_outputs = outputs.view(-1, outputs.size(2))
            #print('flatten_outputs size is:', flatten_outputs.size())
            #print('decoder_tensor size is:', decoder_tensor.view(-1).size())
            loss = criterion(flatten_outputs, decoder_tensor.view(-1))
            optimizer.zero_grad()                
            
            summary = session.run(writer_op, {log_var: loss.detach().cpu().numpy()})
            writer_1.add_summary(summary, iteration)
            writer_1.flush()

            iteration += 1 

            loss.backward()
            optimizer.step()
            tot_iter = epoch*len(train_loader)+i
            
            train_loss = loss.item()
            # if(i % opt.display == 0):
            #     speed = (time.time()-start_time)/(i+1)
            #     eta = speed*(len(train_loader)-i)
            #     print('tot_iter:{},loss:{},eta:{}'.format(tot_iter,loss,eta/3600.0))


            print('iteration:%d, epoch:%d, train_loss:%.6f'%(iteration, epoch, train_loss))
                
            #if(iteration % 10 == 0):
            if (iteration == 1):
                with torch.no_grad():
                    predict_txt_total = []
                    truth_txt_total = []
                    for idx,batch in enumerate(tst_loader):
                        (encoder_tensor, decoder_tensor) \
                            = batch['encoder_tensor'].cuda(), batch['decoder_tensor'].cuda()
                        outputs = model(encoder_tensor)
                        # print('infer outputs size is:', outputs.size())
                        # print('outputs.argmax(-1) size is:', outputs.argmax(-1).size())>>>(16,75)
                        predict_txt = MyDataset.tensor2text(outputs.argmax(-1))
                        truth_txt = MyDataset.tensor2text(decoder_tensor)
                        # print('predict_txt is:', predict_txt)
                        predict_txt_total.extend(predict_txt)
                        # print('predict_txt_total is:', predict_txt_total)
                        truth_txt_total.extend(truth_txt)
                
                print(''.join(101*'-'))                
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-')) 
                
                for (predict, truth) in list(zip(predict_txt_total, truth_txt_total)):
                    if predict != truth:
                        print('{:<50}|{:>50}'.format(predict, truth))                
                
                print(''.join(101 *'-'))
                wer = MyDataset.wer(predict_txt_total, truth_txt_total)
                cer = MyDataset.cer(predict_txt_total, truth_txt_total)                
                print('cer:{}, wer:{}'.format(cer, wer))          
                print(''.join(101*'-'))

                savename = os.path.join(opt.save_dir, 'me_cer_{}_wer_{}.pt'.format(cer, wer))
                savepath = os.path.split(savename)[0]
                if(not os.path.exists(savepath)): os.makedirs(savepath)
                torch.save(model.state_dict(), savename)
                break

