import options as opt
import os
import cv2
from data import MyDataset
from cvtransforms import *
from torch.utils.data import Dataset, DataLoader

model = lipreading("backendGRU")
model = model.cuda()
if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

img_folder = '/home/yangshuang/luomingshuang/seq2seq_baidu/test'
img_files = os.listdir(img_folder)
img_files = sorted(img_files, key=lambda x:int(str(x.split('_')[0])))
images = [cv2.imread(os.path.join(img_folder, file)) for file in img_files]
images = [cv2.resize(img, (96, 96)) for img in images]
print(len(images))
for i in range(75-len(images)):
        images.append(np.zeros((96, 96, 3), dtype=np.uint8))
images = np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                for img in images], axis=0) / 255.0

images = CenterCrop(images, (88, 88))
images = ColorNormalize(images)
input = torch.FloatTensor(images[np.newaxis,...])
txt = 'li shi ji lu'
label = MyDataset.text2tensor(txt, 200, SOS=True)
label = torch.LongTensor(label)
print(input.size())
print(label)
data = {'encoder_tensor':input, 'decoder_tensor':label}
# label = label.unsqueeze(0)
# input = torch.randn(1,75,88,88)
input = input.unsqueeze(0)
# for idx,batch in enumerate(loader):
#     outputs = model(batch['encoder_tensor'])
outputs = model(input.cuda())
predict_txt = MyDataset.tensor2text(outputs.argmax(-1))
print(predict_txt)
