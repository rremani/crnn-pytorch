import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.new_crnn as crnn
import os

model_path = 'data/models/netCRNN_24_1000.pth'
img_path = 'test_images/test/'
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
img_list = os.listdir(img_path)
model = crnn.CRNN(32, 1,len(alphabet)+1, 256)
own = model.state_dict()
#print own.keys()
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)


w = torch.load(model_path)

#print w.keys()[0]
w1={}

for k,v in w.items():
	w1[str.split(k,'module.')[1]]=v


model.load_state_dict(w1)
for i in img_list:
	converter = utils.strLabelConverter(alphabet)
	print converter


	transformer = dataset.resizeNormalize((32, 32))
	image = Image.open(img_path+i).convert('L')
	image = transformer(image)
	if torch.cuda.is_available():
    		image = image.cuda()
	image = image.view(1, *image.size())
	image = Variable(image)


	model.eval()
	preds = model(image)
	#print preds
	_, preds = preds.max(2)
	preds = preds.transpose(1, 0).contiguous().view(-1)

	preds_size = Variable(torch.IntTensor([preds.size(0)]))
	raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
	sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
	print('%-20s => %-20s' % (raw_pred, sim_pred))

