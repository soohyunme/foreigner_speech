#-*- coding: utf-8 -*-

import os, sys
import json
import pdb
import argparse
import time
import yaml
import numpy
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import editdistance
import pickle
import librosa
from flask import Flask, request
from tqdm import tqdm

## ===================================================================
## Load labels
## ===================================================================

def load_label_json(labels_path):
    with open(labels_path, encoding="utf-8") as label_file:
        labels = json.load(label_file)
        char2index = dict()
        index2char = dict()

        for index, char in enumerate(labels):
            char2index[char] = index
            index2char[index] = char
            
        return char2index, index2char

## ===================================================================
## Data loader
## ===================================================================

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, data_path, max_length, char2index):
        super(SpeechDataset, self).__init__()

        # load data from JSON
        with open(data_list,'r') as f:
            data = json.load(f)

        # convert seconds to frames
        max_length *= 16000

        # sort data in length order and filter data less than max_length
        data = sorted(data, key=lambda d: len(d['transcription']['ReadingLabelText']), reverse=True)
        self.data = [x for x in data if len(x['transcription']['ReadingLabelText']) <= max_length]
        
        self.dataset_path   = data_path
        self.char2index     = char2index

    def __getitem__(self, index):

        # read audio using librosa.load
        audio, sample_rate = librosa.load(os.path.join(self.dataset_path, self.data[index]['fileName']), mono=True)

        # read transcript and convert to indices
        transcript = self.data[index]['transcription']['ReadingLabelText']
        transcript = self.parse_transcript(transcript)
        
        return torch.FloatTensor(audio), torch.LongTensor(transcript)

    def parse_transcript(self, transcript):
        transcript = list(filter(None, [self.char2index.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return len(self.data)


## ===================================================================
## Define collate function
## ===================================================================

def pad_collate(batch):
    (xx, yy) = zip(*batch)

    ## compute lengths of each item in xx and yy
    x_lens = [len(x) for x in xx] 
    y_lens = [len(y) for y in yy] 

    ## zero-pad to the longest length
    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0) 
    yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0) 

    return xx_pad, yy_pad, x_lens, y_lens

## ===================================================================
## Baseline speech recognition model
## ===================================================================

class SpeechRecognitionModel(nn.Module):

    def __init__(self, n_classes=11):
        super(SpeechRecognitionModel, self).__init__()
        
        cnns = [nn.Dropout(0.1),  
                nn.Conv1d(40,64,3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),  
                nn.Conv1d(64,64,3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU()] 

        for i in range(2):
          cnns += [nn.Dropout(0.1),  
                   nn.Conv1d(64,64, 3, stride=1, padding=1),
                   nn.BatchNorm1d(64),
                   nn.ReLU()]

        ## define CNN layers
        self.cnns = nn.Sequential(*nn.ModuleList(cnns))

        ## define RNN layers as self.lstm - use a 3-layer bidirectional LSTM with 128 output size and 0.1 dropout
        self.lstm = nn.LSTM(64, 256, 3, batch_first=True, bidirectional=True, dropout=0.1)

        ## define the fully connected layer
        self.classifier = nn.Linear(512,n_classes)

        self.preprocess   = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)
        self.instancenorm = nn.InstanceNorm1d(40)

    def forward(self, x):

        ## compute MFCC and perform mean variance normalisation
        with torch.no_grad():
          x = self.preprocess(x)+1e-6
          x = self.instancenorm(x).detach()

        ## pass the network through the CNN layers
        x = self.cnns(x)

        ## pass the network through the RNN layers - check the input dimensions of nn.LSTM()
        x = self.lstm(x.transpose(1,2))[0]

        ## pass the network through the classifier
        x = self.classifier(x)

        return x

## ===================================================================
## Greedy CTC Decoder
## ===================================================================

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, blank=0):
        super().__init__()
        self.blank = blank
        
    def forward(self, emission: torch.Tensor):
        """
        Given a sequence emission over labels, get the best path.
        """
        indices = torch.argmax(emission, dim=-1)

        indices = torch.unique_consecutive(indices, dim=-1)
        
        indices = numpy.array(indices)
        
        indices = [x for x in indices if x != self.blank] if indices.shape != () else [] if indices == self.blank else indices

        return indices

## ===================================================================
## Define sampler 
## ===================================================================

class BucketingSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):

        # Shuffle bins in random order
        np.random.shuffle(self.bins)

        # For each bin
        for ids in self.bins:
            # Shuffle indices in random order
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

## ===================================================================
## Train an epoch on GPU
## ===================================================================

def process_epoch(model,loader,criterion,optimizer,scheduler,trainmode=True):

    # Set the model to training or eval mode
    if trainmode:
        model.train()
    else:
        model.eval()

    ep_loss = 0
    ep_cnt  = 0

    with tqdm(loader, unit="batch") as tepoch:

        for data in tepoch:

            ## Load x and y
            x = data[0].cuda()
            y = data[1].cuda()
            y_len = torch.LongTensor(data[3])

            output = model(x)

            output = torch.nn.functional.log_softmax(output, dim=2)

            output = output.transpose(0,1)

            ## compute the loss using the CTC objective
            x_len = torch.LongTensor([output.size(0)]).repeat(output.size(1))
            loss = criterion(output, y, x_len, y_len)

            if trainmode:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # keep running average of loss
            ep_loss += loss.detach() * len(x)
            ep_cnt  += len(x)

            # print value to TQDM
            tepoch.set_postfix(loss=ep_loss.item()/ep_cnt)
    if trainmode:
        scheduler.step()
    return ep_loss.item()/ep_cnt

## ===================================================================
## Evaluation script
## ===================================================================

def process_eval(model,data_path,data_list,index2char,save_path=None):

    # set model to evaluation mode
    model.eval()

    # initialise the greedy decoder
    greedy_decoder = GreedyCTCDecoder(blank=len(index2char))

    # load data from JSON
    with open(data_list,'r') as f:
        data = json.load(f)

    results = []

    for file in tqdm(data):
        # read the wav file and convert to PyTorch format
        audio, sample_rate = librosa.load(os.path.join(data_path, file['file']), mono=True)
        x = torch.FloatTensor(audio).unsqueeze(dim=0).cuda()
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
        
        # forward pass through the model
        
        output = model(x)
        output = torch.nn.functional.log_softmax(output, dim=-1)
        pred = output.transpose(0,1)

        # decode using the greedy decoder
        pred = greedy_decoder(output.cpu().detach().squeeze())

        # convert to text
        out_text = ''.join([index2char[x] for x in pred])

        # keep log of the results
        file['pred'] = out_text
        if 'text' in file:
            file['edit_dist']   = editdistance.eval(out_text.replace(' ',''),file['text'].replace(' ',''))
            file['gt_len']     = len(file['text'].replace(' ',''))
        results.append(file)
    
    # save results to json file
    with open(os.path.join(save_path,'results.json'), 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=2)

    # print CER if there is ground truth
    if 'text' in file:
        cer = sum([x['edit_dist'] for x in results]) / sum([x['gt_len'] for x in results])
        print('Character Error Rate is {:.2f}%'.format(cer*100))

## ===================================================================
## Deploy server script
## ===================================================================

def deploy_server(model,index2char,port):

    # initialise the greedy decoder
    greedy_decoder = GreedyCTCDecoder(blank=len(index2char))

    # create the Flask app
    app = Flask(__name__)

    @app.route('/query-window', methods=['POST'])
    def process_chunk():

        # unpack the received data
        data = pickle.loads(request.get_data())
        data = data.unsqueeze(dim=0).cuda()

        # convert to PyTorch format
        x = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
        
        # forward pass through the model
        output = model(x)
        output = torch.nn.functional.log_softmax(output, dim=-1)
        pred = output.transpose(0,1)

        # decode using the greedy decoder
        pred = greedy_decoder(output.cpu().detach().squeeze())

        # join the index
        out_text = ''.join([index2char[x] for x in pred])

        print('Result:',out_text)

        return out_text

    app.run(host='0.0.0.0', debug=True, port=port, threaded=False)

## ===================================================================
## Main execution script
## ===================================================================

def main():

    parser = argparse.ArgumentParser(description='Korean Speech Recognition Project')

    parser.add_argument('--config',      type=str,   default=None,   help='Config YAML file');

    ## related to data loading
    parser.add_argument('--max_length',  type=int, default=10,   help='maximum length of audio file in seconds')
    parser.add_argument('--train_list',  type=str, default='')
    parser.add_argument('--val_list',    type=str, default='')
    parser.add_argument('--train_path',  type=str, default='')
    parser.add_argument('--val_path',    type=str, default='')
    parser.add_argument('--labels_path', type=str, default='')


    ## related to training
    parser.add_argument('--max_epoch',    type=int,   default=10,       help='number of epochs during training')
    parser.add_argument('--batch_size',   type=int,   default=128,      help='batch size')
    parser.add_argument('--weight_decay', type=float, default=0,        help='weight decay')
    parser.add_argument('--lr_decay',     type=float, default=0,        help='learning rate decay')
    parser.add_argument('--lr',           type=float, default=2e-2,     help='learning rate')
    parser.add_argument('--seed',         type=int,   default=2222,     help='random seed initialisation')
    
    ## relating to loading and saving
    parser.add_argument('--initial_model',  type=str, default='',   help='load initial model, e.g. for finetuning')
    parser.add_argument('--save_path',      type=str, default='./result',   help='location to save checkpoints')

    ## related to inference and deploying server
    parser.add_argument('--eval',       dest='eval',     action='store_true', help='Evaluation mode')
    parser.add_argument('--parallel',   dest='parallel', action='store_true', help='Parallel mode')
    parser.add_argument('--server',     dest='server',   action='store_true', help='Server mode')
    parser.add_argument('--port',       type=int,        default=10000,       help='Port for the server')

    args = parser.parse_args()

    ## Parse YAML
    def find_option_type(key, parser):
        for opt in parser._get_optional_actions():
            if ('--' + key) in opt.option_strings:
                return opt.type
        raise ValueError

    if args.config is not None:
        with open(args.config, "r") as f:
            yml_config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in yml_config.items():
            if k in args.__dict__:
                typ = find_option_type(k, parser)
                args.__dict__[k] = typ(v)
            else:
                sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))

    # load labels
    char2index, index2char = load_label_json(args.labels_path)

    ## make an instance of the model on GPU
    model = SpeechRecognitionModel(n_classes=len(char2index)+1).cuda()
    print('Model loaded. Number of parameters:',sum(p.numel() for p in model.parameters()))
    
    ## parallel
    if args.parallel:
        model = nn.DataParallel(model)

    ## load from initial model
    if args.initial_model != '':
        try:
            model.module.load_state_dict(torch.load(args.initial_model))
        except:
            model.load_state_dict(torch.load(args.initial_model))


    ## code for server
    if args.server:
        deploy_server(model,index2char,args.port)
        quit();

    ## make directory for saving models and output
    assert args.save_path != ''
    os.makedirs(args.save_path,exist_ok=True)

    ## code for inference - this uses val_path and val_list
    if args.eval:
        process_eval(model, args.val_path, args.val_list, index2char, save_path=args.save_path)
        quit();

    # initialise seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # define datasets
    trainset  = SpeechDataset(args.train_list, args.train_path, args.max_length, char2index)
    valset    = SpeechDataset(args.val_list,   args.val_path,   args.max_length, char2index)
 
    # initiate loader for each dataset with 'collate_fn' argument
    # do not use more than 6 workers
    trainloader = torch.utils.data.DataLoader(trainset, 
        batch_sampler=BucketingSampler(trainset, args.batch_size), 
        num_workers=6, 
        collate_fn=pad_collate)
    valloader   = torch.utils.data.DataLoader(valset,   
        batch_sampler=BucketingSampler(valset, args.batch_size), 
        num_workers=6, 
        collate_fn=pad_collate)
    
    ## define the optimizer with args.lr learning rate and appropriate weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 

    ## define the scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: args.lr_decay ** epoch if args.lr_decay else epoch,
                                        last_epoch=-1,
                                        verbose=False)
    
    ## set loss function with blank index
    criterion = nn.CTCLoss(blank=len(index2char)).cuda()

    ## initialise training log file
    f_log = open(os.path.join(args.save_path,'train.log'),'w')
    f_log.write('{}\n'.format(args))
    f_log.flush()

    ## Train for args.max_epoch epochs
    for epoch in range(0, args.max_epoch):

        tloss = process_epoch(model, trainloader, criterion, optimizer, scheduler, trainmode=True)
        vloss = process_epoch(model, valloader, criterion, optimizer, scheduler, trainmode=False)

        if (epoch + 1) % 5 == 0:
            # save checkpoint to file
            save_file = '{}/model{:05d}.pt'.format(args.save_path,epoch)
            print('Saving model {}'.format(save_file))
            process_eval(model, args.val_path, args.val_list, index2char, save_path=args.save_path)
            try:
                torch.save(model.module.state_dict(), save_file)
            except:
                torch.save(model.state_dict(), save_file)
            

        # write training progress to log
        f_log.write('Epoch {:03d}, train sloss {:.3f}, val loss {:.3f}\n'.format(epoch, tloss, vloss))
        f_log.flush()

    f_log.close()


if __name__ == "__main__":                          
    main()
