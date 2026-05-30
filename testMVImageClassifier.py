import os
import numpy as np
import torch
from torch.autograd import Variable
from data_utils import grayscale_img_load, listdir

def softmax(x, axis=1):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def testMVImageClassifier(dataset_path, model, info, pooling = 'mean', cuda=False, verbose=False):

    # save pytorch model to eval mode
    model.eval()
    if (cuda):
        model.cuda()
    
    test_err = 0
    count = 0
    
    # Track per-category accuracy
    category_correct = np.zeros(len(info['category_names']))
    category_total = np.zeros(len(info['category_names']))
    
    print("=>Testing... (Pooling: %s)"%(pooling))

    # for each category
    for idx, c in enumerate(info['category_names']):
        category_full_dir = os.path.join(dataset_path,c)
        shape_dirs        = listdir(category_full_dir)
        print('=>Loading shape data: %s'%(c))

        # for each shape
        for s in shape_dirs:
            if verbose: print('=>Loading shape data: %s %s'%(s, c))
            views = listdir(os.path.join(category_full_dir, s))
            scores = np.zeros((len(views),len(info['category_names'])))
            count  += 1
            category_total[idx] += 1

            # for each view
            for i, v in enumerate(views):
                image_full_filename = os.path.join(category_full_dir, s, v)
                if 'png' not in image_full_filename : continue
                if verbose: print(' => Loading image: %s ...'%image_full_filename)
                im  = grayscale_img_load(image_full_filename)/255.
                im -= info['data_mean']
                im  = Variable(torch.from_numpy(im.astype('float32')), requires_grad=False).unsqueeze(0)
                # get predicted scores for each view
                if (cuda):
                    im = im.cuda()
                    scores[i, :] = model(im).detach().cpu().numpy().squeeze()
                else:
                    scores[i, :] = model(im).detach().numpy().squeeze()

            ''' 
            Get category predictions per shape using view-pooling strategies
            '''
            if pooling == 'mean':
                
                # Mean view-pooling: average probabilities across all views
                # Apply softmax to convert scores to probabilities
                probabilities = softmax(scores, axis=1)
                avg_probabilities = np.mean(probabilities, axis=0)
                predicted_label = np.argmax(avg_probabilities)
            elif pooling == 'max':
                # Max view-pooling: take max probability per class across views
                # Apply softmax to convert scores to probabilities
                probabilities = softmax(scores, axis=1)
                max_probabilities = np.max(probabilities, axis=0)
                predicted_label = np.argmax(max_probabilities)
            else:
                raise ValueError("pooling must be 'mean' or 'max'")
            
            if predicted_label != idx:
                test_err += 1
            else:
                category_correct[idx] += 1

            if verbose: print('predicted label:  %s, ground-truth label: %s\n'%(info['category_names'][predicted_label] ,c))

    # Calculate and display per-category accuracy
    print('\n' + '='*60)
    print('Per-Category Accuracy (%s pooling):' % pooling)
    print('='*60)
    for idx, c in enumerate(info['category_names']):
        if category_total[idx] > 0:
            accuracy = (category_correct[idx] / category_total[idx]) * 100
            correct = int(category_correct[idx])
            total = int(category_total[idx])
            print('%-15s: %2d/%2d correct = %6.2f%%' % (c, correct, total, accuracy))
    print('='*60)
    
    test_err = test_err / count
    print('Overall Test Error: %.2f%% (Accuracy: %.2f%%)\n' % (test_err * 100, (1 - test_err) * 100))