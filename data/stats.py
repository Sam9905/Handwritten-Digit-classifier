import numpy as np
import matplotlib.pyplot as plt
try:
   import cPickle as cPickle
except:
   import pickle as cPickle
   
def display_stats(training_data, sample_id):
    """
    Display Stats of the the dataset
    """
    total_samples = len(training_data[0])
    sample_image = training_data[0][sample_id]
    sample_label = training_data[1][sample_id]
    label_names = training_data[1]

    print('Total Samples: {}'.format(total_samples))
    print('Each Label Counts: {}'.format(dict(zip(*np.unique(training_data[1], return_counts=True)))))
    print('First 20 Labels: {}'.format(training_data[1][:20]))
    
    show_image = sample_image.reshape((28, 28))

    print('\nStats of Sample {0}/{1}:'.format(sample_id, total_samples))
    print('Image of Sample {0}/{1}:'.format(sample_id, total_samples))
    print('Image - Min Value: {} Max Value: {}'.format(show_image.min(), show_image.max()))
    print('Image - Shape: {}'.format(show_image.shape))
    #print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))
    display_image(training_data, sample_id)
# Function to unwrap our data set into training, validation and testing data
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo,encoding='latin1')
    fo.close()
    return list(dict)

# Function for displaying a training image by it's index in the MNIST set
def display_image(training_data, index):
    digit_label = training_data[1][index]
    # Reshape 784 array into 28x28 image
    digit_image = training_data[0][index].reshape([28,28])
    

    # Two subplots, unpack the axes array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.set_title('Gray Scale')
    ax1.imshow(digit_image, cmap='gray_r')
    ax2.set_title('Nipy Spectral')
    ax2.imshow(digit_image, cmap='nipy_spectral')
    plt.show()
    
    print('Training data, sample: %d,  Label: %d' % (index, digit_label))
