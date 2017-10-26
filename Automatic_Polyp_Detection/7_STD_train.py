# laod 1st saliency map and calculaye it's mean
# all superpixel's value top 20% and bottom 25% are set to postive and negative, respectively
# selected superpixels are listed to samples.
# all samples's features are used to MKBoosting.
# train MKB. and make strong classifier
training_set = [data, label]
