# Multifaceted-Uncertainty-Estimation-for-Label-Efficient-Deep-Learning
Main function:
	ADL():active learning with the ALD active learner
	X:Training data
	Y:Labels
	trainInd,testInd,candInd: train/test/candidate index
	sample_method : methods for identify OOD set.
	AL_iter: Number of active learning iterations
	epoch : epoch for DNN
	network: Customized NN structure
	activation :activation function for NN
	retrain: whether re-initialize the network or not for every AL iteration.
	vacDecay: the decay rate of vacuity.
