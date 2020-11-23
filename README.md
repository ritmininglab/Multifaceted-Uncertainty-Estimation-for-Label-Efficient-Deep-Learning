ADL.py:
This script contains the functions to run the proposed method.

	1)ADL()
	This function conducts active learning with the ADL active learner.
	Parameters:
		X:Training data. X should be given as N by M numpy array. where N is the number of instances, M is the number of features.
		Y:One-hot coded labels. Y should be give as N by K numpy array. Where K is the number of classes.
		trainInd,testInd,candInd: train/test/candidate index. The indecies should be given as one dimension numpy array or list. 
		sample_method : The form of the active sampling function. Option 'DPlusV' initiates the proposed dissonance and vacuity sampling. For other options, please refer to the in-line comments.
		AL_iter: Number of active learning iterations.
		epoch : epoch number.
		network: Customized NN structure. Option 'mine2' refers to the network structure proposed in the paper. For other options'LeNet_Reg_EDL','LeNetSoftMax','LeNet_EDL','simple_EDL', and 'simple_softmax', please check their structure from the code.
		activation :activation function for NN.
		retrain: whether re-initialize the network or not for every AL iteration.
		vacDecay: the decay rate of vacuity. 
	
	2)OODIdentify()
	This function identifies the anchor set as a part of the training loss in the proposed method. When ALD() runs with the sample_method='DPlusV' this method would be called automatically.  
	Parameters:
		X:Training data. X should be given as N by M numpy array. where N is the number of instances, M is the number of features.
		trainIndex/candidateIndex:train/candidate index. The indecies should be given as one dimension numpy array or list. 
		type: Method used to identify the OODs. option='kernelDensity2' are used for OOD identification in the paper. Other options are mainly used for comparasion purposes.
		top: The balancing coefficient for train-train kernel distance and train-candidate kernel distance. The default value is used in the paper.
		
	3)Dissonance()
	This function computes the dissonance of a multi-nomial opinion. 
	Parameters:
		alpha:The multi-nomial opinion. Alpha should be given as K by one numpy array or list.
		
	4)myopicExaustedTest()
	This function finds the optimal sequence of candidate by evaluating the model improvement for every candidate samples for each annotation iteration and find the myopic optimal sample.
	The function can be used to output the upperbound AL curve for a specific dataset. (Notice: This function is extremely slow to run.)
	
