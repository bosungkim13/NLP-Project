import math
import random
import numpy
from collections import *

#################################################

class HMM:
    """
    Simple class to represent a Hidden Markov Model.
    """
    def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
        self.order = order
        self.initial_distribution = initial_distribution
        self.emission_matrix = emission_matrix
        self.transition_matrix = transition_matrix

def read_pos_file(filename):
    """
    Parses an input tagged text file.
    Input:
    filename --- the file to parse
    Returns:
    The file represented as a list of tuples, where each tuple
    is of the form (word, POS-tag).
    A list of unique words found in the file.
    A list of unique POS tags found in the file.
    """
    file_representation = []
    unique_words = set()
    unique_tags = set()
    f = open(str(filename), "r")
    for line in f:
        if len(line) < 2 or len(line.split("/")) != 2:
            continue
        word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
        tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
        file_representation.append( (word, tag) )
        unique_words.add(word)
        unique_tags.add(tag)
    f.close()
    return file_representation, unique_words, unique_tags

def bigram_viterbi(hmm, sentence):
    """
    Run the Viterbi algorithm to tag a sentence assuming a bigram HMM model.
    Inputs:
      hmm --- the HMM to use to predict the POS of the words in the sentence.
      sentence ---  a list of words.
    Returns:
      A list of tuples where each tuple contains a word in the
      sentence and its predicted corresponding POS.
    """

    # Initialization
    viterbi = defaultdict(lambda: defaultdict(int))
    backpointer = defaultdict(lambda: defaultdict(int))
    unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
    for tag in unique_tags:
    	if (hmm.initial_distribution[tag] != 0) and (hmm.emission_matrix[tag][sentence[0]] != 0):
    		viterbi[tag][0] = math.log(hmm.initial_distribution[tag]) + math.log(hmm.emission_matrix[tag][sentence[0]])
    	else:
        	viterbi[tag][0] = -1 * float('inf')

    # Dynamic programming.
    for t in range(1, len(sentence)):
        backpointer["No_Path"][t] = "No_Path"
        for s in unique_tags:
            max_value = -1 * float('inf')
            max_state = None
            for s_prime in unique_tags:
                val1= viterbi[s_prime][t-1]
                val2 = -1 * float('inf')
                if hmm.transition_matrix[s_prime][s] != 0:
                    val2 = math.log(hmm.transition_matrix[s_prime][s])
                curr_value = val1 + val2
                if curr_value > max_value:
                    max_value = curr_value
                    max_state = s_prime
            val3 = -1 * float('inf')
            if hmm.emission_matrix[s][sentence[t]] != 0:
                val3 = math.log(hmm.emission_matrix[s][sentence[t]])
            viterbi[s][t] = max_value + val3
            if max_state == None:
                backpointer[s][t] = "No_Path"
            else:
                backpointer[s][t] = max_state
    for ut in unique_tags:
        string = ""
        for i in range(0, len(sentence)):
            if (viterbi[ut][i] != float("-inf")):
                string += str(int(viterbi[ut][i])) + "\t"
            else:
                string += str(viterbi[ut][i]) + "\t"

    # Termination
    max_value = -1 * float('inf')
    last_state = None
    final_time = len(sentence) - 1
    for s_prime in unique_tags:
        if viterbi[s_prime][final_time] > max_value:
            max_value = viterbi[s_prime][final_time]
            last_state = s_prime
    if last_state == None:
        last_state = "No_Path"

    # Traceback
    tagged_sentence = []
    tagged_sentence.append((sentence[len(sentence)-1], last_state))
    for i in range(len(sentence)-2, -1, -1):
        next_tag = tagged_sentence[-1][1]
        curr_tag = backpointer[next_tag][i+1]
        tagged_sentence.append((sentence[i], curr_tag))
    tagged_sentence.reverse()
    return tagged_sentence


#####################  STUDENT CODE BELOW THIS LINE  #####################

def compute_counts(training_data: list, order: int) -> tuple:
	"""
	Counts the number of tokens in training _data, times certain combiantions of states appear or a certain word 
	is tagged with a certain tag appears in the training data.
	Input:
	training_data -- A list of word tag tuples corresponding the training data
	order -- An integer corresponding to the order of the markov model
	
	Output:
	Returns a tuple with the corresponding counts. The first element is an integer that is the number of tokens in the training data,
			the second element is a dictionary that counts every unique tag and unique word, the third element is a dictionary that counts 
			the number of times a certain state appears, the fourth element is a dictionary that counts the number of times a certain tag sequence of length 2 appears.  
			For a third order model, there is a 4th element of the tuple that counts the number of times a certain tag sequence of length 3 appears.

	"""
	num_tokens = len(training_data)
	tagword_count = defaultdict(lambda: defaultdict(int))
	tag_count = defaultdict(int)
	state_count1 = defaultdict(lambda: defaultdict(int))
	state_count2 = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

	#set all values of state_count, tagword_count, and tag_count to 0 then increment every repeat
	for i in range(0, len(training_data)):
		if (i!=len(training_data)-1):
			state_count1[training_data[i][1]][training_data[i+1][1]] += 1
		tagword_count[training_data[i][1]][training_data[i][0]] += 1
		tag_count[training_data[i][1]] += 1
	#computes for third order model when order is 3
	if order == 3:
		for i in range(0, len(training_data)-2):
			state_count2[training_data[i][1]][training_data[i+1][1]][training_data[i+2][1]] += 1

		return (num_tokens, tagword_count, tag_count, state_count1, state_count2)

	return (num_tokens, tagword_count, tag_count, state_count1)

def compute_initial_distribution(training_data: list, order: int) -> dict:
	"""
	Counts the number of times a certain state is the initial state in the training data.
	Inputs:
	training_data -- A list of word tag tuples corresponding the training data
	order -- An integer corresponding to the order of the markov model

	Output:
	Returns a dictionary that is the initial distributions
	"""

	#initializing initial distribution dictionary
	distribution1 = defaultdict(int)
	distribution2 = defaultdict(lambda: defaultdict(int))
	#counting the first initial state of the training data
	if len(training_data) > 0:
		distribution1[training_data[0][1]] = 1
		count1 = 1
		if order == 3 and len(training_data) > 1:
			distribution2[training_data[0][1]][training_data[1][1]] = 1
			count2 = 1
	#order 2
	if order == 2: 	
		#counting number of times each state is the initial state
		for i in range(0, len(training_data)-1):
			if training_data[i][1] == '.':
				distribution1[training_data[i+1][1]] += 1
				count1 += 1
		#calculating probabilities
		for key1 in distribution1:
			distribution1[key1] = distribution1[key1]/count1
		return distribution1

	#order 3
	if order == 3:
		#counting number of times each state is the initial state
		for i in range(0, len(training_data)-2):
			if training_data[i][1] == '.':
				distribution2[training_data[i+1][1]][training_data[i+2][1]] += 1
				count2 += 1		
		#calculating probabilities
		for key1 in distribution2:
			for key2 in distribution2[key1]:
				distribution2[key1][key2] = distribution2[key1][key2]/count2
		return distribution2

def compute_emission_probabilities(unique_words: list, unique_tags: list, W: dict, C: dict) -> dict:
	"""
	Computes the emission probabilities 
	Inputs:
	unique_words -- A list that contains the unique words of the training data
	unique_tags -- A list that contains the unique tags of the training data
	W -- A dictionary that counts the number of times a word appears with a certain tag in the training data
	C --  A dictionary that counts the number of times a tag appears in the training data

	Output:
	emission_prob -- A dictionary that maps each state to a dictionary that maps the words that appear
					in each state to their respective probabilities.
	"""
	#initializing emission probability dictionary
	emission_prob = defaultdict(lambda: defaultdict(int))
	#calculate probabilities
	for tag in W:
		for word in W[tag]:
			emission_prob[tag][word] = float(W[tag][word]/C[tag])
	#if word does not appear, then give it zero probability
	for tag in unique_tags:
		for word in unique_words:
			if emission_prob[tag][word] == 0:
				emission_prob[tag][word] = 0
	return emission_prob

def compute_lambdas(unique_tags: list, num_tokens: int, C1: dict, C2: dict, C3: dict, order: int) -> list:
	"""
	Computes lamda values according to equation 8 and 9 in the HW document.
	Inputs:
	unique_tags -- A list that contains the unique tags of the training data
	num_tokens --  An integer that is the 
	C1 -- A dictionary with the count C(t_i)
	C2 -- A dictionary with the count C(t_i-1, t_i)
	C3 -- A dictionary with the count C(t_i-2, t_i-1, t_i)
	order -- An integer corresponding to the order of the markov model
	"""
	#order 2
	if order == 2:
		lst_lambda = [0, 0, 0]
		for tag1 in unique_tags:
			for tag2 in unique_tags:
				if C2[tag1][tag2] > 0:
					#if denominator is zero, set value to zero
					if num_tokens == 0:
						a0 = 0
					else:
						a0 = (C1[tag2]-1)/num_tokens
					#if denominator is zero, set value to zero
					if C1[tag1]-1 == 0:
						a1 = 0
					else:
						a1 = (C2[tag1][tag2]-1)/(C1[tag1]-1)
					#find maximum
					max_value = max(a0, a1)									
					if max_value == a0:
						i = 0						
					elif max_value == a1:
						i = 1	
					lst_lambda[i] = lst_lambda[i] + C2[tag1][tag2]

		lambda_sum = sum(lst_lambda)
		return [lst_lambda[0]/lambda_sum, lst_lambda[1]/lambda_sum, lst_lambda[2]/lambda_sum]		
	
	if order ==3:
		lst_lambda = [0, 0, 0]
		for tag1 in unique_tags:
			for tag2 in unique_tags:
				for tag3 in unique_tags:
					if C3[tag1][tag2][tag3] > 0:
						#if denominator is zero, set value to zero
						if num_tokens == 0:
							a0 = 0
						else:
							a0 = (C1[tag3]-1)/num_tokens
						#if denominator is zero, set value to zero
						if C1[tag2]-1 == 0:
							a1 = 0
						else:
							a1 = (C2[tag2][tag3]-1)/(C1[tag2]-1)
						#if denominator is zero, set value to zero
						if C2[tag1][tag2]-1 == 0:
							a2 = 0
						else:
							a2 = (C3[tag1][tag2][tag3]-1)/(C2[tag1][tag2]-1)
						#find maximum
						i = numpy.argmax([a0, a1, a2])
						lst_lambda[i] = lst_lambda[i] + C3[tag1][tag2][tag3]
		lambda_sum = sum(lst_lambda)
		return [lst_lambda[0]/lambda_sum, lst_lambda[1]/lambda_sum, lst_lambda[2]/lambda_sum]				

def update_HMM(HMM, training_unique, test_unique):
    """
    Updates the emission matrix of HMM so unknown words have a small probability of being emitted
    Inputs:
    HMM -- A HMM object
    training_unique -- A set that contains the unique words of the training data
    test_unique -- A set that contains the unique words of the test data

    Output:
    None
    """
    #initializing epsilon as the probability of emitting an unknown word 
    for word in test_unique:
        for tag in HMM.emission_matrix.keys():
            if word in training_unique:
                continue
            else:
                HMM.emission_matrix[tag][word] = 0.00001
    #adding epsilon to the probabilities that were non zero before
    for tag in HMM.emission_matrix.keys():
        for word in training_unique:
            if HMM.emission_matrix[tag][word] != 0:
                HMM.emission_matrix[tag][word] += 0.00001
    #normalizing probablilities
    for tag in HMM.emission_matrix.keys():
        sums = sum(HMM.emission_matrix[tag].values())
        for word in HMM.emission_matrix[tag]:
            HMM.emission_matrix[tag][word] = HMM.emission_matrix[tag][word] / sums

def build_hmm(training_data: list, unique_tags: list, unique_words: list, order: int, use_smoothing: bool):
	"""
	Computes and returns a hidden markov model

	Inputs:
	training_data -- A list of word and tag tuples that correspond to the training data
	unique_tags -- A list of unique tags in the given training data
	unique_words -- A list of unique words in the given training data
	order -- An integer representing the order of the hmm
	use_smoothing -- A boolean to determine the use of smoothing

	Output:
	model -- A hidden markov model object
	"""
	#computing counts
	counts = compute_counts(training_data, order)
	#computing the intitial distribution
	initial_dist = compute_initial_distribution(training_data, order)
	#computing the emission porbabilities
	emission_prob = compute_emission_probabilities(unique_words, unique_tags, counts[1], counts[2])

	#determine lambda values
	if use_smoothing:
		if order ==2 :
			lst_lambdas	= compute_lambdas(unique_tags, counts[0], counts[2], counts[3], {}, order)
		if order ==3 :
			lst_lambdas	= compute_lambdas(unique_tags, counts[0], counts[2], counts[3], counts[4], order)
	else:
		if order == 2:
			lst_lambdas = [0, 1, 0]
		if order ==3:
			lst_lambdas = [0, 0, 1]

	#calculate the transmission matrix for order 2 according to equation provided
	if order == 2:
		transition_mat = defaultdict(lambda: defaultdict(int))
		for tag2 in counts[3].keys():
			for tag1 in counts[3][tag2].keys():
				if counts[2][tag2] == 0 or counts[0] == 0:
					transition_mat[tag2][tag1] = 0
				else:
					transition_mat[tag2][tag1] = lst_lambdas[1]*(counts[3][tag2][tag1]/counts[2][tag2])+lst_lambdas[0]*(counts[2][tag1]/counts[0])
	#calculate the transmission matrix for order 3 according to equation provided
	if order == 3:
		transition_mat = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
		for tag3 in counts[4].keys():
			for tag2 in counts[4][tag3].keys():
				for tag1 in counts[4][tag3][tag2].keys():
					if counts[3][tag3][tag2] == 0 or counts[2][tag2] == 0:
						transition_mat[tag3][tag2][tag1] = 0
					else:
						transition_mat[tag3][tag2][tag1] = lst_lambdas[2]*(counts[4][tag3][tag2][tag1]/counts[3][tag3][tag2])+lst_lambdas[1]*(counts[3][tag2][tag1]/counts[2][tag2])+lst_lambdas[0]*(counts[2][tag1]/counts[0])



	model = HMM(order, initial_dist, emission_prob, transition_mat)
	
	return model

def trigram_viterbi(hmm, sentence: list) -> list:
	"""
	Run the Viterbi algorithm to tag a sentence assuming a trigram HMM model.
	Inputs:
		hmm -- the HMM to use to predict the POS of the words in the sentence.
		sentence --  a list of words.
	Returns:
		A list of tuples where each tuple contains a word in the
		sentence and its predicted corresponding POS.
	"""
	#initializing back pointer and viterbi dictionaries
	viterbi = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
	backpointer = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
	unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
	for tag1 in unique_tags:
		for tag2 in unique_tags:
			if (hmm.initial_distribution[tag1][tag2] != 0) and (hmm.emission_matrix[tag1][sentence[0]] != 0) and (hmm.emission_matrix[tag2][sentence[1]] != 0):
				viterbi[tag1][tag2][1] = math.log(hmm.initial_distribution[tag1][tag2]) + math.log(hmm.emission_matrix[tag1][sentence[0]]) + math.log(hmm.emission_matrix[tag2][sentence[1]])
			else:
				viterbi[tag1][tag2][1] = -1 * float('inf')
	 

	#dynamic programming
	for t in range(2, len(sentence)):
		backpointer["No_Path"]["No_Path"][t] = "No_Path"
		for s in unique_tags:
			for s_p in unique_tags:
				max_value = -1 * float('inf')
				max_state = None
				for s_pp in unique_tags:
					val1= viterbi[s_pp][s_p][t-1]
					val2 = -1 * float('inf')
					if hmm.transition_matrix[s_pp][s_p][s] != 0:
						val2 = math.log(hmm.transition_matrix[s_pp][s_p][s])
					curr_value = val1 + val2				
					if curr_value > max_value:		
						max_value = curr_value
						max_state = s_pp
				val3 = -1 * float('inf')
				if hmm.emission_matrix[s][sentence[t]] != 0:
					val3 = math.log(hmm.emission_matrix[s][sentence[t]])
				viterbi[s_p][s][t] = max_value + val3
	
				if max_state == None:
					backpointer[s_p][s][t] = "No_Path"
				else:
					backpointer[s_p][s][t] = max_state					
	
	#termination
	max_value = -1 * float('inf')
	last_states = (None, None)
	final_time = len(sentence) - 1
	for s_p in unique_tags:
		for s_pp in unique_tags:
			if viterbi[s_pp][s_p][final_time] > max_value:
				max_value = viterbi[s_pp][s_p][final_time]
				last_states = (s_pp, s_p)
	if last_states == (None, None):
		last_states = ("No_Path", "No_Path")

	#traceback
	tagged_sentence = []
	tagged_sentence.append((sentence[len(sentence)-1], last_states[1]))
	tagged_sentence.append((sentence[len(sentence)-2], last_states[0]))
	for i in range(len(sentence)-3, -1, -1):
		next_tag = tagged_sentence[-1][1]
		nextnext_tag = tagged_sentence[-2][1]
		curr_tag = backpointer[next_tag][nextnext_tag][i+2]
		tagged_sentence.append((sentence[i], curr_tag))
	tagged_sentence.reverse()
	return tagged_sentence

def sentence_separater(test_data):
	"""
	Separates the test_data into sentences
	Input:
	test_data -- A list of tuples that contain a word tag pair that is the test data

	Output:
	Returns a tuple where the first element is a list of lists that contains the words of each sentence
				and the second element is a list of lists that contains the word tag pairs of each sentence
	"""

    list_of_sentences_words = []
    list_of_sentences_pairs = []

	#finding how many sentences are in the test data
    count = 0
    for (word, tag) in test_data:
        if tag == '.':
            count += 1
	#creating lists of lists for each sentence in the test data
    for ctr in range(count):
        list_of_sentences_words.append([])
        list_of_sentences_pairs.append([])

    #adding each sentence into the nested list
	count2 = 0
    for (word, tag) in test_data:
        if tag != '.':
            list_of_sentences_words[count2].append(word)
            list_of_sentences_pairs[count2].append((word, tag))
        else:
            list_of_sentences_words[count2].append(word)
            list_of_sentences_pairs[count2].append((word, tag))
            count2 += 1
    
    return (list_of_sentences_words, list_of_sentences_pairs)

##################################################

#reading training data
training_tuple = read_pos_file('training.txt')
training_data = training_tuple[0]
training_words = training_tuple[1]
training_tags = training_tuple[2]

#generating training data for the first 1%, 5%, 10%, 25%, 50%, and 100% of the training data
training1 = training_data[:int(0.01*len(training_data))+1]
training5 = training_data[:int(0.05*len(training_data))+1]
training10 = training_data[:int(0.10*len(training_data))+1]
training25 = training_data[:int(0.25*len(training_data))+1]
training50 = training_data[:int(0.50*len(training_data))+1]
training75 = training_data[:int(0.75*len(training_data))+1]
training100 = training_data

#creating list that contains each training data that will be sued to create HMM
training_lst = [training1, training5, training10, training25, training50, training75, training100]

#creating set that contains all the unique tags and unique words in each training data
training_words = []
training_tags = []
for trial in training_lst:
	word_set = set()
	tag_set = set()
	for (word, tag) in trial:
		word_set.add(word)
		tag_set.add(tag)
	training_words.append(word_set)
	training_tags.append(tag_set)

#reading test data
test_tuple = read_pos_file('testdata_tagged.txt')
tagtest_data = test_tuple[0]
test_words = test_tuple[1]
test_tags = test_tuple[2]
sentences = sentence_separater(tagtest_data)
word_sentence = sentences[0]
pair_sentence = sentences[1]

#experiment 1
experiment1 = []
for i in range(len(training_lst)):
	hmm1 = build_hmm(training_lst[i], training_tags[i], training_words[i], 2, False)
	train_unique = training_words[i]
	test_unique = test_words
	update_HMM(hmm1, train_unique, test_unique)
	predicted_lst = []
	for i in range(0, len(word_sentence)):
		predicted_sentence = bigram_viterbi(hmm1, word_sentence[i])
		predicted_lst.append(predicted_sentence)

	accuracy = 0
	for i in range(0,len(predicted_lst)):
		for j in range(0, len(predicted_lst[i])):
			if pair_sentence[i][j] == predicted_lst[i][j]:
				accuracy += 1
	experiment1.append(accuracy/len(tagtest_data))
	
print(experiment1)

#experiment 2
experiment2 = []
for i in range(len(training_lst)):
	hmm1 = build_hmm(training_lst[i], training_tags[i], training_words[i], 3, False)
	train_unique = training_words[i]
	test_unique = test_words
	update_HMM(hmm1, train_unique, test_unique)
	predicted_lst = []
	for i in range(0, len(word_sentence)):
		predicted_sentence = trigram_viterbi(hmm1, word_sentence[i])
		predicted_lst.append(predicted_sentence)

	accuracy = 0
	for i in range(0,len(predicted_lst)):
		for j in range(0, len(predicted_lst[i])):
			if pair_sentence[i][j] == predicted_lst[i][j]:
				accuracy += 1
	experiment2.append(accuracy/len(tagtest_data))
	
print(experiment2)

#experiment 3
experiment3 = []
for i in range(len(training_lst)):
	hmm1 = build_hmm(training_lst[i], training_tags[i], training_words[i], 2, True)
	train_unique = training_words[i]
	test_unique = test_words
	update_HMM(hmm1, train_unique, test_unique)
	predicted_lst = []
	for i in range(0, len(word_sentence)):
		predicted_sentence = bigram_viterbi(hmm1, word_sentence[i])
		predicted_lst.append(predicted_sentence)

	accuracy = 0
	for i in range(0,len(predicted_lst)):
		for j in range(0, len(predicted_lst[i])):
			if pair_sentence[i][j] == predicted_lst[i][j]:
				accuracy += 1
	experiment3.append(accuracy/len(tagtest_data))
	
print(experiment3)

#experiment 4
experiment4 = []
for i in range(len(training_lst)):
	hmm1 = build_hmm(training_lst[i], training_tags[i], training_words[i], 3, True)
	train_unique = training_words[i]
	test_unique = test_words
	update_HMM(hmm1, train_unique, test_unique)
	predicted_lst = []
	for i in range(0, len(word_sentence)):
		predicted_sentence = trigram_viterbi(hmm1, word_sentence[i])
		predicted_lst.append(predicted_sentence)

	accuracy = 0
	for i in range(0,len(predicted_lst)):
		for j in range(0, len(predicted_lst[i])):
			if pair_sentence[i][j] == predicted_lst[i][j]:
				accuracy += 1
	experiment4.append(accuracy/len(tagtest_data))
	
#print(experiment4)
