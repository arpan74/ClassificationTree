import math
# Iplementation of CART - Classification and Regression Trees by Leo Breiman

# Recursive Binary Splitting - Greedy Approach to split the space
# Regression - criteria for best split is based off Residual Sum of Squares. Pick a split, then the predictions for each of the new regions will be the mean
#   of that region. Calculate RSS. Find the split which minimizes RSS the most. 
# Classification - criteria for best split can be based off Gini Index or Entropy. These are both measures of node purity / entropy. 

# Gini Index implementation
def gini_index(groups, classes):
    #Count the number of total observations
    numInstances = float(sum([len(group) for group in groups]))

    if numInstances == 0:
        return 0.0

    gini = 0.0
    # Each group can be considered a region that is subdivided by the decision tree or as terminal node / leaf in CART
    for group in groups:
        size = float(len(group))
        proportion = size / numInstances
        if proportion == 0 :
            continue
        score = 0.0

        # Gini Index is calculated by summing for all classes
        #    proportion of observations of that class / total obs
        #    in that region
        for classVal in classes:
            specProp = [row[-1] for row in group].count(classVal) / size
            score += specProp * (1 - specProp)
        # weighing the score by the number of observations in that region vs num total obs
        gini += score * proportion
    return gini

# Shannon Entropy Implementation
def entropyCalc(groups, classes):
    numInstaces = float(sum( [len(group) for group in groups] ))

    if numInstaces == 0:
        return 0.0
    
    entropy = 0.0
    # For each region/terminal node in the decision tree.....
    for group in groups:
        size = float(len(group))
        proportion = size / numInstances

        if proportion == 0:
            continue
        
        score = 0.0

        for classVal in classes:
            specProp = [row[-1] for row in group].count(classVal) / size
            score += (-1.0) * specProp*( math.log(specProp) / math.log(2) ) # we use the proportions as a proxy for probabilities
        entropy += score
    return entropy

def test_split(index, value, dataset): # left is no, right is yes
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values) # Can also use entropy
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value to figure out when to stop
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count) # returns the class that appears the most often
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root
 
# Print a decision tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))