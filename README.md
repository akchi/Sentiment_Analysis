# Sentiment Analysis
Supervised classification of textual reviews based on its sentiment into one of the five polarities:
1. Strong negative
2. Weak negative
3. Neutral
4. Weak Positive
5. Strong Positive

## Methodology
<ol>
	<li> <b>Text Pre-processing</b>: The raw data was processed to convert it into a format that can be used for further processing. The following steps were applied:
		<ul>
			<li>Case normalisation</li>
			<li>Tokenisation</li>
			<li>Lemmitization</li>
		</ul>
	</li>
	<li> <b>Feature Generation</b>: Once the data was cleansed, relevant features were extracted from the it such as:
		<ul>
			<li> Creation of N-grams</li>
			<li> Term and inverse document frequency</li>
		</ul>
	</li>
	<li> <b>Model</b> : Logistic regression is the classifier used for determining the polarity of a review.</li>
		
</ol>

<b>Datasets:</b>

1.	train_data.csv:

	The training set consists of 650,000 product reviews. 

2.	train_label.csv:

	This dataset contains the sentiment lables of the training dataset. The label set 
	(1,2,3,4,5) refer to five polarity levels (strong negative, weak negative, neutral, 
	weak positive, strong and positive) respectively.

3.	test_data.csv:

	The test set consists of 50,000 product reviews. 

4.	predicted_label.csv:

	This dataset contains the predicted sentiment labels of the test data.
