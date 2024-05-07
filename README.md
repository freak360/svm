<h1>SVM Classification Project</h1>

<p>This project implements a simple Support Vector Machine (SVM) classifier from scratch using Python. The SVM is designed to classify data points into two categories based on their attributes. The project also includes visualization of the decision boundary and margins, which helps in understanding how the SVM classifies the data.</p>

<h2>Features</h2>

<ul>
  <li>SVM model implemented from scratch.</li>
  <li>Visualization of decision boundaries and margins.</li>
  <li>Dataset generation using <code>sklearn.datasets</code>.</li>
  <li>Accuracy calculation and performance assessment.</li>
</ul>

<h2>Requirements</h2>

<ul>
  <li>Python 3.6+</li>
  <li>NumPy</li>
  <li>Matplotlib</li>
  <li>scikit-learn</li>
</ul>

<h2>Installation</h2>

<p>First, ensure that you have Python installed on your system. If not, download and install Python from <a href="https://www.python.org/">python.org</a>.</p>

<p>Clone this repository to your local machine using:</p>

<pre>
<code>git clone https://github.com/freak360/svm.git
cd svm-classification
</code>
</pre>

<p>Install the required Python packages:</p>

<pre>
<code>pip install numpy matplotlib scikit-learn
</code>
</pre>

<h2>Usage</h2>

<p>To run the SVM classifier and visualize the results, execute the main script:</p>

<pre>
<code>app.py
</code>
</pre>

<p>This will:</p>
<ol>
  <li>Generate a dataset.</li>
  <li>Split the dataset into training and testing sets.</li>
  <li>Train the SVM model on the training data.</li>
  <li>Predict the labels for the testing set.</li>
  <li>Calculate and print the accuracy of the model.</li>
  <li>Visualize the decision boundary and margins.</li>
</ol>

<h2>Code Structure</h2>

<ul>
  <li><code>svm.py</code>: Contains the main implementation of the SVM model and the visualization code.</li>
</ul>

<h3>Key Functions</h3>

<ul>
  <li><code>fit</code>: Fits the SVM model to the training data.</li>
  <li><code>predict</code>: Predicts the class labels for new data.</li>
  <li><code>visualize_svm</code>: Visualizes the decision boundary and margins using matplotlib.</li>
</ul>

<h2>Contributing</h2>

<p>Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.</p>

<h2>License</h2>

<p>This project is open source and available under the <a href="LICENSE">MIT License</a>.</p>
