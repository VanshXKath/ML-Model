This Python script demonstrates a basic machine learning workflow to predict student placements using logistic regression. It begins by importing necessary libraries and reading a dataset named `placement.csv`, which includes features like CGPA and IQ along with placement status. After cleaning the data by dropping irrelevant columns, it visualizes the relationship between CGPA, IQ, and placement through a scatter plot. The input features (CGPA and IQ) and output label (placement) are separated, and the data is split into training and testing sets. To ensure consistent scaling, the features are normalized using `StandardScaler`. A logistic regression model is then trained on the scaled training data and evaluated on the test set using accuracy score. Finally, the model's decision boundary is visualized to show how it separates placed vs. non-placed students.
