import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DatasetLoader:
    """Load and prepare the Gallstone dataset."""

    def __init__(self, source):
        self.source = source

    def load(self):
        """Return the dataset as a pandas DataFrame."""
        print("Loading dataset from:", self.source)
        try:
            data = pd.read_csv(self.source, low_memory=False)
        except (OSError, ValueError) as error:
            print("Error loading dataset:", error)
            return pd.DataFrame()

        print("Dataset shape:", data.shape)
        return data

    def preprocess(self, data):
        """Apply dataset-specific cleaning before model training."""
        if data.empty:
            return data

        print("Preprocessing dataset...")
        processed = data.copy()
        string_columns = processed.select_dtypes(include="object").columns
        processed[string_columns] = processed[string_columns].apply(
            lambda column: column.str.lower()
        )
        return processed

    def run(self):
        return self.preprocess(self.load())


class DataVisualizer:
    """Create exploratory and evaluation plots."""

    def __init__(self, data):
        self.data = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    def plot_target_distribution(self, target="Gallstone Status"):
        """Plot the number of records in each target class."""
        if target not in self.data.columns:
            raise ValueError(f"Target column not found: {target}")

        counts = self.data[target].value_counts().sort_index()
        figure, axis = plt.subplots()
        axis.bar(counts.index.astype(str), counts.values)
        axis.set_title(f"Distribution of {target}")
        axis.set_xlabel(target)
        axis.set_ylabel("Number of records")
        figure.tight_layout()
        return figure

    def plot_feature_distribution(self, feature, bins=20):
        """Plot a numeric histogram or categorical bar chart."""
        if feature not in self.data.columns:
            raise ValueError(f"Feature column not found: {feature}")

        values = self.data[feature].dropna()
        figure, axis = plt.subplots()
        if pd.api.types.is_numeric_dtype(values):
            axis.hist(values, bins=bins)
            axis.set_ylabel("Number of records")
        else:
            counts = values.value_counts().head(20)
            axis.bar(counts.index.astype(str), counts.values)
            axis.set_ylabel("Number of records")
            axis.tick_params(axis="x", rotation=45)

        axis.set_title(f"Distribution of {feature}")
        axis.set_xlabel(feature)
        figure.tight_layout()
        return figure

    def plot_correlation_matrix(self):
        """Plot correlations between numeric features."""
        numeric_data = self.data.select_dtypes(include="number")
        if numeric_data.empty:
            raise ValueError("At least one numeric column is required.")

        correlations = numeric_data.corr()
        figure, axis = plt.subplots(figsize=(12, 10))
        image = axis.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
        figure.colorbar(image, ax=axis, label="Correlation")
        labels = correlations.columns.tolist()
        axis.set_xticks(range(len(labels)), labels=labels, rotation=90)
        axis.set_yticks(range(len(labels)), labels=labels)
        axis.set_title("Correlation Matrix")
        figure.tight_layout()
        return figure

    def plot_confusion_matrix(self, actual, predicted):
        """Plot a confusion matrix from actual and predicted labels."""
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)
        if actual.shape != predicted.shape or actual.size == 0:
            raise ValueError("Actual and predicted values must be non-empty and aligned.")

        labels = np.unique(np.concatenate((actual, predicted)))
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for actual_label, predicted_label in zip(actual, predicted):
            actual_index = np.where(labels == actual_label)[0][0]
            predicted_index = np.where(labels == predicted_label)[0][0]
            matrix[actual_index, predicted_index] += 1

        figure, axis = plt.subplots()
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set_xticks(range(len(labels)), labels=[str(label) for label in labels])
        axis.set_yticks(range(len(labels)), labels=[str(label) for label in labels])
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
        axis.set_title("Confusion Matrix")
        figure.tight_layout()
        return figure


class DataProcessor:
    """Prepare features and target values for the model."""

    def __init__(self, data):
        self.data = data

    def process(self):
        """Return the processed data as a DataFrame."""
        print("Processing dataset...")
        return self.data.copy() if isinstance(self.data, pd.DataFrame) else pd.DataFrame(self.data)

    def split_features_and_target(self, target_column="Gallstone Status"):
        """Separate predictors from the target column."""
        if target_column not in self.data.columns:
            raise ValueError(f"Target column not found: {target_column}")

        features = self.data.drop(columns=[target_column])
        target = self.data[target_column]
        return features, target


class XGBoostModelTrainer:
    """Template for the project's XGBoost classification workflow."""

    def __init__(self, data):
        self.data = data

    def train(self):
        """Create, split, and fit the XGBoost classifier."""
        # TODO: split features and target.
        # TODO: create a reproducible train/test split.
        # TODO: create and fit an XGBClassifier.
        raise NotImplementedError

    def evaluate(self, model):
        """Calculate classification metrics on the held-out test data."""
        # TODO: calculate accuracy, precision, recall, F1, and ROC-AUC.
        raise NotImplementedError

    def predict(self, model, new_data):
        """Generate predictions for new patient records."""
        # TODO: validate and transform new_data to the training feature format.
        # TODO: return model predictions and probabilities.
        raise NotImplementedError

    def explain(self, model):
        """Explain feature contributions with SHAP."""
        # TODO: create a SHAP explainer and summarize feature contributions.
        raise NotImplementedError

    def run(self):
        """Run training, evaluation, and explainability."""
        model = self.train()
        metrics = self.evaluate(model)
        explanations = self.explain(model)
        return model, metrics, explanations


def main():
    dataset_loader = DatasetLoader("data/dataset-uci/dataset-uci.csv")
    data = dataset_loader.run()
    processed_data = DataProcessor(data).process()

    trainer = XGBoostModelTrainer(processed_data)
    model, metrics, explanations = trainer.run()
    print("Evaluation metrics:", metrics)
    return model, explanations


if __name__ == "__main__":
    main()
