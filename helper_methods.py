def compare_depths(self, depths=(4, 6), n_splits=10):
    """

    Compare tree depths using the same repeated patient splits.
    Useful for comparing the performance of different tree depths on the same data splits.


    """
    processor = DataProcessor(self.data)
    features, target = processor.split_features_and_target()

    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=self.test_size,
        random_state=self.random_state,
    )

    results = []

    for split_number, (train_index, test_index) in enumerate(
        splitter.split(features, target),
        start=1,
    ):
        X_train = features.iloc[train_index]
        X_test = features.iloc[test_index]
        y_train = target.iloc[train_index]
        y_test = target.iloc[test_index]

        for depth in depths:
            model = self.create_model(max_depth=depth)
            model.fit(X_train, y_train)

            probabilities = self.gallstone_probabilities(
                model,
                X_test,
            )
            predictions = self.labels_from_probabilities(probabilities)

            metrics = gallstone_metrics(
                y_test,
                predictions,
                probabilities,
            )

            results.append({
                "split": split_number,
                "max_depth": depth,
                **metrics,
            })

    results_df = pd.DataFrame(results)

    summary = (
        results_df
        .groupby("max_depth")[
            [
                "accuracy",
                "sensitivity_gallstones",
                "precision_gallstones",
                "f1_gallstones",
                "roc_auc_gallstones",
            ]
        ]
        .agg(["mean", "std"])
    )

    return results_df, summary


results, summary = trainer.compare_depths(
    depths=(4, 6),
    n_splits=10,
)

print(summary)