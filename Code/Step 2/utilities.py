import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from kneed import KneeLocator
from shap import TreeExplainer
import matplotlib.pyplot as plt
import pandas as pd


def create_understandability_dict(understandability_df):
    #Identify Keys
    keys = list(set(understandability_df['Feature']))
    
    #Set up output dictionary
    dict_out = {}
    
    for item in keys:
        scores = understandability_df[understandability_df['Feature'] == item]['Average Score'].tolist()
        final_score = sum(scores) / len(scores)
        dict_out[item] = final_score
    
    print(dict_out)
    return dict_out

def clean_feature_name(feat: str) -> str:
    name = feat.split("__", 1)[-1]
    return name.replace("_", " ")

def format_value(value) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.2f}"
    return str(value)

def explain_instance(instance_id, shap_df, class_labels=None, top_n=None):
    class_labels = class_labels or {}

    rows = shap_df[shap_df["instance_id"] == instance_id].sort_values("rank")
    if top_n:
        rows = rows[rows["rank"] <= top_n]

    pred = rows["prediction"].iloc[0]
    pred_label = class_labels.get(pred, str(pred))

    support_phrases, detract_phrases = [], []

    for _, row in rows.iterrows():
        feat_clean = clean_feature_name(row["feature"])
        raw_val = row["feature_value"]

        if row["is_numeric"]:
            comparator = "more than average" if raw_val > row["feature_mean"] else "less than average"
            phrase = f"their {feat_clean} of {format_value(raw_val)} is {comparator}"
        else:
            phrase = f"their {feat_clean} is {raw_val}"

        (support_phrases if row["direction"] == "increases" else detract_phrases).append(phrase)

    def join_phrases(phrases):
        if not phrases:
            return ""
        if len(phrases) == 1:
            return phrases[0]
        return ", ".join(phrases[:-1]) + ", and " + phrases[-1]

    explanation = f"The individual is predicted to {pred_label} because {join_phrases(support_phrases)}."

    if detract_phrases:
        explanation += f" The fact that {join_phrases(detract_phrases)} detracted from this prediction."

    return explanation

#Implement Monte Carlo testing to ensure consistency in accuracy
def Monte_Carlo_Splitting(X, y, test_perc, column_processor, n_runs):
    """Runs repeated Monte Carlo Splitting and returns output"""
    
    accuracy_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    
    for seed_n in range(n_runs):
        
        #First need to split training and testing data in stratified manner, seed updates at every call to ensure changes in split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_perc, stratify=y, random_state=seed_n)
        
        #Define Pipeline function for columns
        
        pipeline = Pipeline(steps=[
        ('preprocessor', column_processor),
        ('classifier', RandomForestClassifier(random_state=seed_n, 
                                            n_estimators=500,
                                            class_weight='balanced'))])
        
        #Fit Pipeline
        pipeline.fit(X_train, y_train)
        
        #Get Predictions
        y_pred = pipeline.predict(X_test)
        
        #Compute Metrics
        accuracy_list.append(balanced_accuracy_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred))
        precision_list.append(precision_score(y_test, y_pred))
        recall_list.append(recall_score(y_test, y_pred))
        
    #Return Metrics
    return accuracy_list, f1_list, precision_list, recall_list

def Feature_Selection(X, y, seed_n, preprocessor, 
                    co_op = True, n_select = 5, plot = False, strictness = 1.0):
    
    #Implement train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed_n)
    
    #Fit preprocessing
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    #Get column names
    feature_names = preprocessor.get_feature_names_out()

    #Transform Back to df
    X_train = pd.DataFrame(X_train, columns=feature_names, index=None)
    X_test = pd.DataFrame(X_test, columns=feature_names, index=None)
    
    #Fit Random Forest
    rf = RandomForestClassifier(n_estimators=100, 
                                random_state=seed_n, 
                                max_depth=6,
                                criterion = 'gini',
                                min_samples_split=0.2)
    
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    feature_names = X_train.columns if hasattr(X_train, 'columns') else np.arange(X_train.shape[1])
    
    order = np.argsort(importances)[::-1]
    sorted_importances = importances[order]
    sorted_names = np.array(feature_names)[order]
    
    ranks = np.arange(1, len(sorted_importances) + 1)
    
    if co_op is True:
        print('In Co_Op')
    
        kn = KneeLocator(
            ranks, sorted_importances,
            curve='convex', direction='decreasing',
            S=strictness  # sensitivity; raise if it's too aggressive, lower if too lax
        )
        
        n_optimal = kn.knee if kn.knee is not None else len(ranks)
        selected_features = sorted_names[:n_optimal]
        
        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(ranks, sorted_importances, marker='o')
            plt.axvline(n_optimal, color='red', linestyle='--', label=f'Elbow at N={n_optimal}')
            plt.xlabel('Feature rank')
            plt.ylabel('Importance')
            plt.title('Feature Importance Elbow')
            plt.legend()
            plt.show()
    
        return selected_features, n_optimal, sorted_importances, sorted_names
    
    if co_op is False:
        print('In Trad')
        selected_features = sorted_names[:n_select]
        n_optimal = n_select
        return selected_features, n_optimal, sorted_importances, sorted_names


def get_understandable_features(pre_selection, understandability_df, n_features, data, y):
    understandability_dict = create_understandability_dict(understandability_df)
    
    #Get understandability scores for selected features
    dict_selected = {x : understandability_dict[x[5:]] for x in pre_selection}
    
    #Sort Dictionary
    dict_sorted = {k: v for k, v in sorted(dict_selected.items(), key=lambda item: item[1])}
    
    #Select top N features
    top_n_keys = list(dict_sorted.keys())[-n_features:]
    
    #Get names of features
    column_names = [x[5:] for x in top_n_keys]
    
    #Subset dataframe
    final_data = data[column_names].copy()
    
    #Add output variable
    final_data['Output'] = data[y]
    
    return top_n_keys, dict_sorted, final_data

def build_inverse_transform_map(preprocessor, numeric_features, categorical_features):
    """
    Returns a dict: transformed column name -> function(value) -> original value
    """
    num_transformer = preprocessor.named_transformers_['num']
    cat_transformer = preprocessor.named_transformers_['cat']

    inverse_map = {}

    # RobustScaler: X_scaled = (X - center) / scale  =>  X = X_scaled * scale + center
    for i, feat in enumerate(numeric_features):
        center = num_transformer.center_[i]
        scale = num_transformer.scale_[i]
        inverse_map[f'num__{feat}'] = lambda v, c=center, s=scale: v * s + c

    # OrdinalEncoder: X_encoded = index into categories_
    for i, feat in enumerate(categorical_features):
        categories = cat_transformer.categories_[i]
        inverse_map[f'cat__{feat}'] = lambda v, cats=categories: cats[int(round(v))]

    return inverse_map

def feature_importance(trained_model, X_train, X_test, y_test, n_instances,
                        preprocessor, numeric_features, categorical_features):
    X_subset = X_test.iloc[:n_instances]
    y_subset = y_test.iloc[:n_instances]

    explainer = TreeExplainer(trained_model)
    raw_shap = explainer.shap_values(X_subset)
    predictions = trained_model.predict(X_subset)

    is_classifier = hasattr(trained_model, "predict_proba")

    if is_classifier:
        class_idx_map = {c: i for i, c in enumerate(trained_model.classes_)}
        if isinstance(raw_shap, list):
            shap_matrix = np.array([
                raw_shap[class_idx_map[pred]][row_i]
                for row_i, pred in enumerate(predictions)
            ])
        else:
            shap_matrix = np.array([
                raw_shap[row_i, :, class_idx_map[pred]]
                for row_i, pred in enumerate(predictions)
            ])
    else:
        shap_matrix = np.asarray(raw_shap)

    feature_names = X_subset.columns
    inverse_map = build_inverse_transform_map(preprocessor, numeric_features, categorical_features)

    # Original-scale means, derived by inverse-transforming the (transformed) X_train
    feature_means = {}
    for feat in feature_names:
        if feat.startswith("num__") and feat in inverse_map:
            original_vals = X_train[feat].apply(inverse_map[feat])
            feature_means[feat] = original_vals.mean()

    records = []

    for row_i in range(shap_matrix.shape[0]):
        instance_id = X_subset.index[row_i]
        row_shap = shap_matrix[row_i]
        order = np.argsort(-np.abs(row_shap))
        actual_val = y_subset.iloc[row_i]

        for rank, feat_i in enumerate(order):
            val = row_shap[feat_i]
            feat_name = feature_names[feat_i]
            transformed_val = X_subset.iloc[row_i, feat_i]

            is_numeric = feat_name.startswith("num__")
            original_val = inverse_map[feat_name](transformed_val) if feat_name in inverse_map else transformed_val

            records.append({
                "instance_id": instance_id,
                "feature": feat_name,
                "feature_value": original_val,
                "feature_mean": feature_means.get(feat_name, np.nan),
                "is_numeric": is_numeric,
                "shap_value": val,
                "importance": abs(val),
                "direction": "increases" if val > 0 else ("decreases" if val < 0 else "no effect"),
                "rank": rank + 1,
                "prediction": predictions[row_i],
                "actual": actual_val,
            })

    return pd.DataFrame.from_records(records)