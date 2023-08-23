import json
import urllib.request
import pandas as pd
from tqdm import tqdm
import itertools
from multiprocessing.pool import Pool


def load_az_json(url):
    with urllib.request.urlopen(url) as f:
        neuron_record = f.read()

    return json.loads(neuron_record)


def load_neuron(layer_index, neuron_index):
    url = f"https://openaipublic.blob.core.windows.net/neuron-explainer/data/explanations/{layer_index}/{neuron_index}.jsonl"

    return load_az_json(url)


def get_neuron_score_and_explanation(layer_index, neuron_index):
    neuron = load_neuron(layer_index, neuron_index)
    explanation = neuron["scored_explanations"][0]["explanation"]
    score = neuron["scored_explanations"][0]["scored_simulation"]["ev_correlation_score"]

    return score, explanation


def download_layer(l):
    n_neurons_per_layer = 6400
    df = pd.DataFrame(columns=["layer", "neuron_number", "score", "explanation"])
    bad_df = pd.DataFrame(columns=["layer", "neuron_number"])

    for n in tqdm(range(n_neurons_per_layer)):
        try:
            if n % 100 == 0:
                print(f"Layer {l} got to {n}")

            score, explanation = get_neuron_score_and_explanation(layer_index=l, neuron_index=n)
            location = len(df)
            df.loc[location] = [l, n, score, explanation]
        except Exception:
            print(f"Couldn't get {l}, {n}")
            location = len(bad_df)
            bad_df.loc[location] = [l, n]

    df.to_csv(f"neurons_data/data_df_layer_{l}.csv", sep=',', index=False)
    bad_df.to_csv(f"bad_data/bad_df_layer_{l}.csv", sep=',', index=False)


def download_missing_neurons(l):
    new_df = pd.DataFrame(columns=["layer", "neuron_number", "score", "explanation"])
    df = pd.read_csv(f"bad_data/bad_df_layer_{l}.csv", sep=',')
    for i, n in enumerate(tqdm(df.neuron_number.values)):
        try:
            if i % 100 == 0:
                print(f"Layer {l} got to {n}")

            score, explanation = get_neuron_score_and_explanation(layer_index=l, neuron_index=n)
            location = len(new_df)
            new_df.loc[location] = [l, n, score, explanation]
        except Exception:
            print(f"Couldn't get {l}, {n}")
    new_df.to_csv(f"new_neurons_data/new_data_df_layer_{l}.csv", sep=',', index=False)


def download_everything_missing():
    almost_full_df = pd.DataFrame(columns=["layer", "neuron_number", "score", "explanation"])
    n_layers = 48
    n_neurons_per_layer = 6400
    for l in range(n_layers):
        df = pd.read_csv(f"neurons_data/data_df_layer_{l}.csv", sep=',')
        almost_full_df = pd.concat([df, almost_full_df], ignore_index=True)

        df = pd.read_csv(f"new_neurons_data/new_data_df_layer_{l}.csv", sep=',')
        almost_full_df = pd.concat([df, almost_full_df], ignore_index=True)

    almost_full_df.reset_index()
    all_pairs = set(itertools.product(range(n_layers), range(n_neurons_per_layer)))
    existing_pairs = set([tuple(a) for a in list(almost_full_df[['layer', 'neuron_number']].values)])
    missing_pairs = all_pairs - existing_pairs
    for l, n in tqdm(list(missing_pairs)):
        try:
            score, explanation = get_neuron_score_and_explanation(layer_index=l, neuron_index=n)
            location = len(almost_full_df)
            almost_full_df.loc[location] = [l, n, score, explanation]
        except Exception:
            print(f"Couldn't get {l}, {n}")
    almost_full_df.to_csv(f"scores_and_explanations.csv", sep=',', index=False)


# program entry point
if __name__ == "__main__":
    df = pd.read_csv(f"scores_and_explanations.csv", sep=',')
