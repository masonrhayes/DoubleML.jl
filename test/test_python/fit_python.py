"""Generic Python model runner for Python validation tests.
Reads configuration from config.toml and runs all specified tests.
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import toml
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from doubleml import DoubleMLData, DoubleMLIRM, DoubleMLPLR, DoubleMLLPLR

# Import data generators from DoubleML submodules
from doubleml.irm.datasets import make_irm_data
from doubleml.plm.datasets import make_plr_CCDDHNR2018, make_lplr_LZZ2020

# Load configuration
CONFIG = toml.load(os.path.join(os.path.dirname(__file__), 'config.toml'))
DATA_GEN = CONFIG['data_generation']
MODEL_FIT = CONFIG['model_fitting']

N_FOLDS = MODEL_FIT['n_folds']
N_REP = MODEL_FIT['n_rep']
RNG_SEED = DATA_GEN['rng_seed']

# Data generator mapping - uses actual DoubleML Python functions
# Note: These don't accept random_state, so we set numpy random seed before calling
DATA_GENERATORS = {
    'make_plr_CCDDHNR2018': lambda n_obs, dim_x: make_plr_CCDDHNR2018(
        n_obs=n_obs, dim_x=dim_x, alpha=DATA_GEN['alpha']
    ),
    'make_irm_data': lambda n_obs, dim_x: make_irm_data(
        n_obs=n_obs, dim_x=dim_x, theta=DATA_GEN['theta']
    ),
    'make_lplr_LZZ2020': lambda n_obs, dim_x: make_lplr_LZZ2020(
        n_obs=n_obs, dim_x=dim_x, alpha=DATA_GEN['lplr_alpha']
    )
}


def instantiate_learners(learner_name, learner_config):
    """Instantiate learners based on configuration."""
    py_config = learner_config['python']
    py_classifier_config = learner_config.get('python_classifier', None)
    
    # Get kwargs for regressor
    regressor_kwargs = {k: v for k, v in py_config.items() 
                        if k not in ['module', 'class']}
    
    # Import and create regressor
    module_name = py_config['module']
    class_name = py_config['class']
    if class_name == 'RandomForestRegressor':
        RegressorClass = RandomForestRegressor
        ClassifierClass = RandomForestClassifier
    elif class_name == 'HistGradientBoostingRegressor':
        RegressorClass = HistGradientBoostingRegressor
        ClassifierClass = HistGradientBoostingClassifier
    else:
        raise ValueError(f"Unknown class: {class_name}")
    
    return RegressorClass, ClassifierClass, regressor_kwargs


def construct_model(model_name, data_path, score_config, learner_name, config):
    """Construct a DoubleML model based on configuration."""
    # Load data
    df = pd.read_csv(data_path)
    
    # Determine y_col and d_col based on model type
    if model_name in ['PLR', 'IRM', 'LPLR']:
        y_col, d_col = 'y', 'd'
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    dml_data = DoubleMLData(df, y_col, d_col)
    
    # Get learner config
    learner_config = config['learners'][learner_name]
    RegressorClass, ClassifierClass, kwargs = instantiate_learners(learner_name, learner_config)
    
    # Construct model
    if model_name == 'PLR':
        ml_l = RegressorClass(**kwargs)
        ml_m = RegressorClass(**kwargs)
        score = score_config['python_score']
        
        if score_config.get('requires_ml_g', False):
            ml_g = RegressorClass(**kwargs)
            model = DoubleMLPLR(
                dml_data, ml_l=ml_l, ml_m=ml_m, ml_g=ml_g,
                n_folds=N_FOLDS, n_rep=N_REP, score=score
            )
        else:
            model = DoubleMLPLR(
                dml_data, ml_l=ml_l, ml_m=ml_m,
                n_folds=N_FOLDS, n_rep=N_REP, score=score
            )
            
    elif model_name == 'IRM':
        ml_g = RegressorClass(**kwargs)
        ml_m = ClassifierClass(**kwargs)
        score = score_config['python_score']
        normalize = score_config.get('normalize_ipw', [False])[0]
        
        model = DoubleMLIRM(
            dml_data, ml_g=ml_g, ml_m=ml_m,
            n_folds=N_FOLDS, n_rep=N_REP, score=score,
            normalize_ipw=normalize
        )
        
    elif model_name == 'LPLR':
        # LPLR has 3 learners: ml_M (classifier), ml_t (regressor), ml_m (regressor)
        ml_M = ClassifierClass(**kwargs)
        ml_t = RegressorClass(**kwargs)
        ml_m = RegressorClass(**kwargs)
        score = score_config['python_score']
        
        model = DoubleMLLPLR(
            dml_data, ml_M=ml_M, ml_t=ml_t, ml_m=ml_m,
            n_folds=N_FOLDS, n_rep=N_REP, score=score
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def run_test(model_name, data_path, score_config, learner_name, config):
    """Run a single test and return results."""
    try:
        model = construct_model(model_name, data_path, score_config, learner_name, config)
        model.fit()
        return {
            'coef': float(model.coef[0]),
            'se': float(model.se[0])
        }
    except Exception as e:
        print(f"  Warning: Test failed - {model_name} with {learner_name} ({score_config['name']}): {str(e)}")
        return {'coef': float('nan'), 'se': float('nan')}


def convert_dml_to_df(dml_data):
    """Convert DoubleMLData to pandas DataFrame."""
    df = pd.DataFrame(dml_data.x, columns=dml_data.x_cols)
    df[dml_data.d_cols[0]] = dml_data.d
    df[dml_data.y_col] = dml_data.y
    # Drop 'p' column if it exists (LPLR generates this but we don't need it)
    if 'p' in df.columns:
        df = df.drop(columns=['p'])
    return df


def generate_model_data(model_def, data_gen, rng):
    """Generate data for a specific model using DoubleML's generators."""
    gen_name = model_def['data_generator_python']
    gen_func = DATA_GENERATORS[gen_name]
    
    # Set numpy random seed for reproducibility
    np.random.seed(rng)
    
    dml_data = gen_func(data_gen['n_obs'], data_gen['dim_x'])
    
    # Convert to DataFrame for CSV saving
    return convert_dml_to_df(dml_data)


def run_all_tests(data_dir, config):
    """Run all tests defined in config."""
    results = {}
    
    # Load data for each model type
    data_cache = {}
    
    for model_def in config['models']:
        model_name = model_def['name']
        gen_name = model_def['data_generator_python']
        
        # Load Python data
        py_file = os.path.join(data_dir, f'{gen_name}_py.csv')
        if gen_name not in data_cache:
            print(f"  Loading data for {model_name} ({gen_name})...")
            data_cache[gen_name] = pd.read_csv(py_file)
        
        df_py = data_cache[gen_name]
        
        # Load Julia data if exists
        jl_file = os.path.join(data_dir, f'{gen_name}_jl.csv')
        has_julia_data = os.path.exists(jl_file)
        
        for score_config in model_def['scores']:
            score_name = score_config['name']
            
            for learner_name in score_config['learners']:
                test_key = f"{model_name}_{score_name}_{learner_name.lower()}"
                
                # Direction 1: Python data → Python models
                print(f"  d1_{test_key}")
                results[f'd1_{test_key}'] = run_test(
                    model_name, py_file, score_config, learner_name, config
                )
                
                # Direction 2: Julia data → Python models (if available)
                if has_julia_data:
                    print(f"  d2_{test_key}")
                    results[f'd2_{test_key}'] = run_test(
                        model_name, jl_file, score_config, learner_name, config
                    )
    
    return results


def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate data using existing script
    print("Generating Python data...")
    # Import and run the data generation
    exec(open(os.path.join(os.path.dirname(__file__), 'generate_data_python.py')).read())
    
    # Run tests
    print("\nRunning Python models...")
    results = run_all_tests(data_dir, CONFIG)
    
    # Save results
    results_path = os.path.join(data_dir, 'results_python.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Python results saved to: {results_path}")
    print(f"  Total tests: {len(results)}")


if __name__ == '__main__':
    main()
