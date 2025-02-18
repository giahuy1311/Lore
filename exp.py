import numpy as np
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from util import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lore
from prepare_dataset import *
from neighbor_generator import *



def prepare_dataset(df):
    columns = df.columns.tolist()
    class_name = 'y'

    possible_outcomes = list(df[class_name].unique())

    type_features, features_type = recognize_features_type(df, class_name)
    print(type_features)
    discrete = ['y']
    discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete=discrete,
                                                   continuous=None)
    print(discrete, continuous)

    columns_tmp = list(columns)
    columns_tmp.remove(class_name)
    idx_features = {i: col for i, col in enumerate(columns_tmp)}

    # Dataset Preparation for Scikit Alorithms
    df_le, label_encoder = label_encode(df, discrete)
    X = df_le.loc[:, df_le.columns != class_name].values
    y = df_le[class_name].values


    dataset = {
        'df': df, 
        'columns': columns, 
        'class_name': class_name,  
        'possible_outcomes': possible_outcomes, 
        'type_features': type_features, 
        'features_type': features_type,
        'discrete': discrete,
        'continuous': continuous, 
        'label_encoder': label_encoder,   
        'idx_features': idx_features,  
        'X': X,
        'y': y 
    }

    return dataset

def main():
    x1 = np.random.uniform(0, 100, 5000)
    x2 = np.random.uniform(0, 100, 5000)
    x3 = np.random.uniform(0, 100, 5000)
    
    y = np.where((x1 + 2*x2 - 0.5*x3 > 120) & (x1 * x2 < 1000), 1, 0)
    
    df = pd.DataFrame({'x1': x1, 'x2':x2, 'x3':x3,'y': y}) 
    
    dataset = prepare_dataset(df) # dùng lại từ lore
    
    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # blackbox
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    path_data = 'datasets/'
    idx_record2explain = 2
    X2E = X_test
    print('X2E', X2E)
    # tạo neighbors = cách thêm nhiễu nhỏ random
    explanation, infos = lore.explain(idx_record2explain, X2E, dataset, model,
                                        ng_function=genetic_neighborhood,
                                        discrete_use_probabilities=True,
                                        continuous_function_estimation=False,
                                        returns_infos=True,
                                        path=path_data, sep=';', log=False)
    
    dfX2E = build_df2explain(model, X2E, dataset).to_dict('records')
    dfx = dfX2E[idx_record2explain]
    # x = build_df2explain(blackbox, X2E[idx_record2explain].reshape(1, -1), dataset).to_dict('records')[0]

    print('x = %s' % dfx)
    
    print('r = %s --> %s' % (explanation[0][1], explanation[0][0]))
    for delta in explanation[1]:
        print('delta', delta)
    
    
if __name__ == "__main__":
    main()

    