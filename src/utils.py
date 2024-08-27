import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.model_selection import train_test_split

def load_customer_dataset():
    return pd.read_csv('../AV1/dataset/customer_data.csv')

def load_new_customer_dataset():
    return pd.read_csv('../AV1/dataset/new_customer_data.csv')

def load_volunteer_dataset():
    return pd.read_csv('../dataset/opportunities.csv')

def load_wine_dataset():
    wine = pd.read_csv('../dataset/wine.csv')
    return wine

def load_hiking_dataset():
    return pd.read_json('../dataset/hiking.json')

def load_df1_unidade1():
    return pd.read_csv('../dataset/df1_unidade1.csv')


def load_df2_unidade1():
    return pd.read_csv('../dataset/df2_unidade1.csv')

def load_df1_unidade2():
    return pd.read_csv('../dataset/df1_unidade2.csv')

def load_df2_unidade2():
    return pd.read_csv('../dataset/df2_unidade2.csv')

def load_churn_dataset():
    df = pd.read_csv('../dataset/churn_train.csv')
    le = LabelEncoder()
    df['churn'] = le.fit_transform(df['churn'])
    return df

def load_iris_dataset():
    df = pd.read_csv('../dataset/iris.csv')
    return df

def load_sales_clean_dataset():
    df = pd.read_csv('../dataset/sales_clean.csv')
    return df

def load_diabetes_clean_dataset():
    df = pd.read_csv('../dataset/diabetes_clean.csv')
    return df

def processing_sales_clean():
    sales_df = load_sales_clean_dataset()
    y = sales_df["sales"].values
    X = sales_df["tv"].values.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(X, y)
    predictions = reg.predict(X)
    return X,y,predictions

def processing_all_features_sales_clean():
    sales_df = load_sales_clean_dataset()
    X = sales_df.drop(["sales", "influencer"], axis=1)
    y = sales_df["sales"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    reg = LinearRegression()
    reg.fit(X_test, y_test)
    predictions = reg.predict(X)
    return X, y, predictions

def process_diabetes():
    from src.utils import load_diabetes_clean_dataset
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    diabetes_df = load_diabetes_clean_dataset()
    X = diabetes_df.drop(['diabetes'], axis=1)
    y = diabetes_df['diabetes'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    knn = KNeighborsClassifier(n_neighbors=6)

    # Fit the model to the training data
    knn.fit(X_train, y_train)

    # Predict the labels of the test data: y_pred
    y_pred = knn.predict(X_test)

    return y_pred , y_test

def log_reg_diabetes():

    diabetes_df = load_diabetes_clean_dataset()
    X = diabetes_df.drop(['diabetes'], axis=1)
    y = diabetes_df['diabetes'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Instantiate the model
    logreg = LogisticRegression()

    # Fit the model
    logreg.fit(X_train, y_train)

    teste = logreg.predict_proba(X_test)
    # Predict probabilities
    y_pred_probs = logreg.predict_proba(X_test)[:, 1]

    y_pred = logreg.predict(X_test)

    return y_pred_probs , y_test, y_pred

def load_points():
    return np.array([[0.06544649, -0.76866376],
                    [-1.52901547, -0.42953079],
                    [1.70993371, 0.69885253],
                    [1.16779145, 1.01262638],
                    [-1.80110088, -0.31861296],
                    [-1.63567888, -0.02859535],
                    [1.21990375, 0.74643463],
                    [-0.26175155, -0.62492939],
                    [-1.61925804, -0.47983949],
                    [-1.84329582, -0.16694431],
                    [1.35999602, 0.94995827],
                    [0.42291856, -0.7349534],
                    [-1.68576139, 0.10686728],
                    [0.90629995, 1.09105162],
                    [-1.56478322, -0.84675394],
                    [-0.0257849, -1.18672539],
                    [0.83027324, 1.14504612],
                    [1.22450432, 1.35066759],
                    [-0.15394596, -0.71704301],
                    [0.86358809, 1.06824613],
                    [-1.43386366, -0.2381297],
                    [0.03844769, -0.74635022],
                    [-1.58567922, 0.08499354],
                    [0.6359888, -0.58477698],
                    [0.24417242, -0.53172465],
                    [-2.19680359, 0.49473677],
                    [1.0323503, -0.55688],
                    [-0.28858067, -0.39972528],
                    [0.20597008, -0.80171536],
                    [-1.2107308, -0.34924109],
                    [1.33423684, 0.7721489],
                    [1.19480152, 1.04788556],
                    [0.9917477, 0.89202008],
                    [-1.8356219, -0.04839732],
                    [0.08415721, -0.71564326],
                    [-1.48970175, -0.19299604],
                    [0.38782418, -0.82060119],
                    [-0.01448044, -0.9779841],
                    [-2.0521341, -0.02129125],
                    [0.10331194, -0.82162781],
                    [-0.44189315, -0.65710974],
                    [1.10390926, 1.02481182],
                    [-1.59227759, -0.17374038],
                    [-1.47344152, -0.02202853],
                    [-1.35514704, 0.22971067],
                    [0.0412337, -1.23776622],
                    [0.4761517, -1.13672124],
                    [1.04335676, 0.82345905],
                    [-0.07961882, -0.85677394],
                    [0.87065059, 1.08052841],
                    [1.40267313, 1.07525119],
                    [0.80111157, 1.28342825],
                    [-0.16527516, -1.23583804],
                    [-0.33779221, -0.59194323],
                    [0.80610749, -0.73752159],
                    [-1.43590032, -0.56384446],
                    [0.54868895, -0.95143829],
                    [0.46803131, -0.74973907],
                    [-1.5137129, -0.83914323],
                    [0.9138436, 1.51126532],
                    [-1.97233903, -0.41155375],
                    [0.5213406, -0.88654894],
                    [0.62759494, -1.18590477],
                    [0.94163014, 1.35399335],
                    [0.56994768, 1.07036606],
                    [-1.87663382, 0.14745773],
                    [0.90612186, 0.91084011],
                    [-1.37481454, 0.28428395],
                    [-1.80564029, -0.96710574],
                    [0.34307757, -0.79999275],
                    [0.70380566, 1.00025804],
                    [-1.68489862, -0.30564595],
                    [1.31473221, 0.98614978],
                    [0.26151216, -0.26069251],
                    [0.9193121, 0.82371485],
                    [-1.21795929, -0.20219674],
                    [-0.17722723, -1.02665245],
                    [0.64824862, -0.66822881],
                    [0.41206786, -0.28783784],
                    [1.01568202, 1.13481667],
                    [0.67900254, -0.91489502],
                    [-1.05182747, -0.01062376],
                    [0.61306599, 1.78210384],
                    [-1.50219748, -0.52308922],
                    [-1.72717293, -0.46173916],
                    [-1.60995631, -0.1821007],
                    [-1.09111021, -0.0781398],
                    [-0.01046978, -0.80913034],
                    [0.32782303, -0.80734754],
                    [1.22038503, 1.1959793],
                    [-1.33328681, -0.30001937],
                    [0.87959517, 1.11566491],
                    [-1.14829098, -0.30400762],
                    [-0.58019755, -1.19996018],
                    [-0.01161159, -0.78468854],
                    [0.17359724, -0.63398145],
                    [1.32738556, 0.67759969],
                    [-1.93467327, 0.30572472],
                    [-1.57761893, -0.27726365],
                    [0.47639, 1.21422648],
                    [-1.65237509, -0.6803981],
                    [-0.12609976, -1.04327457],
                    [-1.89607082, -0.70085502],
                    [0.57466899, 0.74878369],
                    [-0.16660312, -0.83110295],
                    [0.8013355, 1.22244435],
                    [1.18455426, 1.4346467],
                    [1.08864428, 0.64667112],
                    [-1.61158505, 0.22805725],
                    [-1.57512205, -0.09612576],
                    [0.0721357, -0.69640328],
                    [-1.40054298, 0.16390598],
                    [1.09607713, 1.16804691],
                    [-2.54346204, -0.23089822],
                    [-1.34544875, 0.25151126],
                    [-1.35478629, -0.19103317],
                    [0.18368113, -1.15827725],
                    [-1.31368677, -0.376357],
                    [0.09990129, 1.22500491],
                    [1.17225574, 1.30835143],
                    [0.0865397, -0.79714371],
                    [-0.21053923, -1.13421511],
                    [0.26496024, -0.94760742],
                    [-0.2557591, -1.06266022],
                    [-0.26039757, -0.74774225],
                    [-1.91787359, 0.16434571],
                    [0.93021139, 0.49436331],
                    [0.44770467, -0.72877918],
                    [-1.63802869, -0.58925528],
                    [-1.95712763, -0.10125137],
                    [0.9270337, 0.88251423],
                    [1.25660093, 0.60828073],
                    [-1.72818632, 0.08416887],
                    [0.3499788, -0.30490298],
                    [-1.51696082, -0.50913109],
                    [0.18763605, -0.55424924],
                    [0.89609809, 0.83551508],
                    [-1.54968857, -0.17114782],
                    [1.2157457, 1.23317728],
                    [0.20307745, -1.03784906],
                    [0.84589086, 1.03615273],
                    [0.53237919, 1.47362884],
                    [-0.05319044, -1.36150553],
                    [1.38819743, 1.11729915],
                    [1.00696304, 1.0367721],
                    [0.56681869, -1.09637176],
                    [0.86888296, 1.05248874],
                    [-1.16286609, -0.55875245],
                    [0.27717768, -0.83844015],
                    [0.16563267, -0.80306607],
                    [0.38263303, -0.42683241],
                    [1.14519807, 0.89659026],
                    [0.81455857, 0.67533667],
                    [-1.8603152, -0.09537561],
                    [0.965641, 0.90295579],
                    [-1.49897451, -0.33254044],
                    [-0.1335489, -0.80727582],
                    [0.12541527, -1.13354906],
                    [1.06062436, 1.28816358],
                    [-1.49154578, -0.2024641],
                    [1.16189032, 1.28819877],
                    [0.54282033, 0.75203524],
                    [0.89221065, 0.99211624],
                    [-1.49932011, -0.32430667],
                    [0.3166647, -1.34482915],
                    [0.13972469, -1.22097448],
                    [-1.5499724, -0.10782584],
                    [1.23846858, 1.37668804],
                    [1.25558954, 0.72026098],
                    [0.25558689, -1.28529763],
                    [0.45168933, -0.55952093],
                    [1.06202057, 1.03404604],
                    [0.67451908, -0.54970299],
                    [0.22759676, -1.02729468],
                    [-1.45835281, -0.04951074],
                    [0.23273501, -0.70849262],
                    [1.59679589, 1.11395076],
                    [0.80476105, 0.544627],
                    [1.15492521, 1.04352191],
                    [0.59632776, -1.19142897],
                    [0.02839068, -0.43829366],
                    [1.13451584, 0.5632633],
                    [0.21576204, -1.04445753],
                    [1.41048987, 1.02830719],
                    [1.12289302, 0.58029441],
                    [0.25200688, -0.82588436],
                    [-1.28566081, -0.07390909],
                    [1.52849815, 1.11822469],
                    [-0.23907858, -0.70541972],
                    [-0.25792784, -0.81825035],
                    [0.59367818, -0.45239915],
                    [0.07931909, -0.29233213],
                    [-1.27256815, 0.11630577],
                    [0.66930129, 1.00731481],
                    [0.34791546, -1.20822877],
                    [-2.11283993, -0.66897935],
                    [-1.6293824, -0.32718222],
                    [-1.53819139, -0.01501972],
                    [-0.11988545, -0.6036339],
                    [-1.54418956, -0.30389844],
                    [0.30026614, -0.77723173],
                    [0.00935449, -0.53888192],
                    [-1.33424393, -0.11560431],
                    [0.47504489, 0.78421384],
                    [0.59313264, 1.232239],
                    [0.41370369, -1.35205857],
                    [0.55840948, 0.78831053],
                    [0.49855018, -0.789949],
                    [0.35675809, -0.81038693],
                    [-1.86197825, -0.59071305],
                    [-1.61977671, -0.16076687],
                    [0.80779295, -0.73311294],
                    [1.62745775, 0.62787163],
                    [-1.56993593, -0.08467567],
                    [1.02558561, 0.89383302],
                    [0.24293461, -0.6088253],
                    [1.23130242, 1.00262186],
                    [-1.9651013, -0.15886289],
                    [0.42795032, -0.70384432],
                    [-1.58306818, -0.19431923],
                    [-1.57195922, 0.01413469],
                    [-0.98145373, 0.06132285],
                    [-1.48637844, -0.5746531],
                    [0.98745828, 0.69188053],
                    [1.28619721, 1.28128821],
                    [0.85850596, 0.95541481],
                    [0.19028286, -0.82112942],
                    [0.26561046, -0.04255239],
                    [-1.61897897, 0.00862372],
                    [0.24070183, -0.52664209],
                    [1.15220993, 0.43916694],
                    [-1.21967812, -0.2580313],
                    [0.33412533, -0.86117761],
                    [0.17131003, -0.75638965],
                    [-1.19828397, -0.73744665],
                    [-0.12245932, -0.45648879],
                    [1.51200698, 0.88825741],
                    [1.10338866, 0.92347479],
                    [1.30972095, 0.59066989],
                    [0.19964876, 1.14855889],
                    [0.81460515, 0.84538972],
                    [-1.6422739, -0.42296206],
                    [0.01224351, -0.21247816],
                    [0.33709102, -0.74618065],
                    [0.47301054, 0.72712075],
                    [0.34706626, 1.23033757],
                    [-0.00393279, -0.97209694],
                    [-1.64303119, 0.05276337],
                    [1.44649625, 1.14217033],
                    [-1.93030087, -0.40026146],
                    [-2.37296135, -0.72633645],
                    [0.45860122, -1.06048953],
                    [0.4896361, -1.18928313],
                    [-1.02335902, -0.17520578],
                    [-1.32761107, -0.93963549],
                    [-1.50987909, -0.09473658],
                    [0.02723057, -0.79870549],
                    [1.0169412, 1.26461701],
                    [0.47733527, -0.9898471],
                    [-1.27784224, -0.547416],
                    [0.49898802, -0.6237259],
                    [1.06004731, 0.86870008],
                    [1.00207501, 1.38293512],
                    [1.31161394, 0.62833956],
                    [1.13428443, 1.18346542],
                    [1.27671346, 0.96632878],
                    [-0.63342885, -0.97768251],
                    [0.12698779, -0.93142317],
                    [-1.34510812, -0.23754226],
                    [-0.53162278, -1.25153594],
                    [0.21959934, -0.90269938],
                    [-1.78997479, -0.12115748],
                    [1.23197473, -0.07453764],
                    [1.4163536, 1.21551752],
                    [-1.90280976, -0.1638976],
                    [-0.22440081, -0.75454248],
                    [0.59559412, 0.92414553],
                    [1.21930773, 1.08175284],
                    [-1.99427535, -0.37587799],
                    [-1.27818474, -0.52454551],
                    [0.62352689, -1.01430108],
                    [0.14024251, -0.428266],
                    [-0.16145713, -1.16359731],
                    [-1.74795865, -0.06033101],
                    [-1.16659791, 0.0902393],
                    [0.41110408, -0.8084249],
                    [1.14757168, 0.77804528],
                    [-1.65590748, -0.40105446],
                    [-1.15306865, 0.00858699],
                    [0.60892121, 0.68974833],
                    [-0.08434138, -0.97615256],
                    [0.19170053, -0.42331438],
                    [0.29663162, -1.13357399],
                    [-1.36893628, -0.25052124],
                    [-0.08037807, -0.56784155],
                    [0.35695011, -1.15064408],
                    [0.02482179, -0.63594828],
                    [-1.49075558, -0.2482507],
                    [-1.408588, 0.25635431],
                    [-1.98274626, -0.54584475]])

def load_grains_dataset():
    return pd.read_csv('../dataset/grains.csv')

def load_fish_dataset():
    return pd.read_csv('../dataset/fish.csv')

def load_movements_price_dataset():
    return pd.read_csv('../dataset/company-stock-movements-2010-2015-incl.csv')

def load_grains_splited_datadet():
    df = pd.read_csv('../dataset/grains.csv')
    X =  df.drop(['variety','variety_number'],axis=1)
    y =  df['variety'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42, stratify=y)
    return  X_train, X_test, y_train, y_test