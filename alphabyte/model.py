import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.impute import SimpleImputer
from PreProcessor import PreProcessor
from feat_imp import Feature_Selection
from functools import partial
import copy
import joblib
from gen_alg_pre import acc_score, mse_score
from model_selection import ModelSelection,run_model
#from gen_alg import run_evolution,best_feature, generate_population, fitness_func
from random import choices, randint, randrange, random
from typing import List, Optional, Callable, Tuple
import os

accuracy={}
mse={}

#data=pd.DataFrame()
Genome = List[int]
Population = List[Genome]
PopulateFunc = Callable[[], Population]
TARGET_VAR=str
Data_Batch=List[pd.DataFrame]
FitnessFunc = Callable[[Genome,TARGET_VAR,Data_Batch],int ]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]
Data = pd.DataFrame


def generate_genome(length: int) -> Genome:
    return choices([0, 1], k=length)

def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]

def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of same length")

    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome

def population_fitness(population: Population, fitness_func: FitnessFunc) -> int:
    return sum([fitness_func(genome) for genome in population])

def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    weights=[fitness_func(gene) for gene in population]
    # print(weights)
    return choices(
        population=population,
        weights=weights,
        k=2
    )

def sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=True)

def genome_to_string(genome: Genome) -> str:
    return "".join(map(str, genome))

def print_stats(population: Population, generation_id: int, fitness_func: FitnessFunc):
    print("GENERATION %02d" % generation_id)
    print("=============")
    print("Population: [%s]" % ", ".join([genome_to_string(gene) for gene in population]))
    print("Avg. Fitness: %f" % (population_fitness(population, fitness_func) / len(population)))
    sorted_population = sort_population(population, fitness_func)
    print(
        "Best: %s (%f)" % (genome_to_string(sorted_population[0]), fitness_func(sorted_population[0])))
    print("Worst: %s (%f)" % (genome_to_string(sorted_population[-1]),
                              fitness_func(sorted_population[-1])))
    print("")

    return sorted_population[0]


def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100,
        printer: Optional[PrinterFunc] = None) \
        -> Tuple[Population, int]:
    population = populate_func()
    #print(data)
    for i in range(generation_limit):
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)

        if printer is not None:
            printer(population, i, fitness_func)


        next_generation = population[0:2]

        for _ in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

    return population, i

def from_genome(genome: Genome, target: str,columns: List) -> List:    
    new_list = columns.copy()
    #print(type(new_list))
    new_list=[column for column in new_list if column!=target]
    #new_list.remove(target)
   #print(new_list)
    result = []
    #print("Genome",genome)
    #print("New List",new_list)
    for i, column in enumerate(new_list):
        if genome[i] == 1:
            result.append(column)
    result.append(target)
    return result

def entropy(class_probs):
    entropy = 0
    for prob in class_probs:
        prob_array=np.array(prob)
        if prob_array.any() != 0:
            entropy -= prob_array * np.log2(prob_array)
    return entropy

def information_gain(data, target_column):
    total_entropy = entropy([(data[target_column] == c).mean() for c in set(data[target_column])])

    total_info_gain = 0
    for column in data.columns:
        if column != target_column:
            column_entropy = 0
            for value in set(data[column]):
                subset = data[data[column] == value]
                class_probs = [(subset[target_column] == c).mean() for c in set(subset[target_column])]
                column_entropy += (len(subset) / len(data)) * entropy(class_probs)
            total_info_gain += total_entropy - column_entropy

    return total_info_gain

def get_random_subsets(dataframe:pd.DataFrame, batch_count:int)->List[pd.DataFrame]:
    total_rows = len(dataframe)
    batch_size = total_rows // batch_count  # Dynamic calculation of batch size
    subsets = []
    for _ in range(batch_count):
        # Randomly choose rows without replacement
        indices = np.random.choice(total_rows, size=batch_size, replace=False)
        subset = dataframe.iloc[indices].copy()  # Make a copy to avoid modifying the original DataFrame
        subsets.append(subset)

    return subsets

def fitness_func(genome:Genome,target_var:str,Data_Batch:List[pd.DataFrame])->int:
    # FITNESS_COMPUTE_COUNT=FITNESS_COMPUTE_COUNT+1
    #print('im called')
    #batch_samples=Data_Batch
    # index = randint(0,len(batch_samples)-1)
    ##print("FF",Data_Batch)
    index=0
    #print(Data_Batch[index])
    current_batch = pd.DataFrame(Data_Batch[index])
    ##print("FF",current_batch)
    current_batch = current_batch[from_genome(genome,target_var,current_batch.columns.to_list())]
    return information_gain(current_batch,target_var)
    
def best_feature(result:List[int],columns:List[str])->None:
    #if(len(result)!=len(columns)):
    #    columns.remove(target)
    print(columns)
    min=0
    best_feature={}
    while len(best_feature)!=len(result):
        for i,val in enumerate(result):
            if val==min:
                print(columns[i],":",val)
                best_feature[columns[i]]=val
        min=min-1
    #print(best_feature)
    return [features for features in best_feature.keys() if best_feature[features]>-2]

def preprocess(data,target):
    pp=PreProcessor()
    data=pp.RemoveIrrelevantColumn(data)
    data=pp.HandlingMissingData(data)
    if data[target].dtype!='object':
        data=pp.encoding(data)
    else:
        for i in data.columns:
            if i!=target:
                data.loc[:,i:i]=pp.encoding(data.loc[:,i:i])
    
    for i in data.columns:
        if data[i].dtype=="object":
            if (is_float(data[i][0])==True):
                data[i]=data[i].astype(float)
            elif (is_int(data[i][0])==True):
                data[i]=data[i].astype(int)
            else:
                continue
    return data
    
def train(target,df):
    mod_type=str
    org=copy.deepcopy(df)
    data=preprocess(df,target)
    print(org[target].dtype)
    print("No. of columns =",data.shape[1])
    if org[target].dtype=='object':
        print("Classify")
        mod_type="Class"
        accuracy = acc_score(data,target)
        print(accuracy)
    else:
        print("Regressor")
        mod_type="Reg"
        accuracy = mse_score(data,target)
        print(accuracy)
    x=data.drop(columns=target)
    y=data[target]
    before_acc=max(accuracy.values, key=lambda x: x[1])[1]
    X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    before_feat=list(X_train.columns)
    if (len(X_train.columns)>10):
        print("Using Genetic Algorithm")
        prepared=pd.concat([X_train,Y_train],axis=1)
        Data_Batch=get_random_subsets(prepared,1)
        population, generations = run_evolution(
            populate_func=partial(generate_population, size=10, genome_length=len(data.columns)-1),
            fitness_func=partial(fitness_func,target_var=target,Data_Batch=Data_Batch),
            generation_limit=30
        )

        final_result =[0 for _ in range(len(population[0]))]
        for gen in population:
            for i,val in enumerate(gen):
                if val==0:
                    final_result[i]=final_result[i]-1

        chromo_df=best_feature(final_result,prepared.columns.to_list())
        print("Features selected using Genetic Algorithm:",chromo_df)
        X_train=X_train[chromo_df]
        X_test=X_test[chromo_df]
    else:
        print("Using Statistics Analysis")
        X_train,X_test, Y_train, Y_test=feature_select(org,data,target)
        print("Features Selected using Statistics:",X_train.columns)
        chromo_df=X_train.columns
        print("Best Features",chromo_df)
        X_train=X_train[chromo_df]
        X_test=X_test[chromo_df]
    print("X_train features :",X_train.columns)

    if mod_type=="Class":
        le=LabelEncoder()
        Y_train=le.fit_transform(Y_train)
        Y_test=le.fit_transform(Y_test)

    model_selection = ModelSelection(X_train, X_test, Y_train, Y_test)
    task = model_selection.detect_task()
    if task == 'classification':
        model_name=model_selection.choose_best_model()
    elif task == 'regression':
        model_name=model_selection.choose_best_model_regression()

    best_model=run_model(model_name)
    best_model.fit(X_train,Y_train)
    '''if org[target].dtype=='object':
        print("Classify")
        Baccuracy = class_model(X_train,X_test,Y_train,Y_test)
        print(Baccuracy)
    else:
        print("Regressor")
        Baccuracy = reg_model(X_train,X_test,Y_train,Y_test)
        print(Baccuracy)'''

    #best_model=max(Baccuracy.values, key=lambda x: x[1])[2]
    print(X_test.shape)
    print(Y_test.shape)
    y_pred=best_model.predict(X_test)
    

    if org[target].dtype=="object":
        select_feat_scr=accuracy_score(Y_test,y_pred)
        Y_train=le.inverse_transform(Y_train)
        y_pred=le.inverse_transform(y_pred)
        Y_test=le.inverse_transform(Y_test)
        labels=le.classes_
    else:
        select_feat_scr=r2_score(Y_test,y_pred)
    #Save Model
    print(select_feat_scr)
    joblib.dump(best_model,"./model/best_model.sav")
    if mod_type=="Class":
        return before_feat,before_acc,chromo_df,select_feat_scr,X_train,X_test[chromo_df],Y_train,Y_test,y_pred,labels
    else:
        return before_feat,before_acc,chromo_df,select_feat_scr,X_train,X_test[chromo_df],Y_train,Y_test,y_pred,None


def feature_select(org,data,target):
    fs=Feature_Selection()
    data,x,y=fs.constant_variance(data,0.2,target)
    if org[target].dtype=='object':
        mode='classification'
    else:
        mode='regression'
    x=fs.k_select_best(x,y,2,mode)
    x_train,x_test,y_train,y_test=fs.corr_drop(x,y)
    #print("SFS X_Train:",x_train)
    return x_train,x_test,y_train,y_test

def is_float(value):
    try:
        float_value = float(value)
        if isinstance(float_value, float):
            return True
        else:
            return False
    except ValueError:
        return False
    
def is_int(value):
    try:
        int_value = int(value)
        if isinstance(int_value, int):
            return True
        else:
            return False
    except ValueError:
        return False




'''if os.path.isfile("./model/preprocessed.csv") and os.path.isfile("./model/target.txt"):
    data=pd.read_csv("./model/preprocessed.csv")
    with open("./model/target.txt", "r") as file:
        target_val = file.read()

    TARGET_VAR = target_val.strip()'''


#data=pd.read_csv("./data/heart.csv")
#train("target",data)