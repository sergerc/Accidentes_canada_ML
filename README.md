# Análisis de la siniestralidad: 

:wave: Hola! Hemos realizado un análisis de [la base de datos de accidentes de Canadá](https://www.kaggle.com/tbsteal/canadian-car-accidents-19942014?select=drivingLegend.pdf).
El objetivo de este análisis es enfrentarnos a uno de los grandes problemas que tienen las aseguradoras año a año. ***¿Cuánto dinero deberíamos inmobilizar para hacer frente a los fallecimientos/hospitalizaciones de nuestros asegurados?***. Para responder a esta pregunta nos pondremos en la piel de un equipo de Data Science, que buscará abordar con sus conocimientos la mejor solución posible. Nuestra misión es elaborar un modelo que nos prediga la mortalidad de accidentes de tráfico en función de sus variables. 


### *Análisis EDA:* 
El primer paso que hemos realizado, ha sido un análisis exploratorio de datos (EDA). En este primer análisis hemos intentado dar explicación a las siguientes cuestiones: 

- ¿Qué tipos de vehículos (modelos, antigüedad, etc.) y conductores son más propensos a tener accidentes (acción correctiva en prima)? 
- ¿Qué tipos de vehículos (modelos, antigüedad, etc.) y conductores son menos propensos a tener accidentes (descuento en prima)?
- ¿Qué es lo que más contribuye a que existan fallecimientos en un accidente?
- Plus: complementar con datos abiertos de clima (aunque Canadá es muy grande) y de otra tipología, ¿hay algún tipo de relación con temperaturas medias, precipitación media
del día/mes, nieve...? ¿a más días festivos o de vacaciones, más accidentes? etc. 

Podréis encontrar estos análisis en los siguientes notebook/html:

  :page_facing_up: **[analisis_EDA_severidad](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/1.1.analisis_EDA_Severidad.ipynb)**
   
  :page_facing_up: **[analisis_EDA_Tratamiento_Medico](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/1.3.analisis_EDA_Agrupado_Severidad.%20.ipynb)**
  
  :page_facing_up: **[analisis EDA_Agrupado_Severidad](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/1.3.analisis_EDA_Agrupado_Severidad.%20.ipynb)**
  
 
### *Tratamiento de datos:*
Antes de realizar el modelo, debemos enfrentarnos al tratamiento del dataset, conocer y preprocesar cada variable, este trabajo lo hemos realizado en los siguientes notebooks: 

  :page_facing_up: **[Preprocesing. Limpieza del Dataset](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/2.0.1.%20%20Preprocesing.%20Limpieza%20Dataset.%20.ipynb)**

  :page_facing_up: **[Preprocesing. Análisis de variables](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/2.0.2.%20Preprocesing.%20Analisis%20de%20las%20variables.%20.ipynb)**

  :page_facing_up: **[Preprocesing. Split, NA, Outlier](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/2.0.3.%20Preprocesing.%20SPLIT%2C%20NA%2C%20OUTLIER.%20.ipynb)**
  
  :page_facing_up: **[Preprocesing. Seleccion de variables](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/2.0.4.%20Preprocesing.%20Seleccion%20de%20Variables..ipynb)**

### *Ajuste y evaluación de los modelos*: 
Una vez realizado el EDA y el preprocesamiento, tenemos dividida nuestra muestra entre train y test. Toca realizar los modelos de predicción del primero objetivo establecido, que es la detección de mortalidad en un accidente de trafico, por lo tanto nos encontramos ante un problema de clasificación (muerte, no muerte) del que queremos extraer una predicción en tanto porcentual.

#### 1. Arbol de decisión [:bookmark_tabs: Link](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/3.0.1.%20Modelos%20%20DTC.ipynb): 

- Hemos escogido el modelo de [`Árbol de decisión`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html). Sabemos de primera mano que su poder predictivo esta limitado con respecto a otros pero lo hemos escogido debido a su facil implementación. 
- Otra de las características por la que hemos elegido el arbol de decisión es porque suele ser un muy buen selector, es decir, nos puede indicar que variables son las que mas peso tienen con respecto a la variable objetivo. 

Un [`árbol de decisión`](https://www.datacamp.com/community/tutorials/decision-tree-classification-python) es una estructura de árbol similar a un diagrama de flujo donde un nodo interno representa una característica (o atributo), la rama representa una regla de decisión y cada nodo hoja representa el resultado. El nodo superior en un árbol de decisiones se conoce como nodo raíz. Aprende a particionar sobre la base del valor del atributo. Divide el árbol de manera recursiva y lo llama partición recursiva. Esta estructura similar a un diagrama de flujo le ayuda en la toma de decisiones. Es una visualización como un diagrama de flujo que imita fácilmente el pensamiento a nivel humano. Es por eso que los árboles de decisión son fáciles de entender e interpretar.

#### 2. Modelo de Regresión Lineal [:bookmark_tabs: Link](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/3.0.2.%20Modelos.%20Linear%20Regression.%20.ipynb)
[`Los modelos lineales`](https://www.ibm.com/docs/es/spss-modeler/SaaS?topic=node-linear-models) predicen un objetivo continuo basándose en relaciones lineales entre el objetivo y uno o más predictores.Los modelos lineales son relativamente simples y proporcionan una fórmula matemática fácil de interpretar para la puntuación. 

Hay una gran variedad de modelos lineales. De hecho, ya hemos comentado anteriormente dos de ellos en el notebook de selección de variables [`2.0.4.Preprocesing`](http://localhost:8888/notebooks/Desktop/CUNEF/Practica%20Machine%20Learning/project_template/notebooks/2.0.4.%20Preprocesing.%20Seleccion%20de%20Variables..ipynb)

[`El Modelo de regresión Logistica (RL)`](https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python) : La regresión logística es un método estadístico para predecir clases binarias. El resultado o la variable objetivo es de naturaleza dicotómica. Dicotómico significa que solo hay dos clases posibles. Es un caso especial de regresión lineal donde la variable objetivo es de naturaleza categórica. Utiliza un registro de probabilidades como variable dependiente. La regresión logística predice la probabilidad de ocurrencia de un evento binario.

#### 3. Random Forest [:bookmark_tabs: Link](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/3.0.3.%20Modelos%20RANDOM%20FOREST.ipynb)
Un [`random forest`](https://towardsdatascience.com/understanding-random-forest-58381e0602d2) consta de una gran cantidad de arboles de decisión individuales que operan en conjunto. Cada árbol individual en el bosque aleatorio escupe una predicción de clase y la clase con más votos se convierte en la predicción de nuestro modelo. 

#### 4. Modelo XGBoost [:bookmark_tabs: Link](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/3.0.4.%20Modelos.%20XGBOOST.ipynb)
[`XGBoost`](https://xgboost.readthedocs.io/en/latest/tutorials/model.html) Extreme Gradient Boosting es un algoritmo predictivo supervisado que utiliza el principio de boosting.

La idea detrás del boosting es generar múltiples modelos de predicción “débiles” secuenciualmente,y que cada uno de estos tome los resultados del modelo anterior, para generar un modelo más “fuerte”, con mejor poder predictivo y mayor estabilidad en sus resultados.

Para conseguir un modelo más fuerte, se emplea un algoritmo de optimización, este caso Gradient Descent (descenso de gradiente).

Durante el entrenamiento, los parámetros de cada modelo débil son ajustados iterativamente tratando de encontrar el mínimo de una función objetivo, que puede ser la proporción de error en la clasificación, el área bajo la curva (AUC), la raíz del error cuadrático medio (RMSE) o alguna otra.

Cada modelo es comparado con el anterior. Si un nuevo modelo tiene mejores resultados, entonces se toma este como base para realizar nuevas modificaciones. Si, por el contrario, tiene peores resultados, se regresa al mejor modelo anterior y se modifica ese de una manera diferente.

Este proceso se repite hasta llegar a un punto en el que la diferencia entre modelos consecutivos es insignificante, lo cual nos indica que hemos encontrado el mejor modelo posible, o cuando se llega al número de iteraciones máximas definido por el usuario.

#### 5. Modelo LightGBM [:bookmark_tabs: Link](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/3.0.5.%20Modelos%20LightGBM.ipynb)
[`LightGBM`](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc) es un algoritmo de refuerzo (o también de potenciación) de gradientes (gradient boosting) basado en modelos de árboles de decisión. Puede ser utilizado para la categorización, clasificación y muchas otras tareas de aprendizaje automático, en las que es necesario maximizar o minimizar una función objetivo mediante la técnica de gradient boosting, que consiste en combinar clasificadores sencillos, como por ejemplo árboles de decisión de profundidad limitada.

#### 6. Modelo Support Vector Machine [:bookmark_tabs: Link](http://localhost:8888/notebooks/Desktop/CUNEF/Practica_Machine_Learning/Practica-1-ML/notebooks/3.0.6.%20Modelos.%20SVM.ipynb)

[`SVM`]( https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python)ofrece una precisión muy alta en comparación con otros clasificadores, como la regresión logística y los árboles de decisión. `SVM` es un clasificador que separa los puntos de datos usando un hiperplano con el mayor margen posible, es por lo que tambien se le conoce como un clasificador discriminativo. SVM se encarga de encontrar el mejor hiperplano para clasificar. 

Parametros de [`SVM`](https://scikit-learn.org/stable/modules/svm.html):

- Kernel: Puede ser Lineal, Polynomial, y Radial Basis Function Kernel, este ultimo es el mas utilizado en problemas de clasificación y el que pensamos utilizar. Tambien nos encontramos con sigmoid para distribuciones de probabilidad. 

- Gamma: Parametro usado si utilizas 'rbf'. Puede ser scale, si se quiere utilizar la variación de los elementos. o auto para no utilizarla. Por defecto esta Scale. que es la que probaremos. 

- Clash_weight: El peso que se le atribuye a cada clase. Probaremos. 

#### 7. Modelo Voting Classifier [:bookmark_tabs: Link](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/3.0.7.%20Voting%20Classifier%20.ipynb)
La idea detrás de un ([`VotingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)) es combinar clasificadores conceptualmente diferentes y utilizar un voto mayoritario (*hard*) o las probabilidades promedio pronosticadas (*soft-voting*) para predecir las etiquetas. 

Este clasificador puede ser útil para un conjunto de modelos con un rendimiento igualmente bueno, a fin de equilibrar sus debilidades individuales.

#### MÉTODOS DE EVALUACIÓN DE MODELOS: 

Enumeramos a continuación las métricas que vamos a utilizar: 

- `Precisión:` Se define como la división de los verdaderos positivos (TP) de entre todos los positivos predichos (TP + FP).En nuestro caso en particular, será este dato el que este desequilibrado, ya que hay una desproporcionalidad de la muestra de muertes y no muertes. 

- `Recall:` Se define como todos los ejemplos predichos que pertenecen a una clase (TP) de entre todos los positivos predichos (TP + FN). 

- [`Accuracy`](https://machinelearningmastery.com/failure-of-accuracy-for-imbalanced-class-distributions/): Es una metrica que resume el rendimiento de un modelo de clasificación con el número de predicciones correctas dividido por el número total de predicciones. Debido a que la predicción no es una métrica fiable, esta métrica tampoco lo será.

- [`Matriz de confusión:`](https://medium.com/analytics-vidhya/accuracy-on-imbalanced-datasets-and-why-you-need-confusion-matrix-937613bf89bf) La métrica mas simple y al mismo tiempo la más efectiva para mirar el desempeño de los modelos en casos de dataset imbalanceados. Nos muestra la relacción que existe entre positivos acertados (TP), positivos fallados (FN), negativos acertados(TN), y negativos fallados (FP). 


- [`F1-Score:`](https://www.iartificial.net/precision-recall-f1-accuracy-en-clasificacion/) La F-Score combina las dos métricas de precisión y recall dentro de un mismo valor, añadiendo un parámetro `Beta`, que aplica un peso mayor a la precisión (Beta < 1) o un peso mayor al recall (Beta > 1). En nuestro caso, aplicaremos una F-1, ya que el peso lo aplicaremos en el modelo.

 $$2 \times\frac{\textrm{Precision} \times \textrm{Recall}}{\textrm{Precision} + \textrm{Recall}}$$
    
   
- [`La curva ROC:`](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/) Resume el rendimiento del modelo en la clase binaria positiva. Es una herramienta de diagnóstico popular para clasificadores en problemas de predicción binaria balanceados y desequilibrados por igual porque no está sesgada hacia la clase mayoritaria o minoritaria.


- [`Precision-Recall AUC:`](https://www.davidsbatista.net/blog/2018/08/19/NLP_Metrics/) Representa la precisión y el recall para diferentes umbrales de probabilidad. Es una tecnica muy efectiva para dataset desequilibrados debido a su enfoque en la clase minoritaria, es decir, en la mortalidad. 


- [`Cumulative Gain Curve`](https://towardsdatascience.com/meaningful-metrics-cumulative-gains-and-lyft-charts-7aac02fc5c14#:~:text=The%20cumulative%20gains%20curve%20is,target%20according%20to%20the%20model.) Evalua el rendimiento del modelo comparando los resultado con la selección aleatoria.


- [`Lift Curve`](https://towardsdatascience.com/meaningful-metrics-cumulative-gains-and-lyft-charts-7aac02fc5c14#:~:text=The%20cumulative%20gains%20curve%20is,target%20according%20to%20the%20model.) Mide la cantidad de ganancias que tiene nuestro modelo aplicandolo con respecto a la selección aleatorio.  Nos indica cuanto vale la pena implementar ese modelo. 


__El que mejor predicción hemos obtenido es el lightGBM, volvemos a realizar este modelo pero utilizando pipelines. ([Ver link](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/3.0.9.%20EXTRA%20LightGBM%20con%20Pipeline.ipynb))__

### INTERPRETABILIDAD: 

La [`interpretabilidad`](https://christophm.github.io/interpretable-ml-book/interpretability.html) en Machine Learning es el grado de comprensión que tiene un ser humano sobre la decisión que toma un algoritmo. Es decir, la explicación o el razonamiento que le da a la conclusión final. Muchas veces, un algoritmo puede predecir con fiabilidad un problema planteado con soluciones no contempladas por el ser humano a la hora de introducir los datos. Es por ello que queremos interpretar los datos que hemos obtenido de nuestro mejor modelo, [`LightGBM`](http://localhost:8888/notebooks/Desktop/CUNEF/Practica%20Machine%20Learning/project_template/notebooks/3.0.5.%20Modelos%20LightGBM.ipynb). 

Para interpretar nuestro modelo utilizaremos una combinación de los modelos de interpretación de [`ELI5`](https://eli5.readthedocs.io/en/latest/) y [`LIME`](https://lime.readthedocs.io/en/latest/) llamada [`SHAP o (SHapley Additive exPlanations)`](https://shap.readthedocs.io/en/latest/index.html) que contiene un enfoque de teoria de juegos para explicar el resultado de cualquier modelo de aprendizaje automático. 

El objetivo de `SHAP` es explicar la predicción de una instancia x calculando la contribución de cada característica a la predicción.

:page_facing_up: **[Puedes ver el notebook de trabajo aquí](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/3.0.8.%20Interpretabilidad.ipynb)**



# Trabajo realizado por: 

:bust_in_silhouette: SERGIO RANZ CASADO.

:mailbox: sergio.ranz@cunef.edu

:link: https://www.linkedin.com/in/sergio-ranz-casado-3318b713a/




:bust_in_silhouette: MARCOS MEDINA COGOLLUDO

:mailbox: marcos.medina@cunef.edu

:link: https://github.com/marcosmedina97
