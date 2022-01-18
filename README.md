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

  :page_facing_up: [analisis_EDA_severidad](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/1.1.analisis_EDA_Severidad.ipynb)
   
  :page_facing_up: [analisis_EDA_Tratamiento_Medico](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/1.3.analisis_EDA_Agrupado_Severidad.%20.ipynb)
  
  :page_facing_up: [analisis EDA_Agrupado_Severidad](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/1.3.analisis_EDA_Agrupado_Severidad.%20.ipynb)
  
 
### *Tratamiento de datos:*
Antes de realizar el modelo, debemos enfrentarnos al tratamiento del dataset, conocer y preprocesar cada variable, este trabajo lo hemos realizado en los siguientes notebooks: 

  :page_facing_up: [Preprocesing. Limpieza del Dataset](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/2.0.1.%20%20Preprocesing.%20Limpieza%20Dataset.%20.ipynb)

  :page_facing_up: [Preprocesing. Análisis de variables](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/2.0.2.%20Preprocesing.%20Analisis%20de%20las%20variables.%20.ipynb)

  :page_facing_up: [Preprocesing. Split, NA, Outlier](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/2.0.3.%20Preprocesing.%20SPLIT%2C%20NA%2C%20OUTLIER.%20.ipynb)
  
  :page_facing_up: [Preprocesing. Seleccion de variables](https://github.com/sergerc/Accidentes_canada_ML/blob/main/notebooks/2.0.4.%20Preprocesing.%20Seleccion%20de%20Variables..ipynb)

### *Ajuste y evaluación de los modelos*: 








Trabajo realizado por: 

- NOMBRE: SERGIO RANZ CASADO.
- CORREO: sergio.ranz@cunef.edu
- GITHUB: https://github.com/sergerc


- NOMBRE: MARCOS MEDINA COGOLLUDO
- CORREO: marcos.medina@cunef.edu
- GITHUB: https://github.com/marcosmedina97

No hemos podido subir el csv debido a su peso. Dejamos aqui el enlace para su descarga: https://www.kaggle.com/tbsteal/canadian-car-accidents-19942014?select=drivingLegend.pdf

