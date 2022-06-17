# DESARROLLO DE UN SISTEMA INTELIGENTE DE CONTROL DE DIABETES DE TIPO 1 BASADO EN MODELOS PREDICTIVOS.
======================================================================================================
##### Author: Juan José Martínez Cámara
##### Tutores: Carabias Orti, Julio José y Cañadas Quesada, Francisco.
======================================================================================================

###### *Resumen*
El número de pacientes que padecen de diabetes mellitus tipo 1 ha ido incrementando
significativamente durante los últimos 40 años. Es estimado que en 1980 había cerca de
108 millones de adultos con diabetes. Hoy en día se conoce que afecta a más de 460
millones de personas en el mundo (9.3% de la población mundial) con una prevalencia
global en crecimiento.

La diabetes mellitus tipo 1 (T1DM) se trata de la variante más peligrosa, donde el páncreas
no produce ninguna insulina de manera autosuficiente. La insulina es la proteína utilizada
para regular la glucosa en sangre (BG) por lo que los pacientes con dicha enfermedad
crónica han de inyectarse manualmente la insulina usando hasta el momento, dos tipos de
sistemas, inyección o bomba de insulina.

En los últimos años, potenciados por los avances en dispositivos portátiles y técnicas
basadas en big data, diferentes algoritmos de predicción de glucosa en sangre han sido
propuestos y validados en la práctica clínica. Entre ellos, el monitoreo continuo de glucosa
(CGM) se trata de una tecnología esencial para la medida de valores de BG en tiempo real.
El CGM ha proporcionado una gran cantidad de datos de BG con su creciente uso en
pacientes T1DM.

El tratamiento actual disponible para los pacientes T1DM puede llegar a ser engorroso e
implica muchas tomas de decisiones con respecto a la cantidad y frecuencia de inyección
de insulina a lo largo del día, resultando en muchas ocasiones bastante complicado para
el paciente controlar sus niveles de glucosa dentro de los rangos pautados por su médico.

Además, para comprender la complejidad de la diabetes, no solamente se han de tener en
cuenta los factores expuestos hasta ahora, si no que se han de tener en cuenta diversos
eventos tanto internos como externos, que pueden influir en el comportamiento de la
glucosa, tanto la ingesta de carbohidratos (CH), el esfuerzo físico, estrés psicológico y
enfermedades secundarias.

Para remediar el problema planteado, en este trabajo se van a usar técnicas de Deep
Learning (DL, por sus siglas en inglés) como forma de optimizar los bolos de insulina
inyectados por el paciente, en base a sus valores de glucosa. La mayoría de técnicas y
algoritmos de aprendizaje automático tienen la capacidad de predecir resultados futuros
con un conjunto de datos previos. Conocer cuáles podrían ser los niveles de glucosa en
sangre en un futuro muy cercano puede ser de gran relevancia para la toma de decisiones
del paciente en infinidad de momentos diarios.

Se usará una arquitectura especial de red neuronal recurrente artificial (LSTM) utilizada en
el campo del DL. A diferencia de las demás redes neuronales estándar, LSTM tiene
conexiones de retroalimentación, ya que puede procesar no simplemente puntos de datos
individuales, sino también secuencias completas de datos. Por ejemplo, LSTM es aplicable
en la detección de anomalías en el tráfico de red o IDS (sistemas de detección de intrusos).

El objetivo principal de este trabajo es ampliar la investigación actual sobre la predicción
de glucosa en sangre para pacientes T1DM utilizando DL, evaluando modelos basados en
redes LSTM y comparando a su vez con modelos matemáticos basados en modelos
estadísticos que utiliza variaciones y regresiones de datos estadísticos con el fin de
encontrar patrones para predicción hacia el futuro; además se desarrollará una aplicación
para pacientes T1DM en la que podrán obtener gráficas con predicciones de sus valores
de BG.
