# DESARROLLO DE UN SISTEMA INTELIGENTE DE CONTROL DE DIABETES DE TIPO 1 BASADO EN MODELOS PREDICTIVOS.
=======================================================================
##### Author: Juan José Martínez Cámara
##### Tutores: Carabias Orti, Julio José y Cañadas Quesada, Francisco.
=======================================================================

- 👋 Hi, I’m @juanjo-lab

#### *Contenido*

Se presenta en este repositorio el código desarrrollado durante el trabajo de fin de grado en el framework Django.

#### *Guía de instalación*
 
 1º- Se debe de crear un entorno virtual exclusivo para este proyecto.
 
 2º- Se deberá instalar las librerias empleadas, las cuales se encuentran en "pip install -r /path/to/requirements.txt".
 
 3º- Una vez con el entorno virtual creado con las librerias requeridas encontradas en el paso 2º, se accederá a la carpeta donde tenemos alojado manage.py      y con el entorno virtual activado "conda activate env", ejecutaremos "python manage.py runserver *IP del equipo:Puerto*", tanto la Ip del equipo            como el puerto seleccionado se deberán de configurar en settings.py "ALLOWED_HOSTS = ['*IP del equipo*']" (esta IP es para que nuestra aplicación          funcione a nivel local y nos permita acceder al servicio desde los distintos equipos que deseemos como un smartphone).

4º- Adicionalmente al paso 3, también para el documento settings.py será de conveniencia cambiar la cuenta de gmail usada para el envío y recepción de         mensajes, así como los directorios Path que se encuentren los ditintos archivos.py.

Una vez realizado todos los pasos previos, ya podrá disfrutar de la aplicación, como ejemplo, se le proporcionará una red ya entrenada y pacientes modelo.

#### *Correo de recepción*
Tras la creación dentro de la aplicacion, el informe tras la predicción que se crea, se deberá de ver tal que: 
![Informe enviado](https://github.com/juanjo-lab/tfg_deep_learning/blob/main/informe.png)

![Informe enviado](https://github.com/juanjo-lab/tfg_deep_learning/blob/main/Prediccion_12h.jpg)

#### *Notas*

- Se enviará un correo de confirmación tras la creación de la red neuronal, tras su reajuste así como el informe enseñado anteriormente tras cada predicción.

#### *Trabajo futuro*

En este capítulo se identifica el trabajo futuro para extender las contribuciones presentadas.
El proyecto futuro podría mejorar aún más los resultados arrojados sobre la predicción
glucémica.

###### *Preprocesado de datos*

El trabajo propuesto involucra los datos recogidos por el CGM, así como la temporalidad
del mismo. Es de interés la recopilación de información tanto pasada como futura sobre
comidas y ejercicio para estudio del comportamiento en el organismo del paciente, podría
ser mejorado para hacerlo más preciso, lo que haría más útil para los modelos de
predicción de glucosa, incluso, la información puede ser recogida usando una cámara del
teléfono del paciente. Entonces, el reconocimiento de patrones se podría usar para
determinar automáticamente el número de carbohidratos asociados con la comida. Aunque
se tengan resultados cercanos a la realidad, posiblemente mejorarán con lo expuesto
anteriormente puesto que el sistema, se convertirá más robusto ante situaciones
imprevistas.

Por otro lado, se puede trabajar también en la limpieza de los datos recogidos por el CGM,
puesto que, a pesar de ser precisos, hay casos de lecturas de punción digital ruidosas, si
se identificarán estás anomalías y se ignorasen, aumentaría la precisión. Usando una
función simple para estimar el peso de un óptimo local, si bien es útil, puede ser subóptimo.
Las mejoras podrían hacerse identificando casos específicos y modificando el peso óptimo
local basado sobre estos casos como un conjunto de reglas.

Por otro lado, en este proyecto, una de las limitaciones más importantes ha sido los datos
disponibles. Si se obtuviesen registros de un mayor grupo poblacional, sería interesante
realizar el estudio nuevamente; a tal efecto actualmente se está tramitando el acceso y
colaboración con diversos centros geriátricos con considerables pacientes parecientes de
T1DM dispuestos a proporcionar el consentimiento informado, así como asesoría por parte
de los mismos.

###### *Modelo de predicción de glucosa*

El modelo presentado en el capítulo 3 para la predicción de glucosa puede ser mejorado.
Puede ser interesante realizar una investigación análoga con otro modelo predictivo para
ver el comportamiento de otra red neuronal a comparación de la empleada. Por otro lado,
podría ser aún más interesante explotar la relación entre la insulina y los carbohidratos
ingeridos de manera dependiente, caracterizando la farmacocinética que presentan ambos
factores, pudiendo emplearse sobre el modelo SARIMAX haciendo más preciso el
pronóstico SARIMAX, a su vez, SARIMAX produce mejores resultados a corto plazo y
LSTM a largo plazo, con lo que, es de interés realizar un estudio de unificación de ambos
métodos, el comportamiento farmacocinético puede ser extraído del artículo [Chiara Dalla
Man et al. 2014], donde se caracteriza el simulador empleado para generar pacientes o
también se puede combinar modelos de datos fisiológicos como el presentado por Bunescu
en 2013. Esto es algo que podría mejorar aún más el rendimiento [R.Bunescu et al. 2013].


###### *Modelo de aplicación web*

Esta sección puede encontrarse en continua mejora, dado el gran abanico de posibilidades
que proporciona como línea de mejora.

Por ejemplo, se podrían migrar a los distintos dispositivos portátiles como smartphones con
conexión NFC y bluetooth en los que se podría recoger directamente los valores de glucosa
en tiempo real a nivel de usuario. De otro lado, también se podría reajustar la red de manera
automática cada semana en función del error cuadrático medio que se haya obtenido dicha
semana, además la proporción de información adicional como informes detallados sobre
consejos de mejora.
También se puede adaptar a los sistemas informáticos actuales presentes en los centros
de salud, así como la informatización actual de la recolecta de datos en línea de los
pacientes de larga estancia, para un mejor seguimiento de la enfermedad, creando así un
sistema dedicado para centros de salud.
