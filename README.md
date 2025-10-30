# Reto_09_Grupo_Amarillo
Cosas a tener en cuenta:
- Los scripts se deben de ejecutar en orden.
- Se necesitan los csv otorgados por los pm y lagun aro para poder resolver el reto.

## Datos
En estas carpetas estan o se tienen que ubicar todos los datos utilizados.

### Extra
Datos buscados por nosotros, el primero, *DatoshistóricosdelS&P500.csv*, son datos de la rentabilidad del S&P500 históricos mensuales que datan del 1984 hasta hoy. <br>
Por otro lado, *tasas.csv*, son datos de las letras y bonos del estado español desde el 2001.

### Originales
Aquí se tienen que incluir todos los datos que se han otorgado al principido del reto.

## Modelos adicionales
**(Se recomienda encarecidamente ejecutar después de *_01_Ingesta_Limpieza*)**
En este ipynb se encuentra la comprobación y evaluación de diferentes modelos, con ejecutar *_02_modelado* es suficiente, pero aquí se puede ver el proceso de selección seguido y la lógica aplicada.

## _01_Ingesta_Limpieza
En este script, se cargan y se limpian todos los datos que se van a utilizar en el reto.

## _02_Modelado
En este script se preparan los modelos que se van a utilizar para hacer los analisis en el reto. Al ejecutar el script se podrán ver las diferentes técnicas utilizadas para hacer los modelos que luego serán utilizados. Además, algunos modelos se ejecutan en el propio script.

## _03_Finanzas 
En este script se hace todo lo referente al apartado de finanzas del reto, todas las funciones y también se le da respuesta a los diferentes apartados del reto además de hacer los cálculos para las diferentes propuestas.

## Graficos.ipynb
Aquí se ubican los gráficos creados para luego ser utilizado en informe o presentaciones.

## Flask
### Boot de la API y cosas a tener en cuenta
Primero de todo, para que la API funcione, hay que abrir **EL PROYECTO ENTERO**, no solo la carpeta de la API, es decir, tienes que tener en tu entorno disponibles todas las carpetas del proyecto. <br>
Para continuar, el fichero para lanzar la API de flask es **server.py**. <br>
**¡¡IMPORTANTE!!**, hay unas almohadillas puestas en server.py **Hay que quitarlas la primera vez que se ejecuta y luego volverlas a poner** esto se hace para que se cargue la base de datos, por lo cual, si no tienes la base de datos cargada en el proyecto (debería de aparecer en la carpeta bbdd o sino en la carpeta de flask) **Quita los asteriscos**.

### Funcionalidad de la API y guia
Nada más lanzar la API te va a abrir la landing page, (cuyo html es landing.html). Desde ahí te va a aparecer un navegador, completamente funcional y unos botones, se pueden pulsar cualquiera de las dos alternativas para ir a las páginas que harán las funciones. <br>
En este caso lo voy a explicar por botones la funcionalidad de cada página, pero a tener en cuenta que, el navegador está en la misma disposición que los botones, es decir, el primer botón corresponde a la primera instancia del navegador, el segundo a la segunda... (a tener en cuenta que las instancias del navegador se cuentan sin tener en cuenta la que se llama _home_, ya que home te devuelve a la landing page.) <br>
A continuación, se explicará que pasará al pulsar cada botón:
- Consulta los pagos a hacer: Este link te lleva a la página donde está la primera función la cual te da los trabajadores, lo único que tienes que hacer es introducir un ID que esté en la base de datos y la API te redirigirá a una página donde puedes ver la información sobre el trabajador cuya ID has puesto. Para volver a la página principal, como siempre, con pulsar arriba en home basta.
- Añade nuevos casos: Pulsando este botón te redirgirá a la página donde se pueden ingresar nuevos trabajadores a la base de datos. Para ello tienes que ingresar su ID, Edad, Salario Anual, Fecha de ingreso y año de nacimiento, eso llamará a la función y te dará un dataset incluyendo los datos predichos.
- Elimina Trabajadores: Este botón te lleva a una página donde podrás eliminar IDs de trabajadores. Para ello simplemente tienes que introducir la ID del trabajador.
- Mira todos los trabajadores: Este es el último botón y te redirigirá a una tabla en la que podrás ver los datos de todos los trabajadores registrados en la base de datos.
