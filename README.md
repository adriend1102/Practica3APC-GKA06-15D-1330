# Practica3 APC

## Miembros del grup:
Adrian Vargas Orellana: 
* Usuari GitHub: adriend1102
* Correu: adrian.v.o2002@gmail.com


## Estructura del repositorio:
* **data:** Directorio con la base de datos de las imagenes para entrenar y testear
* **chinese_mnist.csv:** CSV con los datos par relacionar el nombre suite, sample, code con el valor resultante de la siguiente forma

	-original name (example): Locate{1,3,4}.jpg  
	-index extracted: suite_id: 1, sample_id: 3, code: 4  
	-resulted file name: input_1_3_4.jpg 
* **chinese_mnist.ipynb:** Codi jupyter desenvolupat
* **chinese_mnist.py:** Codi python desenvolupat
* **PaginaPruebas.html:** Pagina para probar la CNN
* **Memoria-Practica3APC-GKA06-15D-1330.pdf:** Memoria del proyecto, con ejemplos visuales de los diferentes modelos usados
* **CHINESE- MNIST.pdf:** Presentacion de PPT del proyecto


## Ejecucion de la pagina web:
1. Ejecutar el programa **chinese_mnist.py** este programa contiene el modelo, se entrenara y ejecutara el servidor para poder usar la pagina web y poder hacer las peticiones web.
2. Cuando el servidor este levandato: Ejecutar el fichero **PaginaPruebas.html**
3. Darle uso al canvas para probar el modelo
